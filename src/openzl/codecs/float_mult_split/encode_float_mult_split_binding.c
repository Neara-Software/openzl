// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/float_mult_split/encode_float_mult_split_binding.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "openzl/codecs/zl_float_mult_split.h"
#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
#include "openzl/shared/bits.h"
#include "openzl/shared/mem.h"

enum {
    FMS_MIN_SAMPLE                   = 10,
    FMS_MAX_SAMPLE                   = 3072,
    FMS_SAMPLE_RATIO                 = 40,
    FMS_REQUIRED_TRAILING_ZEROS      = 5,
    FMS_REQUIRED_PRECISION_BITS      = 6,
    FMS_GCD_TABLE_SIZE               = 2048,
    FMS_PRIM_TABLE_SIZE              = 8192,
    FMS_CLASSIC_MEMORIZABLE_BINS     = 256,
    FMS_EUCLIDEAN_MAX_ITERS          = 128,
};

static const double FMS_ZETA_OF_2                   = 1.6449340668482264;
static const double FMS_REQUIRED_BITS_SAVED_PER_NUM = 0.5;
static const double FMS_REQUIRED_TZ_FREQUENCY       = 0.5;
static const double FMS_EUCLIDEAN_MIN_SIMILAR_FRAC  = 0.001;
static const double FMS_EUCLIDEAN_SIMILAR_TOL       = 0.01;
static const double FMS_SNAP_THRESHOLD_ABSOLUTE     = 0.02;
static const double FMS_SNAP_THRESHOLD_DECIMAL_REL  = 0.01;

typedef struct {
    uint64_t key;
    uint32_t count;
    uint8_t used;
} FmsGcdEntry;

typedef struct {
    uint64_t key;
    uint32_t count;
    double bitsSaved;
    uint8_t used;
} FmsPrimEntry;

static inline uint64_t fms_splitmix64(uint64_t* state)
{
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z          = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z          = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline uint64_t fms_hash64(uint64_t x)
{
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline uint64_t fms_gcd_u64(uint64_t x, uint64_t y)
{
    while (y != 0) {
        uint64_t const t = x % y;
        x                = y;
        y                = t;
    }
    return x;
}

static inline uint64_t fms_triple_gcd_u64(uint64_t a, uint64_t b, uint64_t c)
{
    if (a > b) { uint64_t t = a; a = b; b = t; }
    if (b > c) { uint64_t t = b; b = c; c = t; }
    if (a > b) { uint64_t t = a; a = b; b = t; }
    return fms_gcd_u64(b - a, c - a);
}

static double fms_solve_concentrated_p(
        double lb,
        double ub,
        double gcdM1InvSq,
        double targetProb)
{
    double flb = lb * lb * lb + (1.0 - lb) * (1.0 - lb) * (1.0 - lb) * gcdM1InvSq - targetProb;
    double fub = ub * ub * ub + (1.0 - ub) * (1.0 - ub) * (1.0 - ub) * gcdM1InvSq - targetProb;
    if (flb > 0.0 || fub < 0.0) {
        return -1.0;
    }
    for (int iter = 0; iter < 64 && (ub - lb) > 1e-4 && (fub - flb) > 0.0; iter++) {
        double const lbProp = 0.001 + 0.998 * fub / (fub - flb);
        double const mid    = lbProp * lb + (1.0 - lbProp) * ub;
        double const fmid   = mid * mid * mid + (1.0 - mid) * (1.0 - mid) * (1.0 - mid) * gcdM1InvSq - targetProb;
        if (fmid < 0.0) {
            lb  = mid;
            flb = fmid;
        } else {
            ub  = mid;
            fub = fmid;
        }
    }
    return 0.5 * (lb + ub);
}

static double fms_score_triple_gcd(
        uint64_t gcdU64,
        uint32_t triplesWithGcd,
        size_t totalTriples)
{
    if (gcdU64 <= 1 || totalTriples == 0 || triplesWithGcd == 0) {
        return -1.0;
    }
    double const gcd   = (double)gcdU64;
    double const count = (double)triplesWithGcd;
    double const total = (double)totalTriples;

    double const probPerTriple        = count / total;
    double const naturalProbPerTriple = 1.0 / (FMS_ZETA_OF_2 * gcd * gcd);
    double const variance             = naturalProbPerTriple * (1.0 - naturalProbPerTriple) / total;
    double const stdev                = variance > 0.0 ? sqrt(variance) : 0.0;
    if (stdev > 0.0) {
        double const zScore = (probPerTriple - naturalProbPerTriple) / stdev;
        if (zScore < 3.0) {
            return -1.0;
        }
    } else if (probPerTriple <= naturalProbPerTriple) {
        return -1.0;
    }

    double const lcb = count - sqrt(count);
    if (lcb <= 0.0) {
        return -1.0;
    }
    double congruenceProb = FMS_ZETA_OF_2 * lcb / total;
    if (congruenceProb > 1.0) {
        congruenceProb = 1.0;
    }

    double const gcdM1      = gcd - 1.0;
    double const gcdM1InvSq = 1.0 / (gcdM1 * gcdM1);
    double const lb         = 1.0 / gcd;
    double ub               = cbrt(congruenceProb) + 1e-12;
    if (ub > 1.0) {
        ub = 1.0;
    }
    double const p = fms_solve_concentrated_p(lb, ub, gcdM1InvSq, congruenceProb);
    if (p < 0.0) {
        return -1.0;
    }

    double H = 0.0;
    if (p > 0.0 && p < 1.0) {
        H = -p * log2(p) - (1.0 - p) * log2((1.0 - p) / gcdM1);
    }
    double const bitsSaved = log2(gcd) - H;
    return bitsSaved >= FMS_REQUIRED_BITS_SAVED_PER_NUM ? bitsSaved : -1.0;
}

static uint64_t fms_choose_int_candidate_base(
        const uint64_t* intSample,
        size_t intSampleLen,
        uint64_t maxBase,
        FmsGcdEntry* gcdTable)
{
    if (intSampleLen < 12) {
        return 1;
    }
    memset(gcdTable, 0, FMS_GCD_TABLE_SIZE * sizeof(FmsGcdEntry));
    size_t const totalTriples = intSampleLen / 3;
    for (size_t t = 0; t < totalTriples; t++) {
        uint64_t const g = fms_triple_gcd_u64(
                intSample[3 * t], intSample[3 * t + 1], intSample[3 * t + 2]);
        if (g <= 1 || g > maxBase) {
            continue;
        }
        size_t slot = (size_t)fms_hash64(g) & (FMS_GCD_TABLE_SIZE - 1);
        for (size_t probe = 0; probe < FMS_GCD_TABLE_SIZE; probe++) {
            if (!gcdTable[slot].used) {
                gcdTable[slot].used  = 1;
                gcdTable[slot].key   = g;
                gcdTable[slot].count = 1;
                break;
            }
            if (gcdTable[slot].key == g) {
                gcdTable[slot].count++;
                break;
            }
            slot = (slot + 1) & (FMS_GCD_TABLE_SIZE - 1);
        }
    }

    uint64_t bestGcd = 1;
    double bestScore = -1.0;
    for (size_t slot = 0; slot < FMS_GCD_TABLE_SIZE; slot++) {
        if (!gcdTable[slot].used) {
            continue;
        }
        double const score = fms_score_triple_gcd(
                gcdTable[slot].key, gcdTable[slot].count, totalTriples);
        if (score > bestScore) {
            bestScore = score;
            bestGcd   = gcdTable[slot].key;
        }
    }
    return bestGcd > 1 ? bestGcd : 1;
}

/* ==== Float bit / ordering / int-float primitives ==== */

static inline uint32_t fms_f32_to_bits(float x)
{
    uint32_t b;
    memcpy(&b, &x, sizeof(b));
    return b;
}

static inline uint64_t fms_f64_to_bits(double x)
{
    uint64_t b;
    memcpy(&b, &x, sizeof(b));
    return b;
}

static inline uint32_t fms_f32_to_ordered(float x)
{
    uint32_t const b    = fms_f32_to_bits(x);
    uint32_t const mask = (uint32_t)(-(int32_t)(b >> 31)) | 0x80000000u;
    return b ^ mask;
}

static inline uint64_t fms_f64_to_ordered(double x)
{
    uint64_t const b    = fms_f64_to_bits(x);
    uint64_t const mask = (uint64_t)(-(int64_t)(b >> 63)) | 0x8000000000000000ULL;
    return b ^ mask;
}

static inline uint32_t fms_int_float_to_i32_bits(float x)
{
    uint32_t const b       = fms_f32_to_bits(x);
    uint32_t const sign    = b >> 31;
    uint32_t const absBits = b & 0x7FFFFFFFu;
    uint32_t const gpi     = 1u << 24;
    uint32_t const gpiBits = 0x4b800000u;
    uint32_t absInt;
    if (absBits < gpiBits) {
        float absF;
        memcpy(&absF, &absBits, sizeof(absF));
        absInt = (uint32_t)absF;
    } else {
        absInt = gpi + (absBits - gpiBits);
    }
    return absInt ^ (uint32_t)(-(int32_t)sign);
}

static inline uint64_t fms_int_float_to_i64_bits(double x)
{
    uint64_t const b       = fms_f64_to_bits(x);
    uint64_t const sign    = b >> 63;
    uint64_t const absBits = b & 0x7FFFFFFFFFFFFFFFULL;
    uint64_t const gpi     = 1ULL << 53;
    uint64_t const gpiBits = 0x4340000000000000ULL;
    uint64_t absInt;
    if (absBits < gpiBits) {
        double absD;
        memcpy(&absD, &absBits, sizeof(absD));
        absInt = (uint64_t)absD;
    } else {
        absInt = gpi + (absBits - gpiBits);
    }
    return absInt ^ (uint64_t)(-(int64_t)sign);
}

static inline int32_t fms_f32_exponent(float x)
{
    uint32_t const absBits = fms_f32_to_bits(x) & 0x7FFFFFFFu;
    return (int32_t)(absBits >> 23) - 127;
}

static inline int32_t fms_f64_exponent(double x)
{
    uint64_t const absBits = fms_f64_to_bits(x) & 0x7FFFFFFFFFFFFFFFULL;
    return (int32_t)(absBits >> 52) - 1023;
}

static inline uint32_t fms_f32_trailing_zeros(float x)
{
    uint32_t const b = fms_f32_to_bits(x);
    return b == 0u ? 32u : (uint32_t)ZL_ctz32(b);
}

static inline uint32_t fms_f64_trailing_zeros(double x)
{
    uint64_t const b = fms_f64_to_bits(x);
    return b == 0ULL ? 64u : (uint32_t)ZL_ctz64(b);
}

static int fms_cmp_f64(const void* a, const void* b)
{
    double const da = *(const double*)a;
    double const db = *(const double*)b;
    return (da > db) - (da < db);
}

static int fms_cmp_f32(const void* a, const void* b)
{
    float const fa = *(const float*)a;
    float const fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

/* ==== Typed float_mult mode selection & split (f32 and f64) ==== */

#define FMS_DEFINE_FLOAT_IMPL(                                                  \
        TAG, F, U, PREC_BITS, U_BITS, MAX_SAMP_EXPR, MAX_INT_BASE, EXP_FN, TZ_FN, LDEXP_FN, ROUND_FN, FABS_FN, ORD_FN, INTFLT_FN, CMP_FN, CLZ_FN) \
    typedef struct {                                                            \
        F base;                                                                 \
        F invBase;                                                              \
        int valid;                                                              \
    } FmsConfig_##TAG;                                                          \
                                                                                \
    static inline FmsConfig_##TAG fms_config_from_base_##TAG(F base)            \
    {                                                                           \
        FmsConfig_##TAG cfg = { 0, 0, 0 };                                      \
        if (isfinite(base) && base > (F)0) {                                    \
            F const invBase = (F)1 / base;                                      \
            if (isfinite(invBase) && invBase > (F)0) {                          \
                cfg.base    = base;                                             \
                cfg.invBase = invBase;                                          \
                cfg.valid   = 1;                                                \
            }                                                                   \
        }                                                                       \
        return cfg;                                                             \
    }                                                                           \
                                                                                \
    static inline FmsConfig_##TAG fms_config_from_inv_base_##TAG(F invBase)     \
    {                                                                           \
        FmsConfig_##TAG cfg = { 0, 0, 0 };                                      \
        if (isfinite(invBase) && invBase > (F)0) {                              \
            F const base = (F)1 / invBase;                                      \
            if (isfinite(base) && base > (F)0) {                                \
                cfg.base    = base;                                             \
                cfg.invBase = invBase;                                          \
                cfg.valid   = 1;                                                \
            }                                                                   \
        }                                                                       \
        return cfg;                                                             \
    }                                                                           \
                                                                                \
    static double fms_bits_saved_##TAG(                                         \
            FmsConfig_##TAG cfg,                                                \
            const F* sample,                                                    \
            size_t sampleLen,                                                   \
            FmsPrimEntry* primTable)                                            \
    {                                                                           \
        if (!cfg.valid || sampleLen == 0) {                                     \
            return -1.0;                                                        \
        }                                                                       \
        memset(primTable, 0, FMS_PRIM_TABLE_SIZE * sizeof(FmsPrimEntry));       \
        for (size_t i = 0; i < sampleLen; i++) {                                \
            F const x         = sample[i];                                      \
            F const mult      = ROUND_FN(x * cfg.invBase);                      \
            uint64_t const primary = (uint64_t)INTFLT_FN(mult);                 \
            uint32_t const uMultExp = (uint32_t)EXP_FN(mult);                   \
            uint32_t const interBaseBits =                                      \
                    uMultExp < (uint32_t)(PREC_BITS)                            \
                    ? (uint32_t)(PREC_BITS) - uMultExp                          \
                    : 0u;                                                       \
            U const approxOrd = ORD_FN(mult * cfg.base);                        \
            U const xOrd      = ORD_FN(x);                                      \
            U const absAdj    = xOrd > approxOrd ? xOrd - approxOrd             \
                                                 : approxOrd - xOrd;            \
            uint32_t const absAdjBitLen =                                       \
                    absAdj == (U)0 ? 0u : ((uint32_t)(U_BITS) - (uint32_t)CLZ_FN(absAdj)); \
            uint32_t const adjBits = 1u + 2u * absAdjBitLen;                    \
            double const saved     = (double)interBaseBits - (double)adjBits;   \
                                                                                \
            size_t slot = (size_t)fms_hash64(primary) & (FMS_PRIM_TABLE_SIZE - 1); \
            for (size_t probe = 0; probe < FMS_PRIM_TABLE_SIZE; probe++) {      \
                if (!primTable[slot].used) {                                    \
                    primTable[slot].used      = 1;                              \
                    primTable[slot].key       = primary;                        \
                    primTable[slot].count     = 1;                              \
                    primTable[slot].bitsSaved = saved;                          \
                    break;                                                      \
                }                                                               \
                if (primTable[slot].key == primary) {                           \
                    primTable[slot].count++;                                    \
                    primTable[slot].bitsSaved += saved;                         \
                    break;                                                      \
                }                                                               \
                slot = (slot + 1) & (FMS_PRIM_TABLE_SIZE - 1);                  \
            }                                                                   \
        }                                                                       \
                                                                                \
        size_t infrequentCutoff = sampleLen / FMS_CLASSIC_MEMORIZABLE_BINS;     \
        if (infrequentCutoff < 1) {                                             \
            infrequentCutoff = 1;                                               \
        }                                                                       \
        double totalSaved = 0.0;                                                \
        for (size_t slot = 0; slot < FMS_PRIM_TABLE_SIZE; slot++) {             \
            if (primTable[slot].used                                            \
                && primTable[slot].count <= infrequentCutoff) {                 \
                totalSaved += primTable[slot].bitsSaved;                        \
            }                                                                   \
        }                                                                       \
        double const perNum = totalSaved / (double)sampleLen;                   \
        return perNum >= FMS_REQUIRED_BITS_SAVED_PER_NUM ? perNum : -1.0;       \
    }                                                                           \
                                                                                \
    static FmsConfig_##TAG fms_choose_by_tz_##TAG(                              \
            const F* sample,                                                    \
            size_t sampleLen,                                                   \
            uint64_t* intSample,                                                \
            FmsGcdEntry* gcdTable)                                              \
    {                                                                           \
        FmsConfig_##TAG none = { 0, 0, 0 };                                     \
        int32_t k            = INT32_MAX;                                       \
        size_t count         = 0;                                               \
        for (size_t i = 0; i < sampleLen; i++) {                                \
            F const x         = sample[i];                                      \
            uint32_t const tz = TZ_FN(x);                                       \
            if (x != (F)0 && tz >= FMS_REQUIRED_TRAILING_ZEROS) {               \
                uint32_t const satSub = tz < (uint32_t)(PREC_BITS)              \
                        ? (uint32_t)(PREC_BITS) - tz                            \
                        : 0u;                                                   \
                int32_t const kPrime  = EXP_FN(x) - (int32_t)satSub;            \
                count++;                                                        \
                if (kPrime < k) {                                               \
                    k = kPrime;                                                 \
                }                                                               \
            }                                                                   \
        }                                                                       \
        size_t requiredSamples =                                                \
                (size_t)ceil((double)sampleLen * FMS_REQUIRED_TZ_FREQUENCY);    \
        if (requiredSamples < FMS_MIN_SAMPLE) {                                 \
            requiredSamples = FMS_MIN_SAMPLE;                                   \
        }                                                                       \
        if (count < requiredSamples) {                                          \
            return none;                                                        \
        }                                                                       \
                                                                                \
        size_t intSampleLen        = 0;                                         \
        uint32_t const lshift      = (uint32_t)(U_BITS) - (uint32_t)(PREC_BITS) - 1u; \
        U const explicitMantissa   = (U)1 << ((uint32_t)(U_BITS) - 1u);         \
        for (size_t i = 0; i < sampleLen; i++) {                                \
            F const x         = sample[i];                                      \
            int32_t const exp = EXP_FN(x);                                      \
            uint32_t const tz = TZ_FN(x);                                       \
            uint32_t const satSub = tz < (uint32_t)(PREC_BITS)                  \
                    ? (uint32_t)(PREC_BITS) - tz                                \
                    : 0u;                                                       \
            int32_t const kPrime = exp - (int32_t)satSub;                       \
            if (kPrime >= k && exp < k + (int32_t)(U_BITS)) {                   \
                uint32_t const rshift = (uint32_t)(U_BITS) - 1u - (uint32_t)(exp - k); \
                U bits;                                                         \
                memcpy(&bits, &x, sizeof(bits));                                \
                U const lshifted    = (bits << lshift) | explicitMantissa;      \
                U const multipleOfK = lshifted >> rshift;                       \
                intSample[intSampleLen++] = (uint64_t)multipleOfK;              \
            }                                                                   \
        }                                                                       \
        if (intSampleLen < requiredSamples) {                                   \
            return none;                                                        \
        }                                                                       \
        uint64_t const intBase = fms_choose_int_candidate_base(                 \
                intSample, intSampleLen, (uint64_t)(MAX_INT_BASE), gcdTable);   \
        F const base = LDEXP_FN((F)intBase, k);                                 \
        return fms_config_from_base_##TAG(base);                                \
    }                                                                           \
                                                                                \
    static F fms_approx_pair_gcd_##TAG(F greater, F lesser)                     \
    {                                                                           \
        if (lesser <= LDEXP_FN(greater, -(PREC_BITS - FMS_REQUIRED_PRECISION_BITS)) \
            || lesser == greater) {                                             \
            return (F)0;                                                        \
        }                                                                       \
        F const machineEps = LDEXP_FN((F)1, -(PREC_BITS));                      \
        F pv = greater, pe = (F)0;                                              \
        F lv = lesser, le = (F)0;                                               \
        for (int iter = 0; iter < FMS_EUCLIDEAN_MAX_ITERS; iter++) {            \
            F const prev  = pv;                                                 \
            F const ratio = ROUND_FN(pv / lv);                                  \
            pe += ratio * le + pv * machineEps;                                 \
            pv = FABS_FN(pv - ratio * lv);                                      \
            if (pv <= LDEXP_FN(prev, -16) || pv <= pe) {                        \
                return lv;                                                      \
            }                                                                   \
            if (pv <= LDEXP_FN(greater, -(PREC_BITS - FMS_REQUIRED_PRECISION_BITS)) \
                || pv <= LDEXP_FN(pe, FMS_REQUIRED_PRECISION_BITS)) {           \
                return (F)0;                                                    \
            }                                                                   \
            F tv = pv; pv = lv; lv = tv;                                        \
            F te = pe; pe = le; le = te;                                        \
        }                                                                       \
        return (F)0;                                                            \
    }                                                                           \
                                                                                \
    static FmsConfig_##TAG fms_choose_by_euclidean_##TAG(                       \
            const F* sample,                                                    \
            size_t sampleLen,                                                   \
            F* gcds)                                                            \
    {                                                                           \
        FmsConfig_##TAG none = { 0, 0, 0 };                                     \
        if (sampleLen < 2) {                                                    \
            return none;                                                        \
        }                                                                       \
        size_t gcdsLen = 0;                                                     \
        for (size_t i = 0; i + 1 < sampleLen; i += 2) {                         \
            F const a       = sample[i];                                        \
            F const b       = sample[i + 1];                                    \
            F const greater = a > b ? a : b;                                    \
            F const lesser  = a < b ? a : b;                                    \
            F const g       = fms_approx_pair_gcd_##TAG(greater, lesser);       \
            if (g > (F)0 && isfinite(g)) {                                      \
                gcds[gcdsLen++] = g;                                            \
            }                                                                   \
        }                                                                       \
        size_t const requiredPairs =                                            \
                1u + (size_t)ceil((double)sampleLen * FMS_EUCLIDEAN_MIN_SIMILAR_FRAC); \
        if (gcdsLen < requiredPairs) {                                          \
            return none;                                                        \
        }                                                                       \
        qsort(gcds, gcdsLen, sizeof(F), CMP_FN);                                \
        static const double kPercentiles[3] = { 0.1, 0.3, 0.5 };                \
        F candidate = (F)0;                                                     \
        int found   = 0;                                                        \
        for (size_t pIdx = 0; pIdx < 3; pIdx++) {                               \
            size_t idx = (size_t)(kPercentiles[pIdx] * (double)gcdsLen);        \
            if (idx >= gcdsLen) {                                               \
                idx = gcdsLen - 1;                                              \
            }                                                                   \
            F const cand  = gcds[idx];                                          \
            F const tol   = (F)FMS_EUCLIDEAN_SIMILAR_TOL * cand;                \
            size_t similar = 0;                                                 \
            for (size_t j = 0; j < gcdsLen; j++) {                              \
                if (FABS_FN(gcds[j] - cand) < tol) {                            \
                    similar++;                                                  \
                }                                                               \
            }                                                                   \
            if (similar >= requiredPairs) {                                     \
                candidate = cand;                                               \
                found     = 1;                                                  \
                break;                                                          \
            }                                                                   \
        }                                                                       \
        if (!found || !(candidate > (F)0) || !isfinite(candidate)) {            \
            return none;                                                        \
        }                                                                       \
                                                                                \
        /* center_sample_base */                                                \
        F const invCand   = (F)1 / candidate;                                   \
        F tweakSum        = (F)0;                                               \
        F tweakWeight     = (F)0;                                               \
        if (isfinite(invCand) && invCand > (F)0) {                              \
            for (size_t i = 0; i < sampleLen; i++) {                            \
                F const x    = sample[i];                                       \
                F const mult = ROUND_FN(x * invCand);                           \
                if (mult != (F)0) {                                             \
                    int32_t const multExp = EXP_FN(mult);                       \
                    if (multExp < (PREC_BITS)) {                                \
                        F const overshoot = (mult * candidate) - x;             \
                        F const weight    = (F)((PREC_BITS) - multExp);         \
                        tweakSum += weight * (overshoot / mult);                \
                        tweakWeight += weight;                                  \
                    }                                                           \
                }                                                               \
            }                                                                   \
        }                                                                       \
        F centered = candidate;                                                 \
        if (tweakWeight > (F)0) {                                               \
            F const adjCand = candidate - tweakSum / tweakWeight;               \
            if (isfinite(adjCand) && adjCand > (F)0) {                          \
                centered = adjCand;                                             \
            }                                                                   \
        }                                                                       \
                                                                                \
        /* snap_to_int_reciprocal */                                            \
        F const invBase = (F)1 / centered;                                      \
        if (!isfinite(invBase) || !(invBase > (F)0)) {                          \
            return none;                                                        \
        }                                                                       \
        F const roundInvBase   = ROUND_FN(invBase);                             \
        F const decimalInvBase = (F)pow(10.0, round(log10((double)invBase)));   \
        if (roundInvBase >= (F)1 && isfinite(roundInvBase)                      \
            && FABS_FN(invBase - roundInvBase) < (F)FMS_SNAP_THRESHOLD_ABSOLUTE) { \
            return fms_config_from_inv_base_##TAG(roundInvBase);                \
        } else if (                                                             \
                decimalInvBase > (F)0 && isfinite(decimalInvBase)               \
                && FABS_FN(invBase - decimalInvBase) / invBase                  \
                        < (F)FMS_SNAP_THRESHOLD_DECIMAL_REL) {                  \
            return fms_config_from_inv_base_##TAG(decimalInvBase);              \
        } else {                                                                \
            return fms_config_from_base_##TAG(centered);                        \
        }                                                                       \
    }                                                                           \
                                                                                \
    static FmsConfig_##TAG fms_choose_config_##TAG(                             \
            ZL_Encoder* eictx,                                                  \
            const void* srcVoid,                                                \
            size_t n)                                                           \
    {                                                                           \
        FmsConfig_##TAG best = { 0, 0, 0 };                                     \
        if (n < FMS_MIN_SAMPLE) {                                               \
            return best;                                                        \
        }                                                                       \
        size_t sampleN = FMS_MIN_SAMPLE + (n - FMS_MIN_SAMPLE) / FMS_SAMPLE_RATIO; \
        if (sampleN > FMS_MAX_SAMPLE) {                                         \
            sampleN = FMS_MAX_SAMPLE;                                           \
        }                                                                       \
        if (sampleN > n) {                                                      \
            sampleN = n;                                                        \
        }                                                                       \
                                                                                \
        F* const sample = (F*)ZL_Encoder_getScratchSpace(                       \
                eictx, sampleN * sizeof(F));                                    \
        uint64_t* const intSample = (uint64_t*)ZL_Encoder_getScratchSpace(      \
                eictx, sampleN * sizeof(uint64_t));                             \
        F* const gcds = (F*)ZL_Encoder_getScratchSpace(                         \
                eictx, (sampleN / 2u + 1u) * sizeof(F));                        \
        FmsGcdEntry* const gcdTable =                                           \
                (FmsGcdEntry*)ZL_Encoder_getScratchSpace(                       \
                        eictx, FMS_GCD_TABLE_SIZE * sizeof(FmsGcdEntry));       \
        FmsPrimEntry* const primTable =                                         \
                (FmsPrimEntry*)ZL_Encoder_getScratchSpace(                      \
                        eictx, FMS_PRIM_TABLE_SIZE * sizeof(FmsPrimEntry));     \
        if (sample == NULL || intSample == NULL || gcds == NULL                 \
            || gcdTable == NULL || primTable == NULL) {                         \
            return best;                                                        \
        }                                                                       \
                                                                                \
        const U* const src = (const U*)srcVoid;                                 \
        F const maxForSamp = (F)(MAX_SAMP_EXPR);                                \
        uint64_t rng       = 0x517cc1b727220a95ULL;                             \
        size_t validN      = 0;                                                 \
        for (size_t i = 0; i < sampleN; i++) {                                  \
            size_t const lo   = (i * n) / sampleN;                              \
            size_t const hi   = ((i + 1) * n) / sampleN;                        \
            size_t const span = hi > lo ? hi - lo : 1;                          \
            size_t const idx  = lo + (size_t)(fms_splitmix64(&rng) % span);     \
            F val;                                                              \
            memcpy(&val, &src[idx], sizeof(val));                               \
            if (isnormal(val)) {                                                \
                F const absVal = FABS_FN(val);                                  \
                if (absVal <= maxForSamp) {                                     \
                    sample[validN++] = absVal;                                  \
                }                                                               \
            }                                                                   \
        }                                                                       \
        if (validN < FMS_MIN_SAMPLE) {                                          \
            return best;                                                        \
        }                                                                       \
        for (size_t i = validN - 1; i > 0; i--) {                               \
            size_t const j = (size_t)(fms_splitmix64(&rng) % (i + 1));          \
            F const tmp    = sample[i];                                         \
            sample[i]      = sample[j];                                         \
            sample[j]      = tmp;                                               \
        }                                                                       \
                                                                                \
        FmsConfig_##TAG const tzCfg = fms_choose_by_tz_##TAG(                   \
                sample, validN, intSample, gcdTable);                           \
        double const tzScore =                                                  \
                fms_bits_saved_##TAG(tzCfg, sample, validN, primTable);         \
                                                                                \
        FmsConfig_##TAG const eucCfg = fms_choose_by_euclidean_##TAG(           \
                sample, validN, gcds);                                          \
        double const eucScore =                                                 \
                fms_bits_saved_##TAG(eucCfg, sample, validN, primTable);        \
                                                                                \
        if (tzScore >= eucScore && tzScore >= FMS_REQUIRED_BITS_SAVED_PER_NUM) { \
            return tzCfg;                                                       \
        }                                                                       \
        if (eucScore >= FMS_REQUIRED_BITS_SAVED_PER_NUM) {                      \
            return eucCfg;                                                      \
        }                                                                       \
        return best;                                                            \
    }                                                                           \
                                                                                \
    static void fms_split_##TAG(                                                \
            U* restrict multDst,                                                \
            U* restrict adjDst,                                                 \
            const U* restrict src,                                              \
            size_t n,                                                           \
            FmsConfig_##TAG cfg)                                                \
    {                                                                           \
        F const base    = cfg.base;                                             \
        F const invBase = cfg.invBase;                                          \
        for (size_t i = 0; i < n; i++) {                                        \
            F num;                                                              \
            memcpy(&num, &src[i], sizeof(num));                                 \
            F const mult   = ROUND_FN(num * invBase);                           \
            multDst[i]     = INTFLT_FN(mult);                                   \
            F const approx = mult * base;                                       \
            adjDst[i]      = (U)(ORD_FN(num) - ORD_FN(approx));                 \
        }                                                                       \
    }

FMS_DEFINE_FLOAT_IMPL(
        f32,
        float,
        uint32_t,
        23,
        32,
        FLT_MAX * 0.5f,
        UINT32_MAX,
        fms_f32_exponent,
        fms_f32_trailing_zeros,
        ldexpf,
        roundf,
        fabsf,
        fms_f32_to_ordered,
        fms_int_float_to_i32_bits,
        fms_cmp_f32,
        ZL_clz32)

FMS_DEFINE_FLOAT_IMPL(
        f64,
        double,
        uint64_t,
        52,
        64,
        DBL_MAX * 0.5,
        UINT64_MAX,
        fms_f64_exponent,
        fms_f64_trailing_zeros,
        ldexp,
        round,
        fabs,
        fms_f64_to_ordered,
        fms_int_float_to_i64_bits,
        fms_cmp_f64,
        ZL_clz64)

ZL_Report EI_float_mult_split(
        ZL_Encoder* eictx,
        const ZL_Input* ins[],
        size_t nbIns)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(eictx);
    ZL_ASSERT_EQ(nbIns, 1);
    const ZL_Input* const in = ins[0];
    ZL_ASSERT_EQ(ZL_Input_type(in), ZL_Type_numeric);

    size_t const eltWidth = ZL_Input_eltWidth(in);
    size_t const n        = ZL_Input_numElts(in);
    ZL_ERR_IF_NOT(eltWidth == 4 || eltWidth == 8, node_invalid_input);

    unsigned const eltWidthLog2 = (unsigned)ZL_nextPow2(eltWidth);
    const void* const srcPtr    = ZL_Input_ptr(in);

    int hasBase           = 0;
    FmsConfig_f32 cfg32   = { 0, 0, 0 };
    FmsConfig_f64 cfg64   = { 0, 0, 0 };
    if (n > 0) {
        if (eltWidth == 4) {
            cfg32   = fms_choose_config_f32(eictx, srcPtr, n);
            hasBase = cfg32.valid;
        } else {
            cfg64   = fms_choose_config_f64(eictx, srcPtr, n);
            hasBase = cfg64.valid;
        }
    }

    ZL_Output* const multOut =
            ZL_Encoder_createTypedStream(eictx, 0, n, eltWidth);
    ZL_Output* const adjOut = ZL_Encoder_createTypedStream(
            eictx, 1, hasBase ? n : 0, eltWidth);
    ZL_ERR_IF(multOut == NULL || adjOut == NULL, allocation);

    uint8_t hdr[1 + 8];
    hdr[0]          = (uint8_t)((eltWidthLog2 & 3u) | ((unsigned)(hasBase & 1) << 2));
    size_t hdrSize  = 1u;
    if (hasBase) {
        if (eltWidth == 4) {
            ZL_writeLE32(hdr + 1, fms_f32_to_bits(cfg32.base));
            hdrSize += 4u;
        } else {
            ZL_writeLE64(hdr + 1, fms_f64_to_bits(cfg64.base));
            hdrSize += 8u;
        }
    }
    ZL_Encoder_sendCodecHeader(eictx, hdr, hdrSize);

    if (n > 0) {
        void* const multPtr = ZL_Output_ptr(multOut);
        if (!hasBase) {
            memcpy(multPtr, srcPtr, n * eltWidth);
        } else {
            void* const adjPtr = ZL_Output_ptr(adjOut);
            if (eltWidth == 4) {
                fms_split_f32(
                        (uint32_t*)multPtr,
                        (uint32_t*)adjPtr,
                        (const uint32_t*)srcPtr,
                        n,
                        cfg32);
            } else {
                fms_split_f64(
                        (uint64_t*)multPtr,
                        (uint64_t*)adjPtr,
                        (const uint64_t*)srcPtr,
                        n,
                        cfg64);
            }
        }
    }

    ZL_ERR_IF_ERR(ZL_Output_commit(multOut, n));
    ZL_ERR_IF_ERR(ZL_Output_commit(adjOut, hasBase ? n : 0));
    return ZL_returnSuccess();
}
