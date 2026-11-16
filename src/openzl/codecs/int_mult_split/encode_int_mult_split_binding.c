// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/int_mult_split/encode_int_mult_split_binding.h"

#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "openzl/codecs/zl_int_mult_split.h"
#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
#include "openzl/shared/bits.h"
#include "openzl/shared/varint.h"

enum {
    IMS_MIN_SAMPLE                = 12,
    IMS_MAX_SAMPLE                = 3072,
    IMS_SAMPLE_RATIO              = 40,
    IMS_GCD_TABLE_SIZE            = 2048,
    IMS_PRIM_TABLE_SIZE           = 8192,
    IMS_CLASSIC_MEMORIZABLE_BINS  = 256,
};

static const double IMS_ZETA_OF_2                     = 1.6449340668482264;
static const double IMS_REQUIRED_BITS_SAVED_PER_NUM   = 0.5;

typedef struct {
    uint64_t key;
    uint32_t count;
    uint8_t used;
} ImsGcdEntry;

typedef struct {
    uint64_t key;
    uint32_t count;
    double bitsSaved;
    uint8_t used;
} ImsPrimEntry;

static inline uint64_t ims_splitmix64(uint64_t* state)
{
    uint64_t z = (*state += 0x9e3779b97f4a7c15ULL);
    z          = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z          = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

static inline uint64_t ims_hash64(uint64_t x)
{
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

static inline uint64_t ims_gcd(uint64_t x, uint64_t y)
{
    while (y != 0) {
        uint64_t const t = x % y;
        x                = y;
        y                = t;
    }
    return x;
}

static inline uint64_t ims_triple_gcd(
        uint64_t ua,
        uint64_t ub,
        uint64_t uc,
        int signedMode)
{
    if (signedMode) {
        int64_t a = (int64_t)ua;
        int64_t b = (int64_t)ub;
        int64_t c = (int64_t)uc;
        if (a > b) { int64_t t = a; a = b; b = t; }
        if (b > c) { int64_t t = b; b = c; c = t; }
        if (a > b) { int64_t t = a; a = b; b = t; }
        uint64_t const d1 = (uint64_t)b - (uint64_t)a;
        uint64_t const d2 = (uint64_t)c - (uint64_t)a;
        return ims_gcd(d1, d2);
    } else {
        uint64_t a = ua;
        uint64_t b = ub;
        uint64_t c = uc;
        if (a > b) { uint64_t t = a; a = b; b = t; }
        if (b > c) { uint64_t t = b; b = c; c = t; }
        if (a > b) { uint64_t t = a; a = b; b = t; }
        return ims_gcd(b - a, c - a);
    }
}

static double ims_solve_concentrated_p(
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

static double ims_score_triple_gcd(
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
    double const naturalProbPerTriple = 1.0 / (IMS_ZETA_OF_2 * gcd * gcd);
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
    double congruenceProb = IMS_ZETA_OF_2 * lcb / total;
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
    double const p = ims_solve_concentrated_p(lb, ub, gcdM1InvSq, congruenceProb);
    if (p < 0.0) {
        return -1.0;
    }

    double H = 0.0;
    if (p > 0.0 && p < 1.0) {
        H = -p * log2(p) - (1.0 - p) * log2((1.0 - p) / gcdM1);
    }
    double const bitsSaved = log2(gcd) - H;
    return bitsSaved >= IMS_REQUIRED_BITS_SAVED_PER_NUM ? bitsSaved : -1.0;
}

static inline uint64_t ims_read_elem_u64(
        const void* src,
        size_t eltWidth,
        size_t idx,
        int signedMode)
{
    if (signedMode) {
        switch (eltWidth) {
            case 1: return (uint64_t)(int64_t)((const int8_t*)src)[idx];
            case 2: return (uint64_t)(int64_t)((const int16_t*)src)[idx];
            case 4: return (uint64_t)(int64_t)((const int32_t*)src)[idx];
            case 8: return (uint64_t)((const int64_t*)src)[idx];
            default: return 0;
        }
    } else {
        switch (eltWidth) {
            case 1: return (uint64_t)((const uint8_t*)src)[idx];
            case 2: return (uint64_t)((const uint16_t*)src)[idx];
            case 4: return (uint64_t)((const uint32_t*)src)[idx];
            case 8: return ((const uint64_t*)src)[idx];
            default: return 0;
        }
    }
}

static uint64_t ims_max_base_for_width(size_t eltWidth, int signedMode)
{
    if (signedMode) {
        switch (eltWidth) {
            case 1: return (uint64_t)INT8_MAX;
            case 2: return (uint64_t)INT16_MAX;
            case 4: return (uint64_t)INT32_MAX;
            case 8: return (uint64_t)INT64_MAX;
            default: return 1;
        }
    } else {
        switch (eltWidth) {
            case 1: return (uint64_t)UINT8_MAX;
            case 2: return (uint64_t)UINT16_MAX;
            case 4: return (uint64_t)UINT32_MAX;
            case 8: return UINT64_MAX;
            default: return 1;
        }
    }
}

static uint64_t ims_choose_base(
        ZL_Encoder* eictx,
        const void* src,
        size_t n,
        size_t eltWidth,
        int signedMode)
{
    if (n < 18) {
        return 1;
    }

    size_t sampleN = IMS_MIN_SAMPLE + (n - 10) / IMS_SAMPLE_RATIO;
    if (sampleN > IMS_MAX_SAMPLE) {
        sampleN = IMS_MAX_SAMPLE;
    }
    sampleN -= sampleN % 3;
    if (sampleN < IMS_MIN_SAMPLE) {
        sampleN = IMS_MIN_SAMPLE;
    }
    if (sampleN > n) {
        sampleN = n - (n % 3);
    }
    if (sampleN < IMS_MIN_SAMPLE) {
        return 1;
    }

    uint64_t* const sample =
            (uint64_t*)ZL_Encoder_getScratchSpace(eictx, sampleN * sizeof(uint64_t));
    ImsGcdEntry* const gcdTable =
            (ImsGcdEntry*)ZL_Encoder_getScratchSpace(
                    eictx, IMS_GCD_TABLE_SIZE * sizeof(ImsGcdEntry));
    ImsPrimEntry* const primTable =
            (ImsPrimEntry*)ZL_Encoder_getScratchSpace(
                    eictx, IMS_PRIM_TABLE_SIZE * sizeof(ImsPrimEntry));
    if (sample == NULL || gcdTable == NULL || primTable == NULL) {
        return 1;
    }
    memset(gcdTable, 0, IMS_GCD_TABLE_SIZE * sizeof(ImsGcdEntry));
    memset(primTable, 0, IMS_PRIM_TABLE_SIZE * sizeof(ImsPrimEntry));

    uint64_t rng = 0x517cc1b727220a95ULL;
    for (size_t i = 0; i < sampleN; i++) {
        size_t const lo    = (i * n) / sampleN;
        size_t const hi    = ((i + 1) * n) / sampleN;
        size_t const span  = hi > lo ? hi - lo : 1;
        size_t const idx   = lo + (size_t)(ims_splitmix64(&rng) % span);
        sample[i]          = ims_read_elem_u64(src, eltWidth, idx, signedMode);
    }
    for (size_t i = sampleN - 1; i > 0; i--) {
        size_t const j  = (size_t)(ims_splitmix64(&rng) % (i + 1));
        uint64_t const tmp = sample[i];
        sample[i]       = sample[j];
        sample[j]       = tmp;
    }

    uint64_t const maxBase   = ims_max_base_for_width(eltWidth, signedMode);
    size_t const totalTriples = sampleN / 3;
    for (size_t t = 0; t < totalTriples; t++) {
        uint64_t const g = ims_triple_gcd(
                sample[3 * t],
                sample[3 * t + 1],
                sample[3 * t + 2],
                signedMode);
        if (g <= 1 || g > maxBase) {
            continue;
        }
        size_t slot = (size_t)ims_hash64(g) & (IMS_GCD_TABLE_SIZE - 1);
        for (size_t probe = 0; probe < IMS_GCD_TABLE_SIZE; probe++) {
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
            slot = (slot + 1) & (IMS_GCD_TABLE_SIZE - 1);
        }
    }

    uint64_t bestGcd = 1;
    double bestScore = -1.0;
    for (size_t slot = 0; slot < IMS_GCD_TABLE_SIZE; slot++) {
        if (!gcdTable[slot].used) {
            continue;
        }
        double const score = ims_score_triple_gcd(
                gcdTable[slot].key, gcdTable[slot].count, totalTriples);
        if (score > bestScore) {
            bestScore = score;
            bestGcd   = gcdTable[slot].key;
        }
    }
    if (bestGcd <= 1 || bestScore < IMS_REQUIRED_BITS_SAVED_PER_NUM) {
        return 1;
    }

    // Check infrequent primary-bin filter (matching pco's est_bits_saved_per_num)
    for (size_t i = 0; i < sampleN; i++) {
        uint64_t primary;
        if (signedMode) {
            int64_t const sx    = (int64_t)sample[i];
            int64_t const sbase = (int64_t)bestGcd;
            int64_t q           = sx / sbase;
            int64_t const r     = sx - q * sbase;
            if (r < 0) {
                q -= 1;
            }
            primary = (uint64_t)q;
        } else {
            primary = sample[i] / bestGcd;
        }
        size_t slot = (size_t)ims_hash64(primary) & (IMS_PRIM_TABLE_SIZE - 1);
        for (size_t probe = 0; probe < IMS_PRIM_TABLE_SIZE; probe++) {
            if (!primTable[slot].used) {
                primTable[slot].used      = 1;
                primTable[slot].key       = primary;
                primTable[slot].count     = 1;
                primTable[slot].bitsSaved = bestScore;
                break;
            }
            if (primTable[slot].key == primary) {
                primTable[slot].count++;
                primTable[slot].bitsSaved += bestScore;
                break;
            }
            slot = (slot + 1) & (IMS_PRIM_TABLE_SIZE - 1);
        }
    }

    size_t infrequentCutoff = sampleN / IMS_CLASSIC_MEMORIZABLE_BINS;
    if (infrequentCutoff < 1) {
        infrequentCutoff = 1;
    }
    double totalSaved = 0.0;
    for (size_t slot = 0; slot < IMS_PRIM_TABLE_SIZE; slot++) {
        if (primTable[slot].used && primTable[slot].count <= infrequentCutoff) {
            totalSaved += primTable[slot].bitsSaved;
        }
    }
    if (totalSaved / (double)sampleN <= IMS_REQUIRED_BITS_SAVED_PER_NUM) {
        return 1;
    }
    return bestGcd;
}

#define IMS_DEFINE_SPLIT_UNSIGNED(W, T)                                \
    static void ims_split_u_##W(                                       \
            T* restrict multDst,                                       \
            T* restrict adjDst,                                        \
            const T* restrict src,                                     \
            size_t n,                                                  \
            uint64_t base64)                                           \
    {                                                                  \
        T const base = (T)base64;                                      \
        for (size_t i = 0; i < n; i++) {                               \
            T const v    = src[i];                                     \
            T const q    = (T)(v / base);                              \
            multDst[i]   = q;                                          \
            adjDst[i]    = (T)(v - (T)(q * base));                     \
        }                                                              \
    }

#define IMS_DEFINE_SPLIT_SIGNED(W, U, S)                               \
    static void ims_split_s_##W(                                       \
            U* restrict multDst,                                       \
            U* restrict adjDst,                                        \
            const S* restrict src,                                     \
            size_t n,                                                  \
            uint64_t base64)                                           \
    {                                                                  \
        S const sbase = (S)base64;                                     \
        for (size_t i = 0; i < n; i++) {                               \
            S const v = src[i];                                        \
            S q       = (S)(v / sbase);                                \
            S r       = (S)(v - (S)(q * sbase));                       \
            if (r < 0) {                                               \
                r = (S)(r + sbase);                                    \
                q = (S)(q - 1);                                        \
            }                                                          \
            multDst[i] = (U)q;                                         \
            adjDst[i]  = (U)r;                                         \
        }                                                              \
    }

IMS_DEFINE_SPLIT_UNSIGNED(8, uint8_t)
IMS_DEFINE_SPLIT_UNSIGNED(16, uint16_t)
IMS_DEFINE_SPLIT_UNSIGNED(32, uint32_t)
IMS_DEFINE_SPLIT_UNSIGNED(64, uint64_t)

IMS_DEFINE_SPLIT_SIGNED(8, uint8_t, int8_t)
IMS_DEFINE_SPLIT_SIGNED(16, uint16_t, int16_t)
IMS_DEFINE_SPLIT_SIGNED(32, uint32_t, int32_t)
IMS_DEFINE_SPLIT_SIGNED(64, uint64_t, int64_t)

ZL_Report EI_int_mult_split(
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
    ZL_ERR_IF_NOT(
            eltWidth == 1 || eltWidth == 2 || eltWidth == 4 || eltWidth == 8,
            node_invalid_input);

    int signedMode = ZL_IntMultSplit_unsigned;
    ZL_IntParam const modeParam =
            ZL_Encoder_getLocalIntParam(eictx, ZL_INT_MULT_SPLIT_MODE_PID);
    if (modeParam.paramId == ZL_INT_MULT_SPLIT_MODE_PID) {
        ZL_ERR_IF(
                modeParam.paramValue < 0 || modeParam.paramValue > 1,
                nodeParameter_invalidValue);
        signedMode = modeParam.paramValue;
    }

    uint64_t forcedBase = 0;
    ZL_IntParam const baseParam =
            ZL_Encoder_getLocalIntParam(eictx, ZL_INT_MULT_SPLIT_BASE_PID);
    if (baseParam.paramId == ZL_INT_MULT_SPLIT_BASE_PID) {
        ZL_ERR_IF(baseParam.paramValue < 0, nodeParameter_invalidValue);
        forcedBase = (uint64_t)baseParam.paramValue;
        if (forcedBase > 1) {
            ZL_ERR_IF_GT(
                    forcedBase,
                    ims_max_base_for_width(eltWidth, signedMode),
                    nodeParameter_invalidValue);
        }
    }

    unsigned const eltWidthLog2 = (unsigned)ZL_nextPow2(eltWidth);
    uint64_t base               = 1;
    if (n > 0) {
        if (forcedBase >= 1) {
            base = forcedBase;
        } else {
            base = ims_choose_base(
                    eictx, ZL_Input_ptr(in), n, eltWidth, signedMode);
        }
    }

    ZL_Output* const multOut =
            ZL_Encoder_createTypedStream(eictx, 0, n, eltWidth);
    ZL_Output* const adjOut = ZL_Encoder_createTypedStream(
            eictx, 1, base == 1 ? 0 : n, eltWidth);
    ZL_ERR_IF(multOut == NULL || adjOut == NULL, allocation);

    uint8_t hdr[1 + ZL_VARINT_LENGTH_64];
    hdr[0]          = (uint8_t)((eltWidthLog2 & 3u) | ((unsigned)(signedMode & 1) << 2));
    size_t const sz = 1u + ZL_varintEncode(base - 1u, hdr + 1);
    ZL_Encoder_sendCodecHeader(eictx, hdr, sz);

    if (n > 0) {
        if (base == 1u) {
            memcpy(ZL_Output_ptr(multOut), ZL_Input_ptr(in), n * eltWidth);
        } else {
            void* const multPtr       = ZL_Output_ptr(multOut);
            void* const adjPtr        = ZL_Output_ptr(adjOut);
            const void* const srcPtr  = ZL_Input_ptr(in);
            if (signedMode) {
                switch (eltWidth) {
                    case 1:
                        ims_split_s_8(
                                (uint8_t*)multPtr,
                                (uint8_t*)adjPtr,
                                (const int8_t*)srcPtr,
                                n,
                                base);
                        break;
                    case 2:
                        ims_split_s_16(
                                (uint16_t*)multPtr,
                                (uint16_t*)adjPtr,
                                (const int16_t*)srcPtr,
                                n,
                                base);
                        break;
                    case 4:
                        ims_split_s_32(
                                (uint32_t*)multPtr,
                                (uint32_t*)adjPtr,
                                (const int32_t*)srcPtr,
                                n,
                                base);
                        break;
                    case 8:
                        ims_split_s_64(
                                (uint64_t*)multPtr,
                                (uint64_t*)adjPtr,
                                (const int64_t*)srcPtr,
                                n,
                                base);
                        break;
                    default:
                        ZL_ERR(node_invalid_input);
                }
            } else {
                switch (eltWidth) {
                    case 1:
                        ims_split_u_8(
                                (uint8_t*)multPtr,
                                (uint8_t*)adjPtr,
                                (const uint8_t*)srcPtr,
                                n,
                                base);
                        break;
                    case 2:
                        ims_split_u_16(
                                (uint16_t*)multPtr,
                                (uint16_t*)adjPtr,
                                (const uint16_t*)srcPtr,
                                n,
                                base);
                        break;
                    case 4:
                        ims_split_u_32(
                                (uint32_t*)multPtr,
                                (uint32_t*)adjPtr,
                                (const uint32_t*)srcPtr,
                                n,
                                base);
                        break;
                    case 8:
                        ims_split_u_64(
                                (uint64_t*)multPtr,
                                (uint64_t*)adjPtr,
                                (const uint64_t*)srcPtr,
                                n,
                                base);
                        break;
                    default:
                        ZL_ERR(node_invalid_input);
                }
            }
        }
    }

    ZL_ERR_IF_ERR(ZL_Output_commit(multOut, n));
    ZL_ERR_IF_ERR(ZL_Output_commit(adjOut, base == 1u ? 0 : n));
    return ZL_returnSuccess();
}
