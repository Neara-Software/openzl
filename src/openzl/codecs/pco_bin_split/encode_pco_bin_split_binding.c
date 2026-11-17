// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/pco_bin_split/encode_pco_bin_split_binding.h"

#include <float.h>
#include <stdint.h>
#include <string.h>

#include "openzl/codecs/zl_pco_bin_split.h"
#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
#include "openzl/shared/bits.h"
#include "openzl/shared/mem.h"
#include "openzl/shared/portability.h"
#include "openzl/shared/varint.h"

enum {
    PBS_MAX_BINS_LOG = 8,
    PBS_MAX_BINS     = 256,
    PBS_SAMPLE_MAX   = 8192,
};

typedef struct {
    uint64_t lower;
    uint64_t upper;
    uint32_t count;
} PbsHistBin;

static inline unsigned pbs_offset_bits(uint64_t range)
{
    return range == 0 ? 0u : (unsigned)(64 - ZL_clz64(range));
}

static inline uint64_t pbs_mask64(unsigned bits)
{
    return bits >= 64u ? UINT64_MAX : ((1ULL << bits) - 1ULL);
}

static inline uint64_t pbs_sat_add_u64(uint64_t lo, uint64_t delta)
{
    return (delta >= UINT64_MAX - lo) ? UINT64_MAX : (lo + delta);
}

static inline float pbs_log2_approx(float x)
{
    uint32_t bits = 0;
    memcpy(&bits, &x, sizeof(bits));
    float const expPart  = (float)((int32_t)(bits >> 23) - 127);
    uint32_t const mBits = (bits & 0x007FFFFFu) | 0x3F800000u;
    float m              = 0.0f;
    memcpy(&m, &mBits, sizeof(m));
    float const t = m - 1.0f;
    return expPart + t * (1.4208645f + t * (-0.5772507f + t * 0.1563861f));
}

static inline unsigned pbs_varint_bits(uint64_t delta)
{
    unsigned const bl = pbs_offset_bits(delta);
    return bl <= 7u ? 8u : ((bl + 6u) / 7u) * 8u;
}

#define PBS_DEFINE_RUNS_FN(W, T)                                 \
    static size_t pbs_extract_runs_##W(                          \
            const T* sorted,                                     \
            size_t n,                                            \
            uint64_t* uVals,                                     \
            uint32_t* prefCnt)                                   \
    {                                                            \
        size_t u   = 0;                                          \
        prefCnt[0] = 0;                                          \
        size_t pos = 0;                                          \
        while (pos < n) {                                        \
            T const v = sorted[pos];                             \
            size_t r  = pos + 1u;                                \
            while (r < n && sorted[r] == v) {                    \
                r++;                                             \
            }                                                    \
            uVals[u]        = (uint64_t)v;                       \
            prefCnt[u + 1u] = (uint32_t)r;                       \
            u++;                                                 \
            pos = r;                                             \
        }                                                        \
        return u;                                                \
    }

PBS_DEFINE_RUNS_FN(2, uint16_t)
PBS_DEFINE_RUNS_FN(4, uint32_t)
PBS_DEFINE_RUNS_FN(8, uint64_t)

static inline float pbs_iv_cost(
        const uint64_t* uVals,
        const uint32_t* prefCnt,
        uint32_t s,
        uint32_t e,
        float binMetaCost,
        float totalCountLog2)
{
    float const c       = (float)(prefCnt[e + 1u] - prefCnt[s]);
    float const ansCost = totalCountLog2 - pbs_log2_approx(c);
    float const obCost  = (float)pbs_offset_bits(uVals[e] - uVals[s]);
    return binMetaCost + (ansCost + obCost) * c;
}

static inline uint32_t pbs_last_le(
        const uint64_t* uVals,
        uint32_t loIdx,
        uint32_t hiIdx,
        uint64_t target)
{
    while (loIdx < hiIdx) {
        uint32_t const mid = loIdx + (hiIdx - loIdx + 1u) / 2u;
        if (uVals[mid] <= target) {
            loIdx = mid;
        } else {
            hiIdx = mid - 1u;
        }
    }
    return loIdx;
}

static inline uint32_t pbs_first_ge(
        const uint64_t* uVals,
        uint32_t loIdx,
        uint32_t hiIdx,
        uint64_t target)
{
    while (loIdx < hiIdx) {
        uint32_t const mid = loIdx + (hiIdx - loIdx) / 2u;
        if (uVals[mid] >= target) {
            hiIdx = mid;
        } else {
            loIdx = mid + 1u;
        }
    }
    return loIdx;
}

static void pbs_eval_best_split(
        const uint64_t* uVals,
        const uint32_t* prefCnt,
        uint32_t s,
        uint32_t e,
        float binMetaCost,
        float totalCountLog2,
        float* outGain,
        uint32_t* outMid)
{
    *outGain = 0.0f;
    *outMid  = s;
    if (s >= e) {
        return;
    }
    uint64_t const lo = uVals[s];
    uint64_t const hi = uVals[e];
    unsigned const ob = pbs_offset_bits(hi - lo);
    if (ob == 0u) {
        return;
    }
    float const baseCost =
            pbs_iv_cost(uVals, prefCnt, s, e, binMetaCost, totalCountLog2);
    float bestGain   = 0.0f;
    uint32_t bestMid = s;

    uint32_t cands[7];
    size_t nCands = 0;
    for (unsigned sh = 1u; sh <= 3u && sh <= ob; sh++) {
        uint64_t const msk   = pbs_mask64(ob - sh);
        uint64_t const capLo = pbs_sat_add_u64(lo, msk);
        cands[nCands++]      = pbs_last_le(uVals, s, e - 1u, capLo);
        uint64_t const capHi = hi - msk;
        cands[nCands++]      = pbs_first_ge(uVals, s + 1u, e, capHi) - 1u;
    }
    cands[nCands++] = s + (e - s - 1u) / 2u;

    for (size_t ci = 0; ci < nCands; ci++) {
        uint32_t const m = cands[ci];
        int dup          = 0;
        for (size_t prev = 0; prev < ci; prev++) {
            if (cands[prev] == m) {
                dup = 1;
                break;
            }
        }
        if (dup) {
            continue;
        }
        float const splitCost =
                pbs_iv_cost(uVals, prefCnt, s, m, binMetaCost, totalCountLog2)
                + pbs_iv_cost(
                        uVals, prefCnt, m + 1u, e, binMetaCost, totalCountLog2);
        float const gain = baseCost - splitCost;
        if (gain > bestGain) {
            bestGain = gain;
            bestMid  = m;
        }
    }
    *outGain = bestGain;
    *outMid  = bestMid;
}

static size_t pbs_optimize_from_runs(
        const uint64_t* uVals,
        const uint32_t* prefCnt,
        size_t u,
        size_t totalElements,
        PbsHistBin* outBins)
{
    if (u <= 1u) {
        outBins[0].lower = uVals[0];
        outBins[0].upper = uVals[0];
        outBins[0].count = prefCnt[1];
        return 1u;
    }

    uint32_t const totalCount  = prefCnt[u];
    float const totalCountLog2 = pbs_log2_approx((float)totalCount);
    float const scale          = (float)totalCount / (float)totalElements;
    uint64_t const totalSpan   = uVals[u - 1u] - uVals[0];
    size_t const estBins       = u < 64u ? u : 64u;
    float const varintBits =
            (float)pbs_varint_bits(totalSpan / (uint64_t)estBins);
    float const binMetaCost = (varintBits + 18.0f) * scale;

    /* 1. Build adaptive remaining-budget initial intervals [hStart, hEnd]. */
    uint32_t hStart[PBS_MAX_BINS];
    uint32_t hEnd[PBS_MAX_BINS];
    size_t nHist = 0;
    if (u <= PBS_MAX_BINS) {
        for (size_t k = 0; k < u; k++) {
            hStart[k] = (uint32_t)k;
            hEnd[k]   = (uint32_t)k;
        }
        nHist = u;
    } else {
        uint32_t remElts = totalCount;
        size_t k         = 0;
        while (k < u) {
            size_t const remBins = (size_t)PBS_MAX_BINS - nHist;
            if (remBins <= 1u) {
                hStart[nHist] = (uint32_t)k;
                hEnd[nHist]   = (uint32_t)(u - 1u);
                nHist++;
                break;
            }
            uint32_t const targ =
                    (remElts + (uint32_t)remBins - 1u) / (uint32_t)remBins;
            uint32_t const c0 = prefCnt[k + 1u] - prefCnt[k];
            if (c0 >= targ || (u - k) <= remBins) {
                hStart[nHist] = (uint32_t)k;
                hEnd[nHist]   = (uint32_t)k;
                nHist++;
                remElts -= c0;
                k++;
            } else {
                uint32_t acc = c0;
                size_t m     = k + 1u;
                while (m < u && (u - m) >= (remBins - 1u)) {
                    uint32_t const cm = prefCnt[m + 1u] - prefCnt[m];
                    if (acc + cm > targ) {
                        break;
                    }
                    acc += cm;
                    m++;
                }
                hStart[nHist] = (uint32_t)k;
                hEnd[nHist]   = (uint32_t)(m - 1u);
                nHist++;
                remElts -= acc;
                k = m;
            }
        }
    }

    /* 2. Exact DP bin merging over the nHist candidate intervals. */
    float bestCosts[PBS_MAX_BINS + 1];
    size_t bestJs[PBS_MAX_BINS];
    bestCosts[0] = 0.0f;
    for (size_t i = 0; i < nHist; i++) {
        float bestCost       = FLT_MAX;
        size_t bestJ         = 0;
        uint32_t const eIdx  = hEnd[i];
        uint64_t const upper = uVals[eIdx];
        uint32_t const ccI   = prefCnt[eIdx + 1u];
        for (size_t j = i + 1u; j-- > 0u;) {
            uint32_t const sIdx    = hStart[j];
            uint32_t const c       = ccI - prefCnt[sIdx];
            unsigned const ob      = pbs_offset_bits(upper - uVals[sIdx]);
            float const fc         = (float)c;
            float const minBinLb   = binMetaCost + (float)ob * fc;
            if (minBinLb >= bestCost) {
                break;
            }
            float const ansCost = totalCountLog2 - pbs_log2_approx(fc);
            float const cost    = bestCosts[j] + minBinLb + ansCost * fc;
            if (cost < bestCost) {
                bestCost = cost;
                bestJ    = j;
            }
        }
        bestCosts[i + 1u] = bestCost;
        bestJs[i]         = bestJ;
    }

    uint32_t bStart[PBS_MAX_BINS];
    uint32_t bEnd[PBS_MAX_BINS];
    size_t numBins = 0;
    {
        size_t revStart[PBS_MAX_BINS];
        size_t revEnd[PBS_MAX_BINS];
        size_t cur = nHist - 1u;
        while (1) {
            size_t const j    = bestJs[cur];
            revStart[numBins] = j;
            revEnd[numBins]   = cur;
            numBins++;
            if (j == 0u) {
                break;
            }
            cur = j - 1u;
        }
        for (size_t b = 0; b < numBins; b++) {
            size_t const rk = numBins - 1u - b;
            bStart[b]       = hStart[revStart[rk]];
            bEnd[b]         = hEnd[revEnd[rk]];
        }
    }

    /* 3. Exact 1-value resolution power-of-2 capacity split when u > 256. */
    if (u > PBS_MAX_BINS) {
        float splitGain[PBS_MAX_BINS];
        uint32_t splitMid[PBS_MAX_BINS];
        for (size_t b = 0; b < numBins; b++) {
            pbs_eval_best_split(
                    uVals,
                    prefCnt,
                    bStart[b],
                    bEnd[b],
                    binMetaCost,
                    totalCountLog2,
                    &splitGain[b],
                    &splitMid[b]);
        }
        while (numBins < PBS_MAX_BINS) {
            float maxGain = 32.0f * scale;
            size_t bestB  = numBins;
            for (size_t b = 0; b < numBins; b++) {
                if (splitGain[b] > maxGain) {
                    maxGain = splitGain[b];
                    bestB   = b;
                }
            }
            if (bestB == numBins) {
                break;
            }
            uint32_t const mid = splitMid[bestB];
            uint32_t const oldE = bEnd[bestB];
            for (size_t sh = numBins; sh > bestB + 1u; sh--) {
                bStart[sh]    = bStart[sh - 1u];
                bEnd[sh]      = bEnd[sh - 1u];
                splitGain[sh] = splitGain[sh - 1u];
                splitMid[sh]  = splitMid[sh - 1u];
            }
            bEnd[bestB]          = mid;
            bStart[bestB + 1u]   = mid + 1u;
            bEnd[bestB + 1u]     = oldE;
            numBins++;
            pbs_eval_best_split(
                    uVals,
                    prefCnt,
                    bStart[bestB],
                    bEnd[bestB],
                    binMetaCost,
                    totalCountLog2,
                    &splitGain[bestB],
                    &splitMid[bestB]);
            pbs_eval_best_split(
                    uVals,
                    prefCnt,
                    bStart[bestB + 1u],
                    bEnd[bestB + 1u],
                    binMetaCost,
                    totalCountLog2,
                    &splitGain[bestB + 1u],
                    &splitMid[bestB + 1u]);
        }
    }
    float curTotalCost = 0.0f;
    int anyNonZeroOb   = 0;
    for (size_t b = 0; b < numBins; b++) {
        curTotalCost += pbs_iv_cost(
                uVals, prefCnt, bStart[b], bEnd[b], binMetaCost, totalCountLog2);
        if (uVals[bEnd[b]] != uVals[bStart[b]]) {
            anyNonZeroOb = 1;
        }
    }

    /* 4. Check allZeroOffsets exact unique-value bins when u <= 256. */
    if (u <= PBS_MAX_BINS && anyNonZeroOb) {
        float trivCost = 0.0f;
        for (size_t k = 0; k < u; k++) {
            trivCost += pbs_iv_cost(
                    uVals,
                    prefCnt,
                    (uint32_t)k,
                    (uint32_t)k,
                    binMetaCost,
                    totalCountLog2);
        }
        if (trivCost <= curTotalCost) {
            for (size_t k = 0; k < u; k++) {
                outBins[k].lower = uVals[k];
                outBins[k].upper = uVals[k];
                outBins[k].count = prefCnt[k + 1u] - prefCnt[k];
            }
            return u;
        }
    }

    /* 5. Check single-bin bypass (saves FSE bin-index stream). */
    float const singleBinCost =
            (float)pbs_offset_bits(totalSpan) * (float)totalCount
            + (float)pbs_varint_bits(uVals[0]) * scale;
    if (numBins > 1u
        && singleBinCost < curTotalCost + 0.10f * (float)totalCount) {
        outBins[0].lower = uVals[0];
        outBins[0].upper = uVals[u - 1u];
        outBins[0].count = totalCount;
        return 1u;
    }

    for (size_t b = 0; b < numBins; b++) {
        outBins[b].lower = uVals[bStart[b]];
        outBins[b].upper = uVals[bEnd[b]];
        outBins[b].count = prefCnt[bEnd[b] + 1u] - prefCnt[bStart[b]];
    }
    return numBins;
}

#define PBS_SHAR_STEP(BASE, V, LOWS, STEP)                                         \
    do {                                                                           \
        size_t const cand_ = (BASE) + (STEP);                                      \
        (BASE)             = ((LOWS)[cand_] <= (V)) ? cand_ : (BASE);              \
    } while (0)

#define PBS_SHAR_SEARCH_ONE(T, B_OUT, V, LOWS, REM, DEPTH)                         \
    do {                                                                           \
        T const v_    = (V);                                                       \
        size_t b_     = ((LOWS)[REM] <= v_) ? (REM) : 0u;                          \
        if ((DEPTH) >= 8) {                                                        \
            PBS_SHAR_STEP(b_, v_, LOWS, 128u);                                     \
        }                                                                          \
        if ((DEPTH) >= 7) {                                                        \
            PBS_SHAR_STEP(b_, v_, LOWS, 64u);                                      \
        }                                                                          \
        if ((DEPTH) >= 6) {                                                        \
            PBS_SHAR_STEP(b_, v_, LOWS, 32u);                                      \
        }                                                                          \
        if ((DEPTH) >= 5) {                                                        \
            PBS_SHAR_STEP(b_, v_, LOWS, 16u);                                      \
        }                                                                          \
        if ((DEPTH) >= 4) {                                                        \
            PBS_SHAR_STEP(b_, v_, LOWS, 8u);                                       \
        }                                                                          \
        if ((DEPTH) >= 3) {                                                        \
            PBS_SHAR_STEP(b_, v_, LOWS, 4u);                                       \
        }                                                                          \
        if ((DEPTH) >= 2) {                                                        \
            PBS_SHAR_STEP(b_, v_, LOWS, 2u);                                       \
        }                                                                          \
        if ((DEPTH) >= 1) {                                                        \
            PBS_SHAR_STEP(b_, v_, LOWS, 1u);                                       \
        }                                                                          \
        (B_OUT) = b_;                                                              \
    } while (0)

#define PBS_SHAR_LOOP_DEPTH(T, LOWS, LATENTS, N, REM, DEPTH, OBITS, BDST, TBITS) \
    do {                                                                             \
        size_t i_          = 0;                                                      \
        uint64_t tbits_    = 0;                                                      \
        for (; i_ + 4u <= (N); i_ += 4u) {                                           \
            size_t b0_, b1_, b2_, b3_;                                               \
            PBS_SHAR_SEARCH_ONE(T, b0_, (LATENTS)[i_ + 0u], LOWS, REM, DEPTH);       \
            PBS_SHAR_SEARCH_ONE(T, b1_, (LATENTS)[i_ + 1u], LOWS, REM, DEPTH);       \
            PBS_SHAR_SEARCH_ONE(T, b2_, (LATENTS)[i_ + 2u], LOWS, REM, DEPTH);       \
            PBS_SHAR_SEARCH_ONE(T, b3_, (LATENTS)[i_ + 3u], LOWS, REM, DEPTH);       \
            (BDST)[i_ + 0u] = (uint8_t)b0_;                                          \
            (BDST)[i_ + 1u] = (uint8_t)b1_;                                          \
            (BDST)[i_ + 2u] = (uint8_t)b2_;                                          \
            (BDST)[i_ + 3u] = (uint8_t)b3_;                                          \
            tbits_ += (uint64_t)(OBITS)[b0_] + (uint64_t)(OBITS)[b1_]                \
                    + (uint64_t)(OBITS)[b2_] + (uint64_t)(OBITS)[b3_];               \
        }                                                                            \
        for (; i_ < (N); i_++) {                                                     \
            size_t b0_;                                                              \
            PBS_SHAR_SEARCH_ONE(T, b0_, (LATENTS)[i_], LOWS, REM, DEPTH);            \
            (BDST)[i_] = (uint8_t)b0_;                                               \
            tbits_ += (uint64_t)(OBITS)[b0_];                                        \
        }                                                                            \
        *(TBITS) = tbits_;                                                           \
    } while (0)

#define PBS_DEFINE_SHAR_FIND(W, T)                                                 \
    static uint64_t pbs_assign_bins_##W##_scalar(                                  \
            const T* lowers,                                                       \
            size_t numBins,                                                        \
            const T* latents,                                                      \
            size_t n,                                                              \
            const uint8_t* offsetBits,                                             \
            uint8_t* binDst)                                                       \
    {                                                                              \
        ZL_ASSERT_GE(numBins, 2u);                                                 \
        unsigned const depth = 31u - (unsigned)ZL_clz32((uint32_t)numBins);        \
        size_t const rem     = numBins - ((size_t)1u << depth);                    \
        uint64_t totalBits   = 0;                                                  \
        switch (depth) {                                                           \
            case 1:                                                                \
                PBS_SHAR_LOOP_DEPTH(                                               \
                        T, lowers, latents, n, rem, 1, offsetBits, binDst, &totalBits); \
                break;                                                             \
            case 2:                                                                \
                PBS_SHAR_LOOP_DEPTH(                                               \
                        T, lowers, latents, n, rem, 2, offsetBits, binDst, &totalBits); \
                break;                                                             \
            case 3:                                                                \
                PBS_SHAR_LOOP_DEPTH(                                               \
                        T, lowers, latents, n, rem, 3, offsetBits, binDst, &totalBits); \
                break;                                                             \
            case 4:                                                                \
                PBS_SHAR_LOOP_DEPTH(                                               \
                        T, lowers, latents, n, rem, 4, offsetBits, binDst, &totalBits); \
                break;                                                             \
            case 5:                                                                \
                PBS_SHAR_LOOP_DEPTH(                                               \
                        T, lowers, latents, n, rem, 5, offsetBits, binDst, &totalBits); \
                break;                                                             \
            case 6:                                                                \
                PBS_SHAR_LOOP_DEPTH(                                               \
                        T, lowers, latents, n, rem, 6, offsetBits, binDst, &totalBits); \
                break;                                                             \
            case 7:                                                                \
                PBS_SHAR_LOOP_DEPTH(                                               \
                        T, lowers, latents, n, rem, 7, offsetBits, binDst, &totalBits); \
                break;                                                             \
            default:                                                               \
                PBS_SHAR_LOOP_DEPTH(                                               \
                        T, lowers, latents, n, 0u, 8, offsetBits, binDst, &totalBits); \
                break;                                                             \
        }                                                                          \
        return totalBits;                                                          \
    }

PBS_DEFINE_SHAR_FIND(2, uint16_t)
PBS_DEFINE_SHAR_FIND(4, uint32_t)
PBS_DEFINE_SHAR_FIND(8, uint64_t)

#if ZL_HAS_AVX2
static void pbs_prep_2_simd(
        const uint16_t* p,
        size_t n,
        int latentMode,
        uint16_t* dst,
        uint16_t* minOut,
        uint16_t* maxOut)
{
    size_t i        = 0;
    uint16_t minVal = UINT16_MAX;
    uint16_t maxVal = 0;
    if (n >= 16u) {
        __m256i vmin      = _mm256_set1_epi16((short)0xFFFF);
        __m256i vmax      = _mm256_setzero_si256();
        __m256i const s16 = _mm256_set1_epi16((short)0x8000);
        for (; i + 16u <= n; i += 16u) {
            __m256i v = _mm256_loadu_si256((const __m256i*)(const void*)(p + i));
            if (latentMode == ZL_PcoBinSplit_signed) {
                v = _mm256_xor_si256(v, s16);
                _mm256_storeu_si256((__m256i*)(void*)(dst + i), v);
            } else if (latentMode == ZL_PcoBinSplit_float) {
                __m256i m = _mm256_or_si256(_mm256_srai_epi16(v, 15), s16);
                v         = _mm256_xor_si256(v, m);
                _mm256_storeu_si256((__m256i*)(void*)(dst + i), v);
            }
            vmin = _mm256_min_epu16(vmin, v);
            vmax = _mm256_max_epu16(vmax, v);
        }
        uint16_t bufMin[16];
        uint16_t bufMax[16];
        _mm256_storeu_si256((__m256i*)(void*)bufMin, vmin);
        _mm256_storeu_si256((__m256i*)(void*)bufMax, vmax);
        for (size_t k = 0; k < 16u; k++) {
            if (bufMin[k] < minVal) minVal = bufMin[k];
            if (bufMax[k] > maxVal) maxVal = bufMax[k];
        }
    }
    for (; i < n; i++) {
        uint16_t v = p[i];
        if (latentMode == ZL_PcoBinSplit_signed) {
            v      = (uint16_t)(v ^ 0x8000u);
            dst[i] = v;
        } else if (latentMode == ZL_PcoBinSplit_float) {
            uint16_t const m = (uint16_t)((0u - (uint32_t)(v >> 15u)) | 0x8000u);
            v                = (uint16_t)(v ^ m);
            dst[i]           = v;
        }
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
    }
    *minOut = minVal;
    *maxOut = maxVal;
}

static void pbs_prep_4_simd(
        const uint32_t* p,
        size_t n,
        int latentMode,
        uint32_t* dst,
        uint32_t* minOut,
        uint32_t* maxOut)
{
    size_t i        = 0;
    uint32_t minVal = UINT32_MAX;
    uint32_t maxVal = 0;
    if (n >= 8u) {
        __m256i vmin      = _mm256_set1_epi32((int)0xFFFFFFFFu);
        __m256i vmax      = _mm256_setzero_si256();
        __m256i const s32 = _mm256_set1_epi32((int)0x80000000u);
        for (; i + 8u <= n; i += 8u) {
            __m256i v = _mm256_loadu_si256((const __m256i*)(const void*)(p + i));
            if (latentMode == ZL_PcoBinSplit_signed) {
                v = _mm256_xor_si256(v, s32);
                _mm256_storeu_si256((__m256i*)(void*)(dst + i), v);
            } else if (latentMode == ZL_PcoBinSplit_float) {
                __m256i m = _mm256_or_si256(_mm256_srai_epi32(v, 31), s32);
                v         = _mm256_xor_si256(v, m);
                _mm256_storeu_si256((__m256i*)(void*)(dst + i), v);
            }
            vmin = _mm256_min_epu32(vmin, v);
            vmax = _mm256_max_epu32(vmax, v);
        }
        uint32_t bufMin[8];
        uint32_t bufMax[8];
        _mm256_storeu_si256((__m256i*)(void*)bufMin, vmin);
        _mm256_storeu_si256((__m256i*)(void*)bufMax, vmax);
        for (size_t k = 0; k < 8u; k++) {
            if (bufMin[k] < minVal) minVal = bufMin[k];
            if (bufMax[k] > maxVal) maxVal = bufMax[k];
        }
    }
    for (; i < n; i++) {
        uint32_t v = p[i];
        if (latentMode == ZL_PcoBinSplit_signed) {
            v      = v ^ 0x80000000u;
            dst[i] = v;
        } else if (latentMode == ZL_PcoBinSplit_float) {
            uint32_t const m = (0u - (v >> 31u)) | 0x80000000u;
            v                = v ^ m;
            dst[i]           = v;
        }
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
    }
    *minOut = minVal;
    *maxOut = maxVal;
}

static void pbs_prep_8_simd(
        const uint64_t* p,
        size_t n,
        int latentMode,
        uint64_t* dst,
        uint64_t* minOut,
        uint64_t* maxOut)
{
    size_t i        = 0;
    uint64_t minVal = UINT64_MAX;
    uint64_t maxVal = 0;
    if (n >= 4u) {
        __m256i const s64 = _mm256_set1_epi64x((long long)0x8000000000000000ULL);
        __m256i vminS     = _mm256_set1_epi64x(INT64_MAX);
        __m256i vmaxS     = _mm256_set1_epi64x(INT64_MIN);
        for (; i + 4u <= n; i += 4u) {
            __m256i v = _mm256_loadu_si256((const __m256i*)(const void*)(p + i));
            __m256i vs;
            if (latentMode == ZL_PcoBinSplit_signed) {
                vs = v;
                v  = _mm256_xor_si256(v, s64);
                _mm256_storeu_si256((__m256i*)(void*)(dst + i), v);
            } else if (latentMode == ZL_PcoBinSplit_float) {
                __m256i hiSign = _mm256_shuffle_epi32(_mm256_srai_epi32(v, 31), 0xF5);
                __m256i m      = _mm256_or_si256(hiSign, s64);
                v              = _mm256_xor_si256(v, m);
                _mm256_storeu_si256((__m256i*)(void*)(dst + i), v);
                vs = _mm256_xor_si256(v, s64);
            } else {
                vs = _mm256_xor_si256(v, s64);
            }
            vminS = _mm256_blendv_epi8(vminS, vs, _mm256_cmpgt_epi64(vminS, vs));
            vmaxS = _mm256_blendv_epi8(vmaxS, vs, _mm256_cmpgt_epi64(vs, vmaxS));
        }
        uint64_t bufMin[4];
        uint64_t bufMax[4];
        _mm256_storeu_si256((__m256i*)(void*)bufMin, _mm256_xor_si256(vminS, s64));
        _mm256_storeu_si256((__m256i*)(void*)bufMax, _mm256_xor_si256(vmaxS, s64));
        for (size_t k = 0; k < 4u; k++) {
            if (bufMin[k] < minVal) minVal = bufMin[k];
            if (bufMax[k] > maxVal) maxVal = bufMax[k];
        }
    }
    for (; i < n; i++) {
        uint64_t v = p[i];
        if (latentMode == ZL_PcoBinSplit_signed) {
            v      = v ^ 0x8000000000000000ULL;
            dst[i] = v;
        } else if (latentMode == ZL_PcoBinSplit_float) {
            uint64_t const m = (0ULL - (v >> 63u)) | 0x8000000000000000ULL;
            v                = v ^ m;
            dst[i]           = v;
        }
        if (v < minVal) minVal = v;
        if (v > maxVal) maxVal = v;
    }
    *minOut = minVal;
    *maxOut = maxVal;
}

static uint64_t pbs_assign_bins_2(
        const uint16_t* lowers,
        size_t numBins,
        const uint16_t* latents,
        size_t n,
        const uint8_t* offsetBits,
        uint8_t* binDst)
{
    if (numBins <= 16u && n >= 16u) {
        __m256i thresh[16];
        __m256i const s16 = _mm256_set1_epi16((short)0x8000);
        for (size_t b = 1; b < numBins; b++) {
            uint16_t const t = (uint16_t)((lowers[b] - 1u) ^ 0x8000u);
            thresh[b]        = _mm256_set1_epi16((short)t);
        }
        uint8_t lutArr[32];
        memset(lutArr, 0, sizeof(lutArr));
        for (size_t b = 0; b < numBins; b++) {
            lutArr[b]      = offsetBits[b];
            lutArr[b + 16] = offsetBits[b];
        }
        __m256i const lut   = _mm256_loadu_si256((const __m256i*)(const void*)lutArr);
        __m256i const ones  = _mm256_set1_epi16(1);
        __m256i const zero  = _mm256_setzero_si256();
        __m256i bitSums     = _mm256_setzero_si256();
        size_t i            = 0;
        for (; i + 16u <= n; i += 16u) {
            __m256i const vx = _mm256_xor_si256(
                    _mm256_loadu_si256((const __m256i*)(const void*)(latents + i)),
                    s16);
            __m256i acc = _mm256_setzero_si256();
            for (size_t b = 1; b < numBins; b++) {
                acc = _mm256_sub_epi16(acc, _mm256_cmpgt_epi16(vx, thresh[b]));
            }
            __m256i const packed = _mm256_permute4x64_epi64(
                    _mm256_packus_epi16(acc, acc), 0xD8);
            __m128i const bins128 = _mm256_castsi256_si128(packed);
            _mm_storeu_si128((__m128i*)(void*)(binDst + i), bins128);
            __m128i const ob128 = _mm_shuffle_epi8(
                    _mm256_castsi256_si128(lut), bins128);
            bitSums = _mm256_add_epi64(
                    bitSums, _mm256_sad_epu8(_mm256_zextsi128_si256(ob128), zero));
            (void)ones;
        }
        uint64_t tmp[4];
        _mm256_storeu_si256((__m256i*)(void*)tmp, bitSums);
        uint64_t totalBits = tmp[0] + tmp[1] + tmp[2] + tmp[3];
        if (i < n) {
            totalBits += pbs_assign_bins_2_scalar(
                    lowers, numBins, latents + i, n - i, offsetBits, binDst + i);
        }
        return totalBits;
    }
    return pbs_assign_bins_2_scalar(
            lowers, numBins, latents, n, offsetBits, binDst);
}

static uint64_t pbs_assign_bins_4(
        const uint32_t* lowers,
        size_t numBins,
        const uint32_t* latents,
        size_t n,
        const uint8_t* offsetBits,
        uint8_t* binDst)
{
    if (numBins <= 32u && n >= 8u) {
        __m256i thresh[32];
        __m256i const s32 = _mm256_set1_epi32((int)0x80000000u);
        for (size_t b = 1; b < numBins; b++) {
            uint32_t const t = (lowers[b] - 1u) ^ 0x80000000u;
            thresh[b]        = _mm256_set1_epi32((int)t);
        }
        size_t i           = 0;
        uint64_t totalBits = 0;
        if (numBins <= 16u) {
            uint8_t lutArr[32];
            memset(lutArr, 0, sizeof(lutArr));
            for (size_t b = 0; b < numBins; b++) {
                lutArr[b]      = offsetBits[b];
                lutArr[b + 16] = offsetBits[b];
            }
            __m128i const lut128 = _mm_loadu_si128((const __m128i*)(const void*)lutArr);
            __m128i const zero   = _mm_setzero_si128();
            __m128i bitSum128    = _mm_setzero_si128();
            for (; i + 8u <= n; i += 8u) {
                __m256i const vx = _mm256_xor_si256(
                        _mm256_loadu_si256(
                                (const __m256i*)(const void*)(latents + i)),
                        s32);
                __m256i acc = _mm256_setzero_si256();
                for (size_t b = 1; b < numBins; b++) {
                    acc = _mm256_sub_epi32(
                            acc, _mm256_cmpgt_epi32(vx, thresh[b]));
                }
                __m256i const p16 = _mm256_permute4x64_epi64(
                        _mm256_packus_epi32(acc, acc), 0xD8);
                __m256i const p8     = _mm256_packus_epi16(p16, p16);
                __m128i const bins64 = _mm256_castsi256_si128(p8);
                _mm_storel_epi64((__m128i*)(void*)(binDst + i), bins64);
                __m128i const ob8 = _mm_shuffle_epi8(lut128, bins64);
                bitSum128 = _mm_add_epi64(
                        bitSum128,
                        _mm_sad_epu8(_mm_unpacklo_epi64(ob8, zero), zero));
            }
            totalBits = (uint64_t)_mm_cvtsi128_si64(bitSum128);
        } else {
            for (; i + 8u <= n; i += 8u) {
                __m256i const vx = _mm256_xor_si256(
                        _mm256_loadu_si256(
                                (const __m256i*)(const void*)(latents + i)),
                        s32);
                __m256i acc = _mm256_setzero_si256();
                for (size_t b = 1; b < numBins; b++) {
                    acc = _mm256_sub_epi32(
                            acc, _mm256_cmpgt_epi32(vx, thresh[b]));
                }
                __m256i const p16 = _mm256_permute4x64_epi64(
                        _mm256_packus_epi32(acc, acc), 0xD8);
                __m256i const p8 = _mm256_packus_epi16(p16, p16);
                uint64_t const v64 =
                        (uint64_t)_mm_cvtsi128_si64(_mm256_castsi256_si128(p8));
                memcpy(binDst + i, &v64, 8);
                totalBits += (uint64_t)offsetBits[binDst[i + 0]]
                        + (uint64_t)offsetBits[binDst[i + 1]]
                        + (uint64_t)offsetBits[binDst[i + 2]]
                        + (uint64_t)offsetBits[binDst[i + 3]]
                        + (uint64_t)offsetBits[binDst[i + 4]]
                        + (uint64_t)offsetBits[binDst[i + 5]]
                        + (uint64_t)offsetBits[binDst[i + 6]]
                        + (uint64_t)offsetBits[binDst[i + 7]];
            }
        }
        if (i < n) {
            totalBits += pbs_assign_bins_4_scalar(
                    lowers, numBins, latents + i, n - i, offsetBits, binDst + i);
        }
        return totalBits;
    }
    return pbs_assign_bins_4_scalar(
            lowers, numBins, latents, n, offsetBits, binDst);
}

static uint64_t pbs_assign_bins_8(
        const uint64_t* lowers,
        size_t numBins,
        const uint64_t* latents,
        size_t n,
        const uint8_t* offsetBits,
        uint8_t* binDst)
{
    if (numBins <= 32u && n >= 4u) {
        __m256i thresh[32];
        __m256i const s64 = _mm256_set1_epi64x((long long)0x8000000000000000ULL);
        for (size_t b = 1; b < numBins; b++) {
            uint64_t const t = (lowers[b] - 1ULL) ^ 0x8000000000000000ULL;
            thresh[b]        = _mm256_set1_epi64x((long long)t);
        }
        size_t i           = 0;
        uint64_t totalBits = 0;
        for (; i + 4u <= n; i += 4u) {
            __m256i const vx = _mm256_xor_si256(
                    _mm256_loadu_si256(
                            (const __m256i*)(const void*)(latents + i)),
                    s64);
            __m256i acc = _mm256_setzero_si256();
            for (size_t b = 1; b < numBins; b++) {
                acc = _mm256_sub_epi64(acc, _mm256_cmpgt_epi64(vx, thresh[b]));
            }
            uint64_t b0 = (uint64_t)_mm256_extract_epi64(acc, 0);
            uint64_t b1 = (uint64_t)_mm256_extract_epi64(acc, 1);
            uint64_t b2 = (uint64_t)_mm256_extract_epi64(acc, 2);
            uint64_t b3 = (uint64_t)_mm256_extract_epi64(acc, 3);
            binDst[i + 0] = (uint8_t)b0;
            binDst[i + 1] = (uint8_t)b1;
            binDst[i + 2] = (uint8_t)b2;
            binDst[i + 3] = (uint8_t)b3;
            totalBits += (uint64_t)offsetBits[b0] + (uint64_t)offsetBits[b1]
                    + (uint64_t)offsetBits[b2] + (uint64_t)offsetBits[b3];
        }
        if (i < n) {
            totalBits += pbs_assign_bins_8_scalar(
                    lowers, numBins, latents + i, n - i, offsetBits, binDst + i);
        }
        return totalBits;
    }
    return pbs_assign_bins_8_scalar(
            lowers, numBins, latents, n, offsetBits, binDst);
}
#else
#    define pbs_assign_bins_2 pbs_assign_bins_2_scalar
#    define pbs_assign_bins_4 pbs_assign_bins_4_scalar
#    define pbs_assign_bins_8 pbs_assign_bins_8_scalar
#endif

#define PBS_DEFINE_PREP_SCALAR(W, T, SIGNED_MASK, SHIFT)                           \
    static void pbs_prep_##W##_scalar(                                             \
            const T* p,                                                            \
            size_t n,                                                              \
            int latentMode,                                                        \
            T* dst,                                                                \
            T* minOut,                                                             \
            T* maxOut)                                                             \
    {                                                                              \
        T minVal = (T)~(T)0;                                                       \
        T maxVal = 0;                                                              \
        if (latentMode == ZL_PcoBinSplit_unsigned) {                               \
            for (size_t i = 0; i < n; i++) {                                       \
                T const v = p[i];                                                  \
                if (v < minVal) minVal = v;                                        \
                if (v > maxVal) maxVal = v;                                        \
            }                                                                      \
        } else if (latentMode == ZL_PcoBinSplit_signed) {                          \
            for (size_t i = 0; i < n; i++) {                                       \
                T const v = (T)(p[i] ^ (SIGNED_MASK));                             \
                dst[i]    = v;                                                     \
                if (v < minVal) minVal = v;                                        \
                if (v > maxVal) maxVal = v;                                        \
            }                                                                      \
        } else {                                                                   \
            for (size_t i = 0; i < n; i++) {                                       \
                T const x = p[i];                                                  \
                T const m = (T)(((T)0 - (T)(x >> (SHIFT))) | (SIGNED_MASK));       \
                T const v = (T)(x ^ m);                                            \
                dst[i]    = v;                                                     \
                if (v < minVal) minVal = v;                                        \
                if (v > maxVal) maxVal = v;                                        \
            }                                                                      \
        }                                                                          \
        *minOut = minVal;                                                          \
        *maxOut = maxVal;                                                          \
    }

#if !ZL_HAS_AVX2
PBS_DEFINE_PREP_SCALAR(2, uint16_t, 0x8000u, 15u)
PBS_DEFINE_PREP_SCALAR(4, uint32_t, 0x80000000u, 31u)
PBS_DEFINE_PREP_SCALAR(8, uint64_t, 0x8000000000000000ULL, 63u)
#    define pbs_prep_2 pbs_prep_2_scalar
#    define pbs_prep_4 pbs_prep_4_scalar
#    define pbs_prep_8 pbs_prep_8_scalar
#else
#    define pbs_prep_2 pbs_prep_2_simd
#    define pbs_prep_4 pbs_prep_4_simd
#    define pbs_prep_8 pbs_prep_8_simd
#endif

#define PBS_DEFINE_RADIX_SORT(W, T, BYTES)                                         \
    static const T* pbs_radix_sort_##W(                                            \
            const T* latents,                                                      \
            T* bufA,                                                               \
            T* bufB,                                                               \
            size_t n,                                                              \
            uint32_t counts[BYTES][256])                                           \
    {                                                                              \
        const T* curSrc = latents;                                                 \
        T* curDst       = bufA;                                                    \
        T* altDst       = bufB;                                                    \
        for (unsigned byteIdx = 0; byteIdx < (BYTES); byteIdx++) {                 \
            unsigned const shift    = byteIdx * 8u;                                \
            unsigned const firstB   = (unsigned)((latents[0] >> shift) & 0xFFu);   \
            if (counts[byteIdx][firstB] == n) {                                    \
                continue;                                                          \
            }                                                                      \
            uint32_t offsets[256];                                                 \
            uint32_t acc = 0;                                                      \
            for (size_t b = 0; b < 256; b++) {                                     \
                offsets[b] = acc;                                                  \
                acc += counts[byteIdx][b];                                         \
            }                                                                      \
            for (size_t i = 0; i < n; i++) {                                       \
                T const v                 = curSrc[i];                             \
                unsigned const bucket     = (unsigned)((v >> shift) & 0xFFu);      \
                curDst[offsets[bucket]++] = v;                                     \
            }                                                                      \
            curSrc    = curDst;                                                    \
            T* tmpDst = curDst;                                                    \
            curDst    = altDst;                                                    \
            altDst    = tmpDst;                                                    \
        }                                                                          \
        return curSrc;                                                             \
    }

PBS_DEFINE_RADIX_SORT(2, uint16_t, 2)
PBS_DEFINE_RADIX_SORT(4, uint32_t, 4)
PBS_DEFINE_RADIX_SORT(8, uint64_t, 8)

#define PBS_DEFINE_BITPACK(W, T)                                                   \
    static void pbs_pack_offsets_##W(                                              \
            uint8_t* dst,                                                          \
            size_t packedBytes,                                                    \
            const T* latents,                                                      \
            size_t n,                                                              \
            const T* lowers,                                                       \
            const uint8_t* offsetBits,                                             \
            const uint8_t* binIds,                                                 \
            int singleBin,                                                         \
            unsigned maxOb)                                                        \
    {                                                                              \
        if (singleBin) {                                                           \
            T const lo        = lowers[0];                                         \
            unsigned const ob = offsetBits[0];                                     \
            if (ob == (W) * 8u && ZL_isLittleEndian()) {                           \
                if (lo == 0) {                                                     \
                    memcpy(dst, latents, n * sizeof(T));                           \
                } else {                                                           \
                    T* const dstT = (T*)(void*)dst;                                \
                    for (size_t j = 0; j < n; j++) {                               \
                        dstT[j] = (T)(latents[j] - lo);                            \
                    }                                                              \
                }                                                                  \
                return;                                                            \
            }                                                                      \
        }                                                                          \
        uint8_t* const dstEnd = dst + packedBytes;                                 \
        uint64_t bitbuf       = 0;                                                 \
        unsigned bitcnt       = 0;                                                 \
        size_t i              = 0;                                                 \
        if (singleBin) {                                                           \
            T const lo        = lowers[0];                                         \
            unsigned const ob = offsetBits[0];                                     \
            if (ob <= 56u && packedBytes >= 8u) {                                  \
                uint8_t* const dstSafe = dstEnd - 8u;                              \
                while (i < n && dst <= dstSafe) {                                  \
                    uint64_t const off = (uint64_t)(latents[i] - lo);              \
                    bitbuf |= off << bitcnt;                                       \
                    bitcnt += ob;                                                  \
                    ZL_writeLE64(dst, bitbuf);                                     \
                    unsigned const adv = bitcnt >> 3;                              \
                    dst += adv;                                                    \
                    bitbuf >>= (adv << 3);                                         \
                    bitcnt &= 7u;                                                  \
                    i++;                                                           \
                }                                                                  \
            }                                                                      \
            for (; i < n; i++) {                                                   \
                uint64_t const off = (uint64_t)(latents[i] - lo);                  \
                bitbuf |= off << bitcnt;                                           \
                if (bitcnt + ob >= 64u) {                                          \
                    if (dst + 8 <= dstEnd) {                                       \
                        ZL_writeLE64(dst, bitbuf);                                 \
                        dst += 8;                                                  \
                    } else {                                                       \
                        for (int k = 0; k < 8 && dst < dstEnd; k++) {              \
                            *dst++ = (uint8_t)(bitbuf >> (k * 8));                 \
                        }                                                          \
                    }                                                              \
                    bitbuf = (bitcnt == 0u) ? 0ULL : (off >> (64u - bitcnt));      \
                    bitcnt = (bitcnt + ob) - 64u;                                  \
                } else {                                                           \
                    bitcnt += ob;                                                  \
                }                                                                  \
            }                                                                      \
        } else {                                                                   \
            if (maxOb <= 14u && packedBytes >= 8u) {                               \
                uint8_t* const dstSafe = dstEnd - 8u;                              \
                while (i + 4u <= n && dst <= dstSafe) {                            \
                    unsigned const b0  = binIds[i];                                \
                    unsigned const b1  = binIds[i + 1u];                           \
                    unsigned const b2  = binIds[i + 2u];                           \
                    unsigned const b3  = binIds[i + 3u];                           \
                    unsigned const ob0 = offsetBits[b0];                           \
                    unsigned const ob1 = offsetBits[b1];                           \
                    unsigned const ob2 = offsetBits[b2];                           \
                    unsigned const ob3 = offsetBits[b3];                           \
                    uint64_t const o0  = (uint64_t)(latents[i] - lowers[b0]);      \
                    uint64_t const o1  = (uint64_t)(latents[i + 1u] - lowers[b1]); \
                    uint64_t const o2  = (uint64_t)(latents[i + 2u] - lowers[b2]); \
                    uint64_t const o3  = (uint64_t)(latents[i + 3u] - lowers[b3]); \
                    unsigned const s1  = bitcnt + ob0;                             \
                    unsigned const s2  = s1 + ob1;                                 \
                    unsigned const s3  = s2 + ob2;                                 \
                    bitbuf |= (o0 << bitcnt) | (o1 << s1) | (o2 << s2) | (o3 << s3); \
                    bitcnt = s3 + ob3;                                             \
                    ZL_writeLE64(dst, bitbuf);                                     \
                    unsigned const adv = bitcnt >> 3;                              \
                    dst += adv;                                                    \
                    bitbuf >>= (adv << 3);                                         \
                    bitcnt &= 7u;                                                  \
                    i += 4u;                                                       \
                }                                                                  \
            }                                                                      \
            if (maxOb <= 56u && packedBytes >= 8u) {                               \
                uint8_t* const dstSafe = dstEnd - 8u;                              \
                while (i < n && dst <= dstSafe) {                                  \
                    unsigned const b   = binIds[i];                                \
                    unsigned const ob  = offsetBits[b];                            \
                    uint64_t const off = (uint64_t)(latents[i] - lowers[b]);       \
                    bitbuf |= off << bitcnt;                                       \
                    bitcnt += ob;                                                  \
                    ZL_writeLE64(dst, bitbuf);                                     \
                    unsigned const adv = bitcnt >> 3;                              \
                    dst += adv;                                                    \
                    bitbuf >>= (adv << 3);                                         \
                    bitcnt &= 7u;                                                  \
                    i++;                                                           \
                }                                                                  \
            }                                                                      \
            for (; i < n; i++) {                                                   \
                unsigned const b   = binIds[i];                                    \
                unsigned const ob  = offsetBits[b];                                \
                uint64_t const off = (uint64_t)(latents[i] - lowers[b]);           \
                bitbuf |= off << bitcnt;                                           \
                if (bitcnt + ob >= 64u) {                                          \
                    if (dst + 8 <= dstEnd) {                                       \
                        ZL_writeLE64(dst, bitbuf);                                 \
                        dst += 8;                                                  \
                    } else {                                                       \
                        for (int k = 0; k < 8 && dst < dstEnd; k++) {              \
                            *dst++ = (uint8_t)(bitbuf >> (k * 8));                 \
                        }                                                          \
                    }                                                              \
                    bitbuf = (bitcnt == 0u) ? 0ULL : (off >> (64u - bitcnt));      \
                    bitcnt = (bitcnt + ob) - 64u;                                  \
                } else {                                                           \
                    bitcnt += ob;                                                  \
                }                                                                  \
            }                                                                      \
        }                                                                          \
        while (bitcnt > 0u && dst < dstEnd) {                                      \
            *dst++ = (uint8_t)(bitbuf & 0xFFu);                                    \
            bitbuf >>= 8;                                                          \
            bitcnt = (bitcnt >= 8u) ? (bitcnt - 8u) : 0u;                          \
        }                                                                          \
    }

PBS_DEFINE_BITPACK(1, uint8_t)
PBS_DEFINE_BITPACK(2, uint16_t)
PBS_DEFINE_BITPACK(4, uint32_t)
PBS_DEFINE_BITPACK(8, uint64_t)

ZL_Report EI_pco_bin_split(
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

    int latentMode = ZL_PcoBinSplit_unsigned;
    ZL_IntParam const lmParam =
            ZL_Encoder_getLocalIntParam(eictx, ZL_PCO_BIN_SPLIT_LATENT_MODE_PID);
    if (lmParam.paramId == ZL_PCO_BIN_SPLIT_LATENT_MODE_PID) {
        ZL_ERR_IF(
                lmParam.paramValue < 0 || lmParam.paramValue > 2,
                nodeParameter_invalidValue);
        latentMode = lmParam.paramValue;
    }
    if (eltWidth == 1 && latentMode == ZL_PcoBinSplit_float) {
        latentMode = ZL_PcoBinSplit_unsigned;
    }

    unsigned const eltWidthLog2 = (unsigned)ZL_nextPow2(eltWidth);
    unsigned const eltBits      = (unsigned)(eltWidth * 8u);

    if (n == 0) {
        ZL_Output* const binOut = ZL_Encoder_createTypedStream(eictx, 0, 0, 1);
        ZL_Output* const offOut = ZL_Encoder_createTypedStream(eictx, 1, 0, 1);
        ZL_ERR_IF(binOut == NULL || offOut == NULL, allocation);

        uint8_t hdr[2];
        hdr[0] = (uint8_t)((eltWidthLog2 & 3u) | ((unsigned)(latentMode & 3) << 2)
                           | (1u << 4) | (1u << 5));
        size_t const sz = 1u + ZL_varintEncode(0, hdr + 1);
        ZL_Encoder_sendCodecHeader(eictx, hdr, sz);
        ZL_ERR_IF_ERR(ZL_Output_commit(binOut, 0));
        ZL_ERR_IF_ERR(ZL_Output_commit(offOut, 0));
        return ZL_returnSuccess();
    }

    const void* const srcPtr = ZL_Input_ptr(in);
    const uint8_t* latents1  = NULL;
    const uint16_t* latents2 = NULL;
    const uint32_t* latents4 = NULL;
    const uint64_t* latents8 = NULL;

    PbsHistBin bins[PBS_MAX_BINS];
    size_t numBins         = 0;
    uint64_t uVals1[256];
    uint32_t pref1[257];
    const uint64_t* uVals  = NULL;
    const uint32_t* pref   = NULL;
    size_t uRuns           = 0;
    int sampled            = 0;
    uint64_t minVal        = 0;
    uint64_t maxVal        = 0;
    uint32_t counts1[256];

    switch (eltWidth) {
        case 1: {
            const uint8_t* const p = (const uint8_t*)srcPtr;
            uint32_t c0[256], c1[256], c2[256], c3[256];
            memset(c0, 0, sizeof(c0));
            memset(c1, 0, sizeof(c1));
            memset(c2, 0, sizeof(c2));
            memset(c3, 0, sizeof(c3));
            if (latentMode == ZL_PcoBinSplit_signed) {
                uint8_t* const l = (uint8_t*)ZL_Encoder_getScratchSpace(eictx, n);
                ZL_ERR_IF_NULL(l, allocation);
                size_t i = 0;
#if ZL_HAS_AVX2
                __m256i const signM = _mm256_set1_epi8((char)0x80);
                for (; i + 32u <= n; i += 32u) {
                    __m256i const xv = _mm256_xor_si256(
                            _mm256_loadu_si256((const __m256i*)(p + i)), signM);
                    _mm256_storeu_si256((__m256i*)(l + i), xv);
                }
#endif
                for (; i < n; i++) {
                    l[i] = (uint8_t)(p[i] ^ 0x80u);
                }
                latents1 = l;
            } else {
                latents1 = p;
            }
            size_t i = 0;
            for (; i + 4u <= n; i += 4u) {
                c0[latents1[i]]++;
                c1[latents1[i + 1u]]++;
                c2[latents1[i + 2u]]++;
                c3[latents1[i + 3u]]++;
            }
            for (; i < n; i++) {
                c0[latents1[i]]++;
            }
            for (unsigned v = 0; v < 256u; v++) {
                counts1[v] = c0[v] + c1[v] + c2[v] + c3[v];
            }
            if (counts1[latents1[0]] == n) {
                bins[0].lower = latents1[0];
                bins[0].upper = latents1[0];
                bins[0].count = (uint32_t)n;
                numBins       = 1u;
            } else {
                pref1[0] = 0;
                for (unsigned v = 0; v < 256u; v++) {
                    uint32_t const cv = counts1[v];
                    if (cv != 0u) {
                        uVals1[uRuns]        = (uint64_t)v;
                        pref1[uRuns + 1u]    = pref1[uRuns] + cv;
                        uRuns++;
                    }
                }
                uVals = uVals1;
                pref  = pref1;
            }
            break;
        }
        case 2: {
            const uint16_t* const p = (const uint16_t*)srcPtr;
            uint16_t* scratch2      = NULL;
            if (latentMode == ZL_PcoBinSplit_unsigned) {
                latents2 = p;
            } else {
                scratch2 = (uint16_t*)ZL_Encoder_getScratchSpace(
                        eictx, n * sizeof(uint16_t));
                ZL_ERR_IF_NULL(scratch2, allocation);
                latents2 = scratch2;
            }
            uint16_t mn2 = 0, mx2 = 0;
            pbs_prep_2(p, n, latentMode, scratch2, &mn2, &mx2);
            minVal = mn2;
            maxVal = mx2;
            if (mn2 == mx2) {
                bins[0].lower = mn2;
                bins[0].upper = mn2;
                bins[0].count = (uint32_t)n;
                numBins       = 1u;
            } else if (n >= 512u) {
                uint32_t* const counts2 = (uint32_t*)ZL_Encoder_getScratchSpace(
                        eictx, 65536u * sizeof(uint32_t));
                ZL_ERR_IF_NULL(counts2, allocation);
                size_t const rangeLen = (size_t)((uint32_t)mx2 - (uint32_t)mn2 + 1u);
                memset(counts2 + mn2, 0, rangeLen * sizeof(uint32_t));
                for (size_t i = 0; i < n; i++) {
                    counts2[latents2[i]]++;
                }
                size_t const allocU = rangeLen < n ? rangeLen : n;
                uint64_t* const uv  = (uint64_t*)ZL_Encoder_getScratchSpace(
                        eictx, allocU * sizeof(uint64_t));
                uint32_t* const pc = (uint32_t*)ZL_Encoder_getScratchSpace(
                        eictx, (allocU + 1u) * sizeof(uint32_t));
                ZL_ERR_IF(uv == NULL || pc == NULL, allocation);
                pc[0] = 0;
                for (uint32_t v = (uint32_t)mn2; v <= (uint32_t)mx2; v++) {
                    uint32_t const cv = counts2[v];
                    if (cv != 0u) {
                        uv[uRuns]        = (uint64_t)v;
                        pc[uRuns + 1u]   = pc[uRuns] + cv;
                        uRuns++;
                    }
                }
                uVals = uv;
                pref  = pc;
            } else {
                uint32_t counts[2][256];
                memset(counts, 0, sizeof(counts));
                for (size_t i = 0; i < n; i++) {
                    uint16_t const v = latents2[i];
                    counts[0][v & 0xFFu]++;
                    counts[1][(v >> 8) & 0xFFu]++;
                }
                uint16_t* const bufA = (uint16_t*)ZL_Encoder_getScratchSpace(
                        eictx, n * sizeof(uint16_t));
                uint16_t* const bufB = (uint16_t*)ZL_Encoder_getScratchSpace(
                        eictx, n * sizeof(uint16_t));
                uint64_t* const uv = (uint64_t*)ZL_Encoder_getScratchSpace(
                        eictx, n * sizeof(uint64_t));
                uint32_t* const pc = (uint32_t*)ZL_Encoder_getScratchSpace(
                        eictx, (n + 1u) * sizeof(uint32_t));
                ZL_ERR_IF(
                        bufA == NULL || bufB == NULL || uv == NULL || pc == NULL,
                        allocation);
                const uint16_t* const sorted =
                        pbs_radix_sort_2(latents2, bufA, bufB, n, counts);
                uRuns = pbs_extract_runs_2(sorted, n, uv, pc);
                uVals = uv;
                pref  = pc;
            }
            break;
        }
        case 4: {
            const uint32_t* const p = (const uint32_t*)srcPtr;
            uint32_t* scratch4      = NULL;
            if (latentMode == ZL_PcoBinSplit_unsigned) {
                latents4 = p;
            } else {
                scratch4 = (uint32_t*)ZL_Encoder_getScratchSpace(
                        eictx, n * sizeof(uint32_t));
                ZL_ERR_IF_NULL(scratch4, allocation);
                latents4 = scratch4;
            }
            uint32_t mn4 = 0, mx4 = 0;
            pbs_prep_4(p, n, latentMode, scratch4, &mn4, &mx4);
            minVal = mn4;
            maxVal = mx4;
            if (mn4 == mx4) {
                bins[0].lower = mn4;
                bins[0].upper = mn4;
                bins[0].count = (uint32_t)n;
                numBins       = 1u;
            } else {
                size_t sn            = n;
                const uint32_t* sSrc = latents4;
                if (n > PBS_SAMPLE_MAX) {
                    sampled             = 1;
                    sn                  = PBS_SAMPLE_MAX;
                    uint32_t* const smp = (uint32_t*)ZL_Encoder_getScratchSpace(
                            eictx, sn * sizeof(uint32_t));
                    ZL_ERR_IF_NULL(smp, allocation);
                    for (size_t i = 0; i < sn; i++) {
                        smp[i] = latents4[(i * n) / sn];
                    }
                    sSrc = smp;
                }
                uint32_t counts[4][256];
                memset(counts, 0, sizeof(counts));
                for (size_t i = 0; i < sn; i++) {
                    uint32_t const v = sSrc[i];
                    counts[0][v & 0xFFu]++;
                    counts[1][(v >> 8) & 0xFFu]++;
                    counts[2][(v >> 16) & 0xFFu]++;
                    counts[3][(v >> 24) & 0xFFu]++;
                }
                uint32_t const v0 = sSrc[0];
                if (counts[0][v0 & 0xFFu] == sn && counts[1][(v0 >> 8) & 0xFFu] == sn
                    && counts[2][(v0 >> 16) & 0xFFu] == sn
                    && counts[3][(v0 >> 24) & 0xFFu] == sn) {
                    bins[0].lower = minVal;
                    bins[0].upper = maxVal;
                    bins[0].count = (uint32_t)sn;
                    numBins       = 1u;
                } else {
                    uint32_t* const bufA = (uint32_t*)ZL_Encoder_getScratchSpace(
                            eictx, sn * sizeof(uint32_t));
                    uint32_t* const bufB = (uint32_t*)ZL_Encoder_getScratchSpace(
                            eictx, sn * sizeof(uint32_t));
                    uint64_t* const uv = (uint64_t*)ZL_Encoder_getScratchSpace(
                            eictx, sn * sizeof(uint64_t));
                    uint32_t* const pc = (uint32_t*)ZL_Encoder_getScratchSpace(
                            eictx, (sn + 1u) * sizeof(uint32_t));
                    ZL_ERR_IF(
                            bufA == NULL || bufB == NULL || uv == NULL
                                    || pc == NULL,
                            allocation);
                    const uint32_t* const sorted =
                            pbs_radix_sort_4(sSrc, bufA, bufB, sn, counts);
                    uRuns = pbs_extract_runs_4(sorted, sn, uv, pc);
                    uVals = uv;
                    pref  = pc;
                }
            }
            break;
        }
        case 8: {
            const uint64_t* const p = (const uint64_t*)srcPtr;
            uint64_t* scratch8      = NULL;
            if (latentMode == ZL_PcoBinSplit_unsigned) {
                latents8 = p;
            } else {
                scratch8 = (uint64_t*)ZL_Encoder_getScratchSpace(
                        eictx, n * sizeof(uint64_t));
                ZL_ERR_IF_NULL(scratch8, allocation);
                latents8 = scratch8;
            }
            uint64_t mn8 = 0, mx8 = 0;
            pbs_prep_8(p, n, latentMode, scratch8, &mn8, &mx8);
            minVal = mn8;
            maxVal = mx8;
            if (mn8 == mx8) {
                bins[0].lower = mn8;
                bins[0].upper = mn8;
                bins[0].count = (uint32_t)n;
                numBins       = 1u;
            } else {
                size_t sn            = n;
                const uint64_t* sSrc = latents8;
                if (n > PBS_SAMPLE_MAX) {
                    sampled             = 1;
                    sn                  = PBS_SAMPLE_MAX;
                    uint64_t* const smp = (uint64_t*)ZL_Encoder_getScratchSpace(
                            eictx, sn * sizeof(uint64_t));
                    ZL_ERR_IF_NULL(smp, allocation);
                    for (size_t i = 0; i < sn; i++) {
                        smp[i] = latents8[(i * n) / sn];
                    }
                    sSrc = smp;
                }
                uint32_t counts[8][256];
                memset(counts, 0, sizeof(counts));
                for (size_t i = 0; i < sn; i++) {
                    uint64_t const v = sSrc[i];
                    counts[0][v & 0xFFu]++;
                    counts[1][(v >> 8) & 0xFFu]++;
                    counts[2][(v >> 16) & 0xFFu]++;
                    counts[3][(v >> 24) & 0xFFu]++;
                    counts[4][(v >> 32) & 0xFFu]++;
                    counts[5][(v >> 40) & 0xFFu]++;
                    counts[6][(v >> 48) & 0xFFu]++;
                    counts[7][(v >> 56) & 0xFFu]++;
                }
                uint64_t const v0 = sSrc[0];
                int allConst      = 1;
                for (unsigned b = 0; b < 8u; b++) {
                    if (counts[b][(v0 >> (b * 8u)) & 0xFFu] != sn) {
                        allConst = 0;
                        break;
                    }
                }
                if (allConst) {
                    bins[0].lower = minVal;
                    bins[0].upper = maxVal;
                    bins[0].count = (uint32_t)sn;
                    numBins       = 1u;
                } else {
                    uint64_t* const bufA = (uint64_t*)ZL_Encoder_getScratchSpace(
                            eictx, sn * sizeof(uint64_t));
                    uint64_t* const bufB = (uint64_t*)ZL_Encoder_getScratchSpace(
                            eictx, sn * sizeof(uint64_t));
                    uint64_t* const uv = (uint64_t*)ZL_Encoder_getScratchSpace(
                            eictx, sn * sizeof(uint64_t));
                    uint32_t* const pc = (uint32_t*)ZL_Encoder_getScratchSpace(
                            eictx, (sn + 1u) * sizeof(uint32_t));
                    ZL_ERR_IF(
                            bufA == NULL || bufB == NULL || uv == NULL
                                    || pc == NULL,
                            allocation);
                    const uint64_t* const sorted =
                            pbs_radix_sort_8(sSrc, bufA, bufB, sn, counts);
                    uRuns = pbs_extract_runs_8(sorted, sn, uv, pc);
                    uVals = uv;
                    pref  = pc;
                }
            }
            break;
        }
    }

    if (numBins == 0u) {
        numBins = pbs_optimize_from_runs(uVals, pref, uRuns, n, bins);
    }

    if (sampled) {
        if (bins[0].lower > minVal) {
            bins[0].lower = minVal;
        }
        if (bins[numBins - 1u].upper < maxVal) {
            bins[numBins - 1u].upper = maxVal;
        }
        for (size_t b = 0; b + 1u < numBins; b++) {
            uint64_t const lo           = bins[b].lower;
            unsigned const ob           = pbs_offset_bits(bins[b].upper - lo);
            uint64_t const nextLo       = bins[b + 1u].lower;
            uint64_t const nextLoMinus1 = nextLo - 1ULL;
            if (pbs_offset_bits(nextLoMinus1 - lo) == ob) {
                bins[b].upper = nextLoMinus1;
            } else {
                uint64_t const ms = (ob >= eltBits)
                        ? UINT64_MAX
                        : pbs_sat_add_u64(lo, pbs_mask64(ob));
                if (ms < nextLoMinus1) {
                    bins[b].upper         = ms;
                    uint64_t const candLo = ms + 1ULL;
                    uint64_t const nextHi = bins[b + 1u].upper;
                    if (pbs_offset_bits(nextHi - candLo)
                        == pbs_offset_bits(nextHi - nextLo)) {
                        bins[b + 1u].lower = candLo;
                    }
                }
            }
        }
    }
    uint64_t lowers[PBS_MAX_BINS];
    uint8_t lowers1[PBS_MAX_BINS];
    uint16_t lowers2[PBS_MAX_BINS];
    uint32_t lowers4[PBS_MAX_BINS];
    uint8_t offsetBits[PBS_MAX_BINS];
    uint64_t maxSafe[PBS_MAX_BINS];
    int allZeroOffsets = 1;
    int anyTightGap    = 0;
    unsigned maxOb     = 0;
    for (size_t b = 0; b < numBins; b++) {
        uint64_t const lo = bins[b].lower;
        lowers[b]         = lo;
        lowers1[b]        = (uint8_t)lo;
        lowers2[b]        = (uint16_t)lo;
        lowers4[b]        = (uint32_t)lo;
        unsigned const ob = pbs_offset_bits(bins[b].upper - lo);
        offsetBits[b]     = (uint8_t)ob;
        uint64_t const ms = (ob >= eltBits) ? UINT64_MAX : pbs_sat_add_u64(lo, pbs_mask64(ob));
        maxSafe[b]        = ms;
        if (sampled && b + 1u < numBins && ms < bins[b + 1u].lower - 1ULL) {
            anyTightGap = 1;
        }
        if (ob != 0u) {
            allZeroOffsets = 0;
            if (ob > maxOb) {
                maxOb = ob;
            }
        }
    }

    int const singleBin = (numBins == 1u);
    ZL_Output* const binOut =
            ZL_Encoder_createTypedStream(eictx, 0, singleBin ? 0 : n, 1);
    ZL_ERR_IF_NULL(binOut, allocation);

    uint64_t totalBits    = 0;
    uint8_t* const binDst = (uint8_t*)ZL_Output_ptr(binOut);
    if (singleBin) {
        totalBits = (uint64_t)n * (uint64_t)offsetBits[0];
    } else {
        switch (eltWidth) {
            case 1: {
                uint8_t valToBin[256];
                for (size_t b = 0; b < numBins; b++) {
                    size_t const s = (b == 0) ? 0u : (size_t)lowers1[b];
                    size_t const e =
                            (b + 1u < numBins) ? (size_t)lowers1[b + 1u] : 256u;
                    if (e > s) {
                        memset(valToBin + s, (int)b, e - s);
                    }
                }
                for (unsigned v = 0; v < 256u; v++) {
                    totalBits += (uint64_t)counts1[v]
                            * (uint64_t)offsetBits[valToBin[v]];
                }
                size_t i = 0;
                for (; i + 8u <= n; i += 8u) {
                    uint64_t const pack =
                            (uint64_t)valToBin[latents1[i]]
                            | ((uint64_t)valToBin[latents1[i + 1u]] << 8)
                            | ((uint64_t)valToBin[latents1[i + 2u]] << 16)
                            | ((uint64_t)valToBin[latents1[i + 3u]] << 24)
                            | ((uint64_t)valToBin[latents1[i + 4u]] << 32)
                            | ((uint64_t)valToBin[latents1[i + 5u]] << 40)
                            | ((uint64_t)valToBin[latents1[i + 6u]] << 48)
                            | ((uint64_t)valToBin[latents1[i + 7u]] << 56);
                    ZL_writeLE64(binDst + i, pack);
                }
                for (; i < n; i++) {
                    binDst[i] = valToBin[latents1[i]];
                }
                break;
            }
            case 2: {
                if (n >= 4096u && numBins > 32u) {
                    uint8_t* const valToBin =
                            (uint8_t*)ZL_Encoder_getScratchSpace(eictx, 65536u);
                    ZL_ERR_IF_NULL(valToBin, allocation);
                    for (size_t b = 0; b < numBins; b++) {
                        size_t const s = (b == 0) ? 0u : (size_t)lowers2[b];
                        size_t const e = (b + 1u < numBins)
                                ? (size_t)lowers2[b + 1u]
                                : 65536u;
                        if (e > s) {
                            memset(valToBin + s, (int)b, e - s);
                        }
                    }
                    for (size_t i = 0; i < n; i++) {
                        uint8_t const b = valToBin[latents2[i]];
                        binDst[i]       = b;
                        totalBits += offsetBits[b];
                    }
                } else {
                    totalBits = pbs_assign_bins_2(
                            lowers2, numBins, latents2, n, offsetBits, binDst);
                }
                break;
            }
            case 4:
                totalBits = pbs_assign_bins_4(
                        lowers4, numBins, latents4, n, offsetBits, binDst);
                break;
            case 8:
                totalBits = pbs_assign_bins_8(
                        lowers, numBins, latents8, n, offsetBits, binDst);
                break;
        }

        if (anyTightGap) {
            int widened = 0;
            switch (eltWidth) {
                case 2:
                    for (size_t i = 0; i < n; i++) {
                        uint8_t const b  = binDst[i];
                        uint64_t const v = latents2[i];
                        if (__builtin_expect(v > maxSafe[b], 0)) {
                            unsigned const ob = pbs_offset_bits(v - lowers[b]);
                            offsetBits[b]     = (uint8_t)ob;
                            maxSafe[b]        = (ob >= 16u)
                                           ? UINT64_MAX
                                           : pbs_sat_add_u64(lowers[b], pbs_mask64(ob));
                            if (ob > maxOb) maxOb = ob;
                            allZeroOffsets = 0;
                            widened        = 1;
                        }
                    }
                    break;
                case 4:
                    for (size_t i = 0; i < n; i++) {
                        uint8_t const b  = binDst[i];
                        uint64_t const v = latents4[i];
                        if (__builtin_expect(v > maxSafe[b], 0)) {
                            unsigned const ob = pbs_offset_bits(v - lowers[b]);
                            offsetBits[b]     = (uint8_t)ob;
                            maxSafe[b]        = (ob >= 32u)
                                           ? UINT64_MAX
                                           : pbs_sat_add_u64(lowers[b], pbs_mask64(ob));
                            if (ob > maxOb) maxOb = ob;
                            allZeroOffsets = 0;
                            widened        = 1;
                        }
                    }
                    break;
                case 8:
                    for (size_t i = 0; i < n; i++) {
                        uint8_t const b  = binDst[i];
                        uint64_t const v = latents8[i];
                        if (__builtin_expect(v > maxSafe[b], 0)) {
                            unsigned const ob = pbs_offset_bits(v - lowers[b]);
                            offsetBits[b]     = (uint8_t)ob;
                            maxSafe[b]        = (ob >= 64u)
                                           ? UINT64_MAX
                                           : pbs_sat_add_u64(lowers[b], pbs_mask64(ob));
                            if (ob > maxOb) maxOb = ob;
                            allZeroOffsets = 0;
                            widened        = 1;
                        }
                    }
                    break;
            }
            if (widened) {
                totalBits = 0;
                for (size_t i = 0; i < n; i++) {
                    totalBits += offsetBits[binDst[i]];
                }
            }
        }
    }

    size_t const packedBytes = (size_t)((totalBits + 7ULL) >> 3);
    ZL_Output* const offOut =
            ZL_Encoder_createTypedStream(eictx, 1, packedBytes, 1);
    ZL_ERR_IF_NULL(offOut, allocation);

    if (packedBytes > 0) {
        uint8_t* const dst = (uint8_t*)ZL_Output_ptr(offOut);
        switch (eltWidth) {
            case 1:
                pbs_pack_offsets_1(
                        dst,
                        packedBytes,
                        latents1,
                        n,
                        lowers1,
                        offsetBits,
                        binDst,
                        singleBin,
                        maxOb);
                break;
            case 2:
                pbs_pack_offsets_2(
                        dst,
                        packedBytes,
                        latents2,
                        n,
                        lowers2,
                        offsetBits,
                        binDst,
                        singleBin,
                        maxOb);
                break;
            case 4:
                pbs_pack_offsets_4(
                        dst,
                        packedBytes,
                        latents4,
                        n,
                        lowers4,
                        offsetBits,
                        binDst,
                        singleBin,
                        maxOb);
                break;
            case 8:
                pbs_pack_offsets_8(
                        dst,
                        packedBytes,
                        latents8,
                        n,
                        lowers,
                        offsetBits,
                        binDst,
                        singleBin,
                        maxOb);
                break;
        }
    }

    size_t const hdrCap =
            2u + ZL_VARINT_LENGTH_64 + numBins * (ZL_VARINT_LENGTH_64 + 1u);
    uint8_t* const hdr = (uint8_t*)ZL_Encoder_getScratchSpace(eictx, hdrCap);
    ZL_ERR_IF_NULL(hdr, allocation);

    size_t hsz = 0;
    hdr[hsz++] = (uint8_t)((eltWidthLog2 & 3u) | ((unsigned)(latentMode & 3) << 2)
                           | ((unsigned)singleBin << 4)
                           | ((unsigned)allZeroOffsets << 5));
    hsz += ZL_varintEncode((uint64_t)n, hdr + hsz);
    if (!singleBin) {
        hdr[hsz++] = (uint8_t)(numBins - 1u);
    }
    uint64_t prevLower = 0;
    for (size_t b = 0; b < numBins; b++) {
        hsz += ZL_varintEncode(lowers[b] - prevLower, hdr + hsz);
        prevLower = lowers[b];
        if (!allZeroOffsets) {
            hdr[hsz++] = offsetBits[b];
        }
    }
    ZL_Encoder_sendCodecHeader(eictx, hdr, hsz);
    ZL_ERR_IF_ERR(ZL_Output_commit(binOut, singleBin ? 0 : n));
    ZL_ERR_IF_ERR(ZL_Output_commit(offOut, packedBytes));
    return ZL_returnSuccess();
}
