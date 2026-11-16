// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/lookback_delta/encode_lookback_delta_binding.h"

#include <stdint.h>
#include <string.h>

#include "openzl/codecs/zl_lookback_delta.h"
#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
#include "openzl/shared/bits.h"
#include "openzl/shared/portability.h"

enum {
    LBD_MODE_RAW           = 0,
    LBD_MODE_CONSECUTIVE   = 1,
    LBD_MODE_LOOKBACK      = 2,
    LBD_MODE_CONSECUTIVE_2 = 3,
    LBD_MODE_CONSECUTIVE_3 = 4,

    LBD_PROPOSED_N         = 16,
    LBD_BRUTE_N            = 6,
    LBD_REPEATING_N        = 4,
    LBD_DEFAULT_WLOG       = 15,
    LBD_SAMPLE_BLOCK_N     = 512,
    LBD_SAMPLE_BLOCKS      = 3,
    LBD_SAMPLE_MAX_N       = 1536,
    LBD_EST_MAX_BINS       = 64,
    LBD_LB_PENALTY_BITS_X8 = 1,  /* 0.125 bits/elem tie-breaker against lookback */
};

static inline uint32_t lbd_hash(uint64_t x, size_t hashMask)
{
    x = (x ^ (x >> 32)) * 11400714819323197441ULL;
    x = x ^ (x >> 32);
    return (uint32_t)(x & (uint64_t)hashMask);
}

static inline unsigned lbd_clz8(uint8_t x)
{
#if defined(__LZCNT__) && (defined(__x86_64__) || defined(_M_X64))
    return (unsigned)_lzcnt_u32((uint32_t)x) - 24u;
#else
    return (unsigned)ZL_clz32(((uint32_t)x << 1) | 1u) - 23u;
#endif
}

static inline unsigned lbd_clz16(uint16_t x)
{
#if defined(__LZCNT__) && (defined(__x86_64__) || defined(_M_X64))
    return (unsigned)_lzcnt_u32((uint32_t)x) - 16u;
#else
    return (unsigned)ZL_clz32(((uint32_t)x << 1) | 1u) - 15u;
#endif
}

static inline unsigned lbd_clz32(uint32_t x)
{
#if defined(__LZCNT__) && (defined(__x86_64__) || defined(_M_X64))
    return (unsigned)_lzcnt_u32(x);
#else
    return (unsigned)ZL_clz64(((uint64_t)x << 1) | 1ULL) - 31u;
#endif
}

static inline unsigned lbd_clz64(uint64_t x)
{
#if defined(__LZCNT__) && (defined(__x86_64__) || defined(_M_X64))
    return (unsigned)_lzcnt_u64(x);
#else
    return x == 0 ? 64u : (unsigned)ZL_clz64(x);
#endif
}

static inline uint32_t lbd_log2_x8(uint32_t x)
{
    static const uint8_t frac[8] = { 0, 1, 3, 4, 5, 6, 7, 7 };
    unsigned const hb            = 31u - (unsigned)ZL_clz32(x);
    uint32_t const idx           = (hb >= 3u) ? ((x >> (hb - 3u)) & 7u)
                                              : ((x << (3u - hb)) & 7u);
    return (uint32_t)hb * 8u + (uint32_t)frac[idx];
}

static void lbd_sort_u64(
        uint64_t* arr,
        uint64_t* tmp,
        size_t m,
        size_t nBytes,
        uint64_t xorMask)
{
    if (xorMask != 0) {
        for (size_t i = 0; i < m; i++) {
            arr[i] ^= xorMask;
        }
    }
    if (m <= 1) {
        return;
    }
    if (m <= 64) {
        for (size_t i = 1; i < m; i++) {
            uint64_t const key = arr[i];
            size_t j           = i;
            while (j > 0 && arr[j - 1] > key) {
                arr[j] = arr[j - 1];
                j--;
            }
            arr[j] = key;
        }
        return;
    }

    uint32_t counts[8][256];
    memset(counts, 0, nBytes * sizeof(counts[0]));
    if (nBytes == 2) {
        for (size_t i = 0; i < m; i++) {
            uint64_t const v = arr[i];
            counts[0][v & 0xFFu]++;
            counts[1][(v >> 8) & 0xFFu]++;
        }
    } else if (nBytes == 4) {
        for (size_t i = 0; i < m; i++) {
            uint64_t const v = arr[i];
            counts[0][v & 0xFFu]++;
            counts[1][(v >> 8) & 0xFFu]++;
            counts[2][(v >> 16) & 0xFFu]++;
            counts[3][(v >> 24) & 0xFFu]++;
        }
    } else {
        for (size_t i = 0; i < m; i++) {
            uint64_t const v = arr[i];
            counts[0][v & 0xFFu]++;
            counts[1][(v >> 8) & 0xFFu]++;
            counts[2][(v >> 16) & 0xFFu]++;
            counts[3][(v >> 24) & 0xFFu]++;
            counts[4][(v >> 32) & 0xFFu]++;
            counts[5][(v >> 40) & 0xFFu]++;
            counts[6][(v >> 48) & 0xFFu]++;
            counts[7][(v >> 56) & 0xFFu]++;
        }
    }

    uint64_t const firstVal = arr[0];
    uint64_t* cur = arr;
    uint64_t* nxt = tmp;
    for (size_t pass = 0; pass < nBytes; pass++) {
        unsigned const shift    = (unsigned)(pass * 8u);
        uint8_t const firstByte = (uint8_t)((firstVal >> shift) & 0xFFu);
        if (counts[pass][firstByte] == m) {
            continue;
        }
        uint32_t offsets[256];
        uint32_t acc = 0;
        for (size_t b = 0; b < 256; b++) {
            offsets[b] = acc;
            acc += counts[pass][b];
        }
        for (size_t i = 0; i < m; i++) {
            uint64_t const v = cur[i];
            uint32_t const d = offsets[(v >> shift) & 0xFFu]++;
            nxt[d]           = v;
        }
        uint64_t* const sw = cur;
        cur                = nxt;
        nxt                = sw;
    }
    if (cur != arr) {
        memcpy(arr, cur, m * sizeof(uint64_t));
    }
}

typedef struct {
    uint64_t lo;
    uint64_t hi;
    uint32_t cnt;
} LbdBin;

static inline uint64_t lbd_bin_cost_x8(
        LbdBin b,
        uint32_t logM8,
        uint64_t metaBitsX8)
{
    uint64_t const span = b.hi - b.lo;
    unsigned const ob   = (span == 0) ? 0u : (64u - lbd_clz64(span));
    uint32_t const lc8  = lbd_log2_x8(b.cnt);
    uint32_t const ent8 = (logM8 > lc8) ? (logM8 - lc8) : 0u;
    return (uint64_t)b.cnt * (uint64_t)(ent8 + ob * 8u) + metaBitsX8;
}

static uint64_t lbd_est_rle_bin_bits_x8(
        uint64_t* buf,
        uint64_t* tmp,
        size_t m,
        size_t eltWidth,
        int isSigned)
{
    if (m == 0) {
        return 0;
    }
    unsigned const eltBits    = (unsigned)eltWidth * 8u;
    uint64_t const signBit    = 1ULL << (eltBits - 1u);
    uint64_t const metaBitsX8 = (uint64_t)(eltBits + 16u) * 8ULL;
    LbdBin bins[LBD_EST_MAX_BINS];
    size_t nBins = 0;

    if (eltWidth == 1) {
        uint32_t cnt1[256];
        memset(cnt1, 0, sizeof(cnt1));
        uint32_t const sMask = isSigned ? 0x80u : 0u;
        for (size_t i = 0; i < m; i++) {
            cnt1[((uint32_t)buf[i]) ^ sMask]++;
        }
        size_t nRuns = 0;
        for (size_t v = 0; v < 256; v++) {
            if (cnt1[v] != 0) {
                nRuns++;
            }
        }
        if (nRuns <= LBD_EST_MAX_BINS) {
            for (size_t v = 0; v < 256; v++) {
                if (cnt1[v] != 0) {
                    bins[nBins++] = (LbdBin){ (uint64_t)v, (uint64_t)v, cnt1[v] };
                }
            }
        } else {
            size_t remElts = m;
            for (size_t v = 0; v < 256; v++) {
                uint32_t const runC = cnt1[v];
                if (runC == 0) {
                    continue;
                }
                if (nBins == 0) {
                    bins[0] = (LbdBin){ (uint64_t)v, (uint64_t)v, runC };
                    nBins   = 1;
                } else {
                    size_t const remBins   = LBD_EST_MAX_BINS - nBins;
                    uint32_t const curTarg = (remBins > 0)
                            ? (uint32_t)((remElts + remBins) / (remBins + 1u))
                            : UINT32_MAX;
                    if (remBins == 0 || bins[nBins - 1].cnt + runC <= curTarg) {
                        bins[nBins - 1].hi = (uint64_t)v;
                        bins[nBins - 1].cnt += runC;
                    } else {
                        bins[nBins++] = (LbdBin){ (uint64_t)v, (uint64_t)v, runC };
                    }
                }
                remElts -= runC;
            }
        }
    } else {
        lbd_sort_u64(buf, tmp, m, eltWidth, isSigned ? signBit : 0ULL);

    size_t nRuns = 1;
        for (size_t i = 1; i < m; i++) {
            if (buf[i] != buf[i - 1]) {
                nRuns++;
            }
        }

        if (nRuns <= LBD_EST_MAX_BINS) {
            uint64_t curV = buf[0];
            uint32_t curC = 1u;
            for (size_t i = 1; i < m; i++) {
                if (buf[i] == curV) {
                    curC++;
                } else {
                    bins[nBins++] = (LbdBin){ curV, curV, curC };
                    curV          = buf[i];
                    curC          = 1u;
                }
            }
            bins[nBins++] = (LbdBin){ curV, curV, curC };
        } else {
            size_t remElts = m;
            size_t idx     = 0;
            while (idx < m) {
                uint64_t const runV = buf[idx];
                uint32_t runC       = 1u;
                while (idx + runC < m && buf[idx + runC] == runV) {
                    runC++;
                }
                if (nBins == 0) {
                    bins[0] = (LbdBin){ runV, runV, runC };
                    nBins   = 1;
                } else {
                    size_t const remBins   = LBD_EST_MAX_BINS - nBins;
                    uint32_t const curTarg = (remBins > 0)
                            ? (uint32_t)((remElts + remBins) / (remBins + 1u))
                            : UINT32_MAX;
                    if (remBins == 0 || bins[nBins - 1].cnt + runC <= curTarg) {
                        bins[nBins - 1].hi = runV;
                        bins[nBins - 1].cnt += runC;
                    } else {
                        bins[nBins++] = (LbdBin){ runV, runV, runC };
                    }
                }
                remElts -= runC;
                idx += runC;
            }
        }
    }

    uint32_t const logM8 = lbd_log2_x8((uint32_t)m);
    int changed          = 1;
    while (changed && nBins > 1) {
        changed       = 0;
        size_t outIdx = 1;
        for (size_t i = 1; i < nBins; i++) {
            LbdBin const cur    = bins[outIdx - 1];
            LbdBin const nxt    = bins[i];
            LbdBin const merged = { cur.lo, nxt.hi, cur.cnt + nxt.cnt };
            if (lbd_bin_cost_x8(merged, logM8, metaBitsX8)
                <= lbd_bin_cost_x8(cur, logM8, metaBitsX8)
                        + lbd_bin_cost_x8(nxt, logM8, metaBitsX8)) {
                bins[outIdx - 1] = merged;
                changed          = 1;
            } else {
                bins[outIdx++] = nxt;
            }
        }
        nBins = outIdx;
    }

    uint64_t totalX8 = 0;
    for (size_t i = 0; i < nBins; i++) {
        totalX8 += lbd_bin_cost_x8(bins[i], logM8, metaBitsX8);
    }
    return totalX8;
}

#define LBD_DEFINE_WIDTH(W, T, CLZ_FN, EBITS)                                  \
    static void lbd_consec1_##W(T* dst, const T* src, size_t n)                \
    {                                                                          \
        dst[0] = src[0];                                                       \
        for (size_t i = 1; i < n; i++) {                                       \
            dst[i] = (T)(src[i] - src[i - 1]);                                 \
        }                                                                      \
    }                                                                          \
    static void lbd_consec2_##W(T* dst, const T* src, size_t n)                \
    {                                                                          \
        dst[0] = src[0];                                                       \
        if (n > 1) {                                                           \
            T d1_prev = (T)(src[1] - src[0]);                                  \
            dst[1]    = d1_prev;                                               \
            for (size_t i = 2; i < n; i++) {                                   \
                T const d1 = (T)(src[i] - src[i - 1]);                         \
                dst[i]     = (T)(d1 - d1_prev);                                \
                d1_prev    = d1;                                               \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    static void lbd_consec3_##W(T* dst, const T* src, size_t n)                \
    {                                                                          \
        dst[0] = src[0];                                                       \
        if (n > 1) {                                                           \
            T d1_prev = (T)(src[1] - src[0]);                                  \
            dst[1]    = d1_prev;                                               \
            if (n > 2) {                                                       \
                T const d1_2 = (T)(src[2] - src[1]);                           \
                T d2_prev    = (T)(d1_2 - d1_prev);                            \
                dst[2]       = d2_prev;                                        \
                d1_prev      = d1_2;                                           \
                for (size_t i = 3; i < n; i++) {                               \
                    T const d1 = (T)(src[i] - src[i - 1]);                     \
                    T const d2 = (T)(d1 - d1_prev);                            \
                    dst[i]     = (T)(d2 - d2_prev);                            \
                    d1_prev    = d1;                                           \
                    d2_prev    = d2;                                           \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    static void lbd_encode_lb_##W(                                             \
            T* resDst,                                                         \
            uint16_t* lbDst,                                                   \
            const T* src,                                                      \
            size_t n,                                                          \
            size_t windowN,                                                    \
            size_t hashTableN,                                                 \
            uint32_t* idxHashTable,                                            \
            uint32_t* lbCounts,                                                \
            uint8_t* lbGoodness)                                               \
    {                                                                          \
        size_t const hashMask = hashTableN - 1;                                \
        size_t const nLb      = windowN < n ? windowN : n;                     \
        memset(lbGoodness, 1, nLb);                                            \
        for (size_t k = 0; k < nLb; k++) {                                     \
            lbCounts[k] = 1;                                                   \
        }                                                                      \
        memset(idxHashTable, 0, 2 * hashTableN * sizeof(uint32_t));            \
        uint32_t proposed[LBD_PROPOSED_N];                                     \
        for (size_t k = 0; k < LBD_PROPOSED_N; k++) {                          \
            proposed[k] = 1u;                                                  \
        }                                                                      \
                                                                               \
        resDst[0]                  = src[0];                                   \
        uint32_t bestLb            = 1u;                                       \
        size_t repeatingIdx        = 0;                                        \
        unsigned maxNonOneGoodness = 1u;                                       \
        uint32_t* const ht1        = idxHashTable + hashTableN;                \
                                                                               \
        size_t const prefixEnd = (n - 1u < (size_t)LBD_PROPOSED_N)             \
                ? n                                                            \
                : (size_t)LBD_PROPOSED_N + 1u;                                 \
        for (size_t i = 1; i < prefixEnd; i++) {                               \
            T const l       = src[i];                                          \
            proposed[i - 1] = (uint32_t)i;                                     \
            uint64_t const lu64 = (uint64_t)l;                                 \
            uint64_t const b1   = lu64 >> 8;                                   \
            T const prev1       = src[i - 1];                                  \
            T const c1Fwd       = (T)(l - prev1);                              \
            T const c1Rev       = (T)(prev1 - l);                              \
            T const c1Abs       = c1Fwd < c1Rev ? c1Fwd : c1Rev;               \
            unsigned const c1g  = CLZ_FN(c1Abs);                               \
            unsigned bestGoodness = (unsigned)lbGoodness[0] + c1g;             \
            {                                                                  \
                uint32_t const h0 = lbd_hash(lu64 - 1ULL, hashMask);           \
                uint32_t const h1 = lbd_hash(lu64, hashMask);                  \
                uint32_t const h2 = lbd_hash(lu64 + 1ULL, hashMask);           \
                size_t const d0   = i - (size_t)idxHashTable[h0];              \
                size_t const d1   = i - (size_t)idxHashTable[h1];              \
                size_t const d2   = i - (size_t)idxHashTable[h2];              \
                proposed[10]      = (uint32_t)(d0 <= windowN ? d0 : (10u < i ? 10u : i)); \
                proposed[11]      = (uint32_t)(d1 <= windowN ? d1 : (11u < i ? 11u : i)); \
                proposed[12]      = (uint32_t)(d2 <= windowN ? d2 : (12u < i ? 12u : i)); \
                idxHashTable[h1]  = (uint32_t)i;                               \
            }                                                                  \
            {                                                                  \
                uint32_t const h0 = lbd_hash(b1 - 1ULL, hashMask);             \
                uint32_t const h1 = lbd_hash(b1, hashMask);                    \
                uint32_t const h2 = lbd_hash(b1 + 1ULL, hashMask);             \
                size_t const d0   = i - (size_t)ht1[h0];                       \
                size_t const d1   = i - (size_t)ht1[h1];                       \
                size_t const d2   = i - (size_t)ht1[h2];                       \
                proposed[13]      = (uint32_t)(d0 <= windowN ? d0 : (13u < i ? 13u : i)); \
                proposed[14]      = (uint32_t)(d1 <= windowN ? d1 : (14u < i ? 14u : i)); \
                proposed[15]      = (uint32_t)(d2 <= windowN ? d2 : (15u < i ? 15u : i)); \
                ht1[h1]           = (uint32_t)i;                               \
            }                                                                  \
            uint32_t newBestLb = 1u;                                           \
            for (size_t k = 1; k < LBD_PROPOSED_N; k++) {                      \
                uint32_t const cand = proposed[k];                             \
                T const other       = src[i - cand];                           \
                T const dFwd        = (T)(l - other);                          \
                T const dRev        = (T)(other - l);                          \
                T const dAbs        = dFwd < dRev ? dFwd : dRev;               \
                unsigned const dg   = CLZ_FN(dAbs);                            \
                unsigned const g    = (unsigned)lbGoodness[cand - 1] + dg;    \
                if (g > bestGoodness) {                                        \
                    bestGoodness = g;                                          \
                    newBestLb    = cand;                                       \
                }                                                              \
            }                                                                  \
            if (newBestLb != bestLb) {                                         \
                repeatingIdx++;                                                \
            }                                                                  \
            proposed[LBD_BRUTE_N + (repeatingIdx & 3u)] = newBestLb;           \
            bestLb                                      = newBestLb;           \
            resDst[i]          = (T)(l - src[i - newBestLb]);                  \
            lbDst[i - 1]       = (uint16_t)(newBestLb - 1u);                   \
            uint32_t const cnt = ++lbCounts[newBestLb - 1u];                   \
            if (ZL_UNLIKELY((cnt & (cnt - 1u)) == 0u)) {                       \
                uint8_t const ng = ++lbGoodness[newBestLb - 1u];               \
                if (newBestLb != 1u && (unsigned)ng > maxNonOneGoodness) {     \
                    maxNonOneGoodness = (unsigned)ng;                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
                                                                               \
        uint32_t rep0 = proposed[6];                                           \
        uint32_t rep1 = proposed[7];                                           \
        uint32_t rep2 = proposed[8];                                           \
        uint32_t rep3 = proposed[9];                                           \
        for (size_t i = prefixEnd; i < n; i++) {                               \
            T const l           = src[i];                                      \
            uint64_t const lu64 = (uint64_t)l;                                 \
            uint64_t const b1   = lu64 >> 8;                                   \
            T const prev1       = src[i - 1];                                  \
            T const c1Fwd       = (T)(l - prev1);                              \
            T const c1Rev       = (T)(prev1 - l);                              \
            T const c1Abs       = c1Fwd < c1Rev ? c1Fwd : c1Rev;               \
            unsigned const c1g  = CLZ_FN(c1Abs);                               \
            unsigned bestGoodness = (unsigned)lbGoodness[0] + c1g;             \
                                                                               \
            if (ZL_UNLIKELY(bestGoodness >= maxNonOneGoodness + (EBITS))) {    \
                idxHashTable[lbd_hash(lu64, hashMask)] = (uint32_t)i;          \
                ht1[lbd_hash(b1, hashMask)]            = (uint32_t)i;          \
                if (bestLb != 1u) {                                            \
                    repeatingIdx++;                                            \
                    bestLb = 1u;                                               \
                }                                                              \
                switch (repeatingIdx & 3u) {                                   \
                    case 0: rep0 = 1u; break;                                  \
                    case 1: rep1 = 1u; break;                                  \
                    case 2: rep2 = 1u; break;                                  \
                    case 3: rep3 = 1u; break;                                  \
                }                                                              \
                resDst[i]          = c1Fwd;                                    \
                lbDst[i - 1]       = 0u;                                       \
                uint32_t const cnt = ++lbCounts[0];                            \
                if (ZL_UNLIKELY((cnt & (cnt - 1u)) == 0u)) {                   \
                    lbGoodness[0]++;                                           \
                }                                                              \
                continue;                                                      \
            }                                                                  \
                                                                               \
            uint32_t const h0a = lbd_hash(lu64 - 1ULL, hashMask);              \
            uint32_t const h1a = lbd_hash(lu64, hashMask);                     \
            uint32_t const h2a = lbd_hash(lu64 + 1ULL, hashMask);              \
            uint32_t const h0b = lbd_hash(b1 - 1ULL, hashMask);                \
            uint32_t const h1b = lbd_hash(b1, hashMask);                       \
            uint32_t const h2b = lbd_hash(b1 + 1ULL, hashMask);                \
            size_t const d0a   = i - (size_t)idxHashTable[h0a];                \
            size_t const d1a   = i - (size_t)idxHashTable[h1a];                \
            size_t const d2a   = i - (size_t)idxHashTable[h2a];                \
            idxHashTable[h1a]  = (uint32_t)i;                                  \
            size_t const d0b   = i - (size_t)ht1[h0b];                         \
            size_t const d1b   = i - (size_t)ht1[h1b];                         \
            size_t const d2b   = i - (size_t)ht1[h2b];                         \
            ht1[h1b]           = (uint32_t)i;                                  \
                                                                               \
            uint32_t const cands[15] = {                                       \
                2u, 3u, 4u, 5u, 6u,                                            \
                rep0, rep1, rep2, rep3,                                        \
                (uint32_t)(d0a <= windowN ? d0a : 10u),                        \
                (uint32_t)(d1a <= windowN ? d1a : 11u),                        \
                (uint32_t)(d2a <= windowN ? d2a : 12u),                        \
                (uint32_t)(d0b <= windowN ? d0b : 13u),                        \
                (uint32_t)(d1b <= windowN ? d1b : 14u),                        \
                (uint32_t)(d2b <= windowN ? d2b : 15u),                        \
            };                                                                 \
                                                                               \
            uint32_t newBestLb = 1u;                                           \
            for (size_t k = 0; k < 15; k++) {                                  \
                uint32_t const cand = cands[k];                                \
                T const other       = src[i - cand];                           \
                T const dFwd        = (T)(l - other);                          \
                T const dRev        = (T)(other - l);                          \
                T const dAbs        = dFwd < dRev ? dFwd : dRev;               \
                unsigned const dg   = CLZ_FN(dAbs);                            \
                unsigned const g    = (unsigned)lbGoodness[cand - 1] + dg;    \
                if (g > bestGoodness) {                                        \
                    bestGoodness = g;                                          \
                    newBestLb    = cand;                                       \
                }                                                              \
            }                                                                  \
                                                                               \
            if (newBestLb != bestLb) {                                         \
                repeatingIdx++;                                                \
            }                                                                  \
            switch (repeatingIdx & 3u) {                                       \
                case 0: rep0 = newBestLb; break;                               \
                case 1: rep1 = newBestLb; break;                               \
                case 2: rep2 = newBestLb; break;                               \
                case 3: rep3 = newBestLb; break;                               \
            }                                                                  \
            bestLb             = newBestLb;                                    \
            resDst[i]          = (T)(l - src[i - newBestLb]);                  \
            lbDst[i - 1]       = (uint16_t)(newBestLb - 1u);                   \
            uint32_t const cnt = ++lbCounts[newBestLb - 1u];                   \
            if (ZL_UNLIKELY((cnt & (cnt - 1u)) == 0u)) {                       \
                uint8_t const ng = ++lbGoodness[newBestLb - 1u];               \
                if (newBestLb != 1u && (unsigned)ng > maxNonOneGoodness) {     \
                    maxNonOneGoodness = (unsigned)ng;                          \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    static int lbd_choose_and_encode_##W(                                      \
            T* resDst,                                                         \
            uint16_t* lbDst,                                                   \
            const T* src,                                                      \
            size_t n,                                                          \
            int windowLog,                                                     \
            uint32_t* idxHashTable,                                            \
            uint32_t* lbCounts,                                                \
            uint8_t* lbGoodness,                                               \
            uint64_t* sortBuf,                                                 \
            uint64_t* tmpBuf,                                                  \
            T* sampBuf,                                                        \
            T* lbResSamp,                                                      \
            uint16_t* lbDistSamp)                                              \
    {                                                                          \
        size_t offs[LBD_SAMPLE_BLOCKS];                                        \
        size_t bLens[LBD_SAMPLE_BLOCKS];                                       \
        size_t nBlocks = 0;                                                    \
        size_t m       = 0;                                                    \
        if (n <= LBD_SAMPLE_MAX_N) {                                           \
            offs[0]  = 0;                                                      \
            bLens[0] = n;                                                      \
            nBlocks  = 1;                                                      \
            m        = n;                                                      \
        } else {                                                               \
            offs[0]  = 0;                                                      \
            offs[1]  = n / 2 - LBD_SAMPLE_BLOCK_N / 2;                         \
            offs[2]  = n - LBD_SAMPLE_BLOCK_N;                                 \
            bLens[0] = LBD_SAMPLE_BLOCK_N;                                     \
            bLens[1] = LBD_SAMPLE_BLOCK_N;                                     \
            bLens[2] = LBD_SAMPLE_BLOCK_N;                                     \
            nBlocks  = LBD_SAMPLE_BLOCKS;                                      \
            m        = LBD_SAMPLE_MAX_N;                                       \
        }                                                                      \
                                                                               \
        /* 0. RAW */                                                           \
        {                                                                      \
            size_t pos = 0;                                                    \
            for (size_t b = 0; b < nBlocks; b++) {                             \
                const T* const blk = src + offs[b];                            \
                size_t const bl    = bLens[b];                                 \
                for (size_t i = 0; i < bl; i++) {                              \
                    sortBuf[pos++] = (uint64_t)blk[i];                         \
                }                                                              \
            }                                                                  \
        }                                                                      \
        uint64_t bestCostX8 =                                                  \
                (lbd_est_rle_bin_bits_x8(sortBuf, tmpBuf, m, W, 0)             \
                 * (uint64_t)n)                                                \
                / (uint64_t)m;                                                 \
        int bestMode = LBD_MODE_RAW;                                           \
                                                                               \
        /* 1. CONSECUTIVE-1 */                                                 \
        if (n >= 2) {                                                          \
            size_t m1 = 0;                                                     \
            for (size_t b = 0; b < nBlocks; b++) {                             \
                const T* const blk = src + offs[b];                            \
                size_t const bl    = bLens[b];                                 \
                for (size_t i = 1; i < bl; i++) {                              \
                    sortBuf[m1++] = (uint64_t)(T)(blk[i] - blk[i - 1]);        \
                }                                                              \
            }                                                                  \
            uint64_t const c1X8 = (uint64_t)(EBITS) * 8ULL                     \
                    + (lbd_est_rle_bin_bits_x8(sortBuf, tmpBuf, m1, W, 1)      \
                       * (uint64_t)(n - 1))                                    \
                            / (uint64_t)m1;                                    \
            if (c1X8 < bestCostX8) {                                           \
                bestCostX8 = c1X8;                                             \
                bestMode   = LBD_MODE_CONSECUTIVE;                             \
            }                                                                  \
        }                                                                      \
                                                                               \
        /* 2. CONSECUTIVE-2 */                                                 \
        if (n >= 3) {                                                          \
            size_t m2 = 0;                                                     \
            for (size_t b = 0; b < nBlocks; b++) {                             \
                const T* const blk = src + offs[b];                            \
                size_t const bl    = bLens[b];                                 \
                if (bl >= 3) {                                                 \
                    T d1p = (T)(blk[1] - blk[0]);                              \
                    for (size_t i = 2; i < bl; i++) {                          \
                        T const d1    = (T)(blk[i] - blk[i - 1]);              \
                        sortBuf[m2++] = (uint64_t)(T)(d1 - d1p);               \
                        d1p           = d1;                                    \
                    }                                                          \
                }                                                              \
            }                                                                  \
            if (m2 > 0) {                                                      \
                uint64_t const c2X8 = 2ULL * (uint64_t)(EBITS) * 8ULL          \
                        + (lbd_est_rle_bin_bits_x8(sortBuf, tmpBuf, m2, W, 1)  \
                           * (uint64_t)(n - 2))                                \
                                / (uint64_t)m2;                                \
                if (c2X8 < bestCostX8) {                                       \
                    bestCostX8 = c2X8;                                         \
                    bestMode   = LBD_MODE_CONSECUTIVE_2;                       \
                }                                                              \
            }                                                                  \
        }                                                                      \
                                                                               \
        /* 3. CONSECUTIVE-3 */                                                 \
        if (n >= 4) {                                                          \
            size_t m3 = 0;                                                     \
            for (size_t b = 0; b < nBlocks; b++) {                             \
                const T* const blk = src + offs[b];                            \
                size_t const bl    = bLens[b];                                 \
                if (bl >= 4) {                                                 \
                    T d1p = (T)(blk[1] - blk[0]);                              \
                    T d1_2 = (T)(blk[2] - blk[1]);                             \
                    T d2p = (T)(d1_2 - d1p);                                   \
                    d1p   = d1_2;                                              \
                    for (size_t i = 3; i < bl; i++) {                          \
                        T const d1    = (T)(blk[i] - blk[i - 1]);              \
                        T const d2    = (T)(d1 - d1p);                         \
                        sortBuf[m3++] = (uint64_t)(T)(d2 - d2p);               \
                        d1p           = d1;                                    \
                        d2p           = d2;                                    \
                    }                                                          \
                }                                                              \
            }                                                                  \
            if (m3 > 0) {                                                      \
                uint64_t const c3X8 = 3ULL * (uint64_t)(EBITS) * 8ULL          \
                        + (lbd_est_rle_bin_bits_x8(sortBuf, tmpBuf, m3, W, 1)  \
                           * (uint64_t)(n - 3))                                \
                                / (uint64_t)m3;                                \
                if (c3X8 < bestCostX8) {                                       \
                    bestCostX8 = c3X8;                                         \
                    bestMode   = LBD_MODE_CONSECUTIVE_3;                       \
                }                                                              \
            }                                                                  \
        }                                                                      \
                                                                               \
        /* 4. LOOKBACK (only if n > 16 and best consec/raw can be beaten by >= 2 b/elt) */ \
        uint64_t const lbMinX8 = (uint64_t)(EBITS) * 8ULL                      \
                + (uint64_t)LBD_LB_PENALTY_BITS_X8 * (uint64_t)(n - 1);        \
        uint64_t const lbSkipX8 = (uint64_t)(EBITS) * 8ULL                     \
                + 2ULL * (uint64_t)(n - 1);                                    \
        if (n > LBD_PROPOSED_N && bestCostX8 > lbSkipX8) {                      \
            size_t const fullWindowN    = (size_t)1u << windowLog;             \
            size_t const fullHashTableN = (size_t)1u << (windowLog + 1);       \
            if (n <= LBD_SAMPLE_MAX_N) {                                       \
                lbd_encode_lb_##W(                                             \
                        resDst,                                                \
                        lbDst,                                                 \
                        src,                                                   \
                        n,                                                     \
                        fullWindowN,                                           \
                        fullHashTableN,                                        \
                        idxHashTable,                                          \
                        lbCounts,                                              \
                        lbGoodness);                                           \
                for (size_t i = 1; i < n; i++) {                               \
                    sortBuf[i - 1] = (uint64_t)resDst[i];                      \
                }                                                              \
                uint64_t const resCostX8 = lbd_est_rle_bin_bits_x8(            \
                        sortBuf, tmpBuf, n - 1, W, 1);                         \
                for (size_t i = 0; i < n - 1; i++) {                           \
                    sortBuf[i] = (uint64_t)lbDst[i];                           \
                }                                                              \
                uint64_t const distCostX8 = lbd_est_rle_bin_bits_x8(           \
                        sortBuf, tmpBuf, n - 1, 2, 0);                         \
                uint64_t const lbCostX8 = lbMinX8 + resCostX8 + distCostX8;    \
                if (lbCostX8 < bestCostX8) {                                   \
                    return LBD_MODE_LOOKBACK;                                  \
                }                                                              \
            } else {                                                           \
                size_t pos = 0;                                                \
                for (size_t b = 0; b < nBlocks; b++) {                         \
                    memcpy(sampBuf + pos,                                      \
                           src + offs[b],                                      \
                           bLens[b] * sizeof(T));                              \
                    pos += bLens[b];                                           \
                }                                                              \
                int const sampWLog = windowLog < 12 ? windowLog : 12;          \
                size_t const sampWindowN    = (size_t)1u << sampWLog;          \
                size_t const sampHashTableN = (size_t)1u << (sampWLog + 1);    \
                lbd_encode_lb_##W(                                             \
                        lbResSamp,                                             \
                        lbDistSamp,                                            \
                        sampBuf,                                               \
                        m,                                                     \
                        sampWindowN,                                           \
                        sampHashTableN,                                        \
                        idxHashTable,                                          \
                        lbCounts,                                              \
                        lbGoodness);                                           \
                for (size_t i = 1; i < m; i++) {                               \
                    sortBuf[i - 1] = (uint64_t)lbResSamp[i];                   \
                }                                                              \
                uint64_t const resCostX8 = lbd_est_rle_bin_bits_x8(            \
                        sortBuf, tmpBuf, m - 1, W, 1);                         \
                for (size_t i = 0; i < m - 1; i++) {                           \
                    sortBuf[i] = (uint64_t)lbDistSamp[i];                      \
                }                                                              \
                uint64_t const distCostX8 = lbd_est_rle_bin_bits_x8(           \
                        sortBuf, tmpBuf, m - 1, 2, 0);                         \
                uint64_t const lbCostX8 = lbMinX8                              \
                        + ((resCostX8 + distCostX8) * (uint64_t)(n - 1))       \
                                / (uint64_t)(m - 1);                           \
                if (lbCostX8 < bestCostX8) {                                   \
                    lbd_encode_lb_##W(                                         \
                            resDst,                                            \
                            lbDst,                                             \
                            src,                                               \
                            n,                                                 \
                            fullWindowN,                                       \
                            fullHashTableN,                                    \
                            idxHashTable,                                      \
                            lbCounts,                                          \
                            lbGoodness);                                       \
                    return LBD_MODE_LOOKBACK;                                  \
                }                                                              \
            }                                                                  \
        }                                                                      \
                                                                               \
        switch (bestMode) {                                                    \
            case LBD_MODE_RAW:                                                 \
                memcpy(resDst, src, n * sizeof(T));                            \
                break;                                                         \
            case LBD_MODE_CONSECUTIVE:                                         \
                lbd_consec1_##W(resDst, src, n);                               \
                break;                                                         \
            case LBD_MODE_CONSECUTIVE_2:                                       \
                lbd_consec2_##W(resDst, src, n);                               \
                break;                                                         \
            case LBD_MODE_CONSECUTIVE_3:                                       \
                lbd_consec3_##W(resDst, src, n);                               \
                break;                                                         \
        }                                                                      \
        return bestMode;                                                       \
    }

LBD_DEFINE_WIDTH(1, uint8_t, lbd_clz8, 8u)
LBD_DEFINE_WIDTH(2, uint16_t, lbd_clz16, 16u)
LBD_DEFINE_WIDTH(4, uint32_t, lbd_clz32, 32u)
LBD_DEFINE_WIDTH(8, uint64_t, lbd_clz64, 64u)

ZL_Report EI_lookback_delta(
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
    ZL_ERR_IF_GT(n, (size_t)UINT32_MAX, node_invalid_input);

    int windowLog = LBD_DEFAULT_WLOG;
    ZL_IntParam const wlParam =
            ZL_Encoder_getLocalIntParam(eictx, ZL_LOOKBACK_DELTA_WINDOW_LOG_PID);
    if (wlParam.paramId == ZL_LOOKBACK_DELTA_WINDOW_LOG_PID) {
        ZL_ERR_IF(
                wlParam.paramValue < 4 || wlParam.paramValue > 16,
                nodeParameter_invalidValue);
        windowLog = wlParam.paramValue;
    }

    int forceLookback = 0;
    ZL_IntParam const forceParam =
            ZL_Encoder_getLocalIntParam(eictx, ZL_LOOKBACK_DELTA_FORCE_PID);
    if (forceParam.paramId == ZL_LOOKBACK_DELTA_FORCE_PID) {
        forceLookback = forceParam.paramValue != 0;
    }

    ZL_Output* const resOut = ZL_Encoder_createTypedStream(eictx, 0, n, eltWidth);
    ZL_ERR_IF_NULL(resOut, allocation);

    size_t const lbCap = (n > LBD_PROPOSED_N) ? (n - 1) : 0;
    ZL_Output* const lbOut = ZL_Encoder_createTypedStream(eictx, 1, lbCap, 2);
    ZL_ERR_IF_NULL(lbOut, allocation);

    if (n <= 1) {
        if (n == 1) {
            memcpy(ZL_Output_ptr(resOut), ZL_Input_ptr(in), eltWidth);
        }
        uint8_t const hdr = (uint8_t)LBD_MODE_RAW;
        ZL_Encoder_sendCodecHeader(eictx, &hdr, 1);
        ZL_ERR_IF_ERR(ZL_Output_commit(resOut, n));
        ZL_ERR_IF_ERR(ZL_Output_commit(lbOut, 0));
        return ZL_returnSuccess();
    }

    int effWindowLog = windowLog;
    {
        unsigned const nBits = 32u - (unsigned)ZL_clz32((uint32_t)(n - 1));
        int const minLog     = nBits > 4u ? (int)nBits : 4;
        if (minLog < effWindowLog) {
            effWindowLog = minLog;
        }
    }

    size_t const windowN    = (size_t)1u << effWindowLog;
    size_t const hashTableN = (size_t)1u << (effWindowLog + 1);
    size_t const nLb        = windowN < n ? windowN : n;

    uint32_t* idxHashTable = NULL;
    uint32_t* lbCounts     = NULL;
    uint8_t* lbGoodness    = NULL;
    if (n > LBD_PROPOSED_N) {
        idxHashTable = (uint32_t*)ZL_Encoder_getScratchSpace(
                eictx, 2 * hashTableN * sizeof(uint32_t));
        lbCounts = (uint32_t*)ZL_Encoder_getScratchSpace(
                eictx, nLb * sizeof(uint32_t));
        lbGoodness = (uint8_t*)ZL_Encoder_getScratchSpace(
                eictx, nLb * sizeof(uint8_t));
        ZL_ERR_IF(
                idxHashTable == NULL || lbCounts == NULL || lbGoodness == NULL,
                allocation);
    }

    if (forceLookback && n > LBD_PROPOSED_N) {
        switch (eltWidth) {
            case 1:
                lbd_encode_lb_1(
                        (uint8_t*)ZL_Output_ptr(resOut),
                        (uint16_t*)ZL_Output_ptr(lbOut),
                        (const uint8_t*)ZL_Input_ptr(in),
                        n,
                        windowN,
                        hashTableN,
                        idxHashTable,
                        lbCounts,
                        lbGoodness);
                break;
            case 2:
                lbd_encode_lb_2(
                        (uint16_t*)ZL_Output_ptr(resOut),
                        (uint16_t*)ZL_Output_ptr(lbOut),
                        (const uint16_t*)ZL_Input_ptr(in),
                        n,
                        windowN,
                        hashTableN,
                        idxHashTable,
                        lbCounts,
                        lbGoodness);
                break;
            case 4:
                lbd_encode_lb_4(
                        (uint32_t*)ZL_Output_ptr(resOut),
                        (uint16_t*)ZL_Output_ptr(lbOut),
                        (const uint32_t*)ZL_Input_ptr(in),
                        n,
                        windowN,
                        hashTableN,
                        idxHashTable,
                        lbCounts,
                        lbGoodness);
                break;
            case 8:
                lbd_encode_lb_8(
                        (uint64_t*)ZL_Output_ptr(resOut),
                        (uint16_t*)ZL_Output_ptr(lbOut),
                        (const uint64_t*)ZL_Input_ptr(in),
                        n,
                        windowN,
                        hashTableN,
                        idxHashTable,
                        lbCounts,
                        lbGoodness);
                break;
        }
        uint8_t const hdr = (uint8_t)LBD_MODE_LOOKBACK;
        ZL_Encoder_sendCodecHeader(eictx, &hdr, 1);
        ZL_ERR_IF_ERR(ZL_Output_commit(resOut, n));
        ZL_ERR_IF_ERR(ZL_Output_commit(lbOut, n - 1));
        return ZL_returnSuccess();
    }

    size_t const mMax = n <= LBD_SAMPLE_MAX_N ? n : LBD_SAMPLE_MAX_N;
    uint64_t* const sortBuf = (uint64_t*)ZL_Encoder_getScratchSpace(
            eictx, mMax * sizeof(uint64_t));
    uint64_t* const tmpBuf = (uint64_t*)ZL_Encoder_getScratchSpace(
            eictx, mMax * sizeof(uint64_t));
    ZL_ERR_IF(sortBuf == NULL || tmpBuf == NULL, allocation);

    void* sampBuf        = NULL;
    void* lbResSamp      = NULL;
    uint16_t* lbDistSamp = NULL;
    if (n > LBD_SAMPLE_MAX_N) {
        sampBuf = ZL_Encoder_getScratchSpace(eictx, mMax * eltWidth);
        lbResSamp = ZL_Encoder_getScratchSpace(eictx, mMax * eltWidth);
        lbDistSamp = (uint16_t*)ZL_Encoder_getScratchSpace(
                eictx, mMax * sizeof(uint16_t));
        ZL_ERR_IF(
                sampBuf == NULL || lbResSamp == NULL || lbDistSamp == NULL,
                allocation);
    }

    int mode = LBD_MODE_RAW;
    switch (eltWidth) {
        case 1:
            mode = lbd_choose_and_encode_1(
                    (uint8_t*)ZL_Output_ptr(resOut),
                    (uint16_t*)ZL_Output_ptr(lbOut),
                    (const uint8_t*)ZL_Input_ptr(in),
                    n,
                    effWindowLog,
                    idxHashTable,
                    lbCounts,
                    lbGoodness,
                    sortBuf,
                    tmpBuf,
                    (uint8_t*)sampBuf,
                    (uint8_t*)lbResSamp,
                    lbDistSamp);
            break;
        case 2:
            mode = lbd_choose_and_encode_2(
                    (uint16_t*)ZL_Output_ptr(resOut),
                    (uint16_t*)ZL_Output_ptr(lbOut),
                    (const uint16_t*)ZL_Input_ptr(in),
                    n,
                    effWindowLog,
                    idxHashTable,
                    lbCounts,
                    lbGoodness,
                    sortBuf,
                    tmpBuf,
                    (uint16_t*)sampBuf,
                    (uint16_t*)lbResSamp,
                    lbDistSamp);
            break;
        case 4:
            mode = lbd_choose_and_encode_4(
                    (uint32_t*)ZL_Output_ptr(resOut),
                    (uint16_t*)ZL_Output_ptr(lbOut),
                    (const uint32_t*)ZL_Input_ptr(in),
                    n,
                    effWindowLog,
                    idxHashTable,
                    lbCounts,
                    lbGoodness,
                    sortBuf,
                    tmpBuf,
                    (uint32_t*)sampBuf,
                    (uint32_t*)lbResSamp,
                    lbDistSamp);
            break;
        case 8:
            mode = lbd_choose_and_encode_8(
                    (uint64_t*)ZL_Output_ptr(resOut),
                    (uint16_t*)ZL_Output_ptr(lbOut),
                    (const uint64_t*)ZL_Input_ptr(in),
                    n,
                    effWindowLog,
                    idxHashTable,
                    lbCounts,
                    lbGoodness,
                    sortBuf,
                    tmpBuf,
                    (uint64_t*)sampBuf,
                    (uint64_t*)lbResSamp,
                    lbDistSamp);
            break;
    }

    uint8_t const hdr = (uint8_t)mode;
    ZL_Encoder_sendCodecHeader(eictx, &hdr, 1);
    ZL_ERR_IF_ERR(ZL_Output_commit(resOut, n));
    ZL_ERR_IF_ERR(
            ZL_Output_commit(lbOut, mode == LBD_MODE_LOOKBACK ? (n - 1) : 0));
    return ZL_returnSuccess();
}
