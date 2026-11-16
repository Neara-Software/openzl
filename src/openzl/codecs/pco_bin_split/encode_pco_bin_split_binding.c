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
#include "openzl/shared/varint.h"

enum {
    PBS_MAX_BINS_LOG = 8,
    PBS_MAX_BINS     = 256,
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

static inline float pbs_log2_approx(float x)
{
    const float Z              = 0.674f;
    const uint32_t SIGNIF_MASK = 0x7FFFFFu;
    uint32_t zBits             = 0;
    memcpy(&zBits, &Z, sizeof(zBits));
    uint32_t const zSignif = zBits & SIGNIF_MASK;

    const float B = 2.0f / Z;
    const float C = -B / (6.0f * Z);
    const float A = -B - C;

    uint32_t bits = 0;
    memcpy(&bits, &x, sizeof(bits));
    uint32_t const exp      = bits >> 23;
    uint32_t const signif   = bits & SIGNIF_MASK;
    uint32_t const high     = (signif > zSignif) ? 1u : 0u;
    int32_t const logInt    = (int32_t)(exp + high) - 127;
    uint32_t const normExp  = 0x7Fu ^ high;
    uint32_t const normBits = (normExp << 23) | signif;
    float normalized        = 0.0f;
    memcpy(&normalized, &normBits, sizeof(normalized));
    return (float)logInt + A + normalized * (B + C * normalized);
}

typedef struct {
    uint64_t n;
    uint64_t nBins;
    unsigned nBinsLog;
    size_t nApplied;
    size_t nextAvailBinIdx;
    int hasIncomplete;
    PbsHistBin incomplete;
    PbsHistBin* dst;
    size_t dstLen;
} PbsHistBuilder;

static inline size_t pbs_hb_bin_idx(const PbsHistBuilder* hb, size_t cCount)
{
    return (size_t)(((uint64_t)cCount << hb->nBinsLog) / hb->n);
}

static inline size_t pbs_hb_c_count(const PbsHistBuilder* hb, size_t binIdx)
{
    return (size_t)((((uint64_t)(binIdx + 1u) * hb->n) + hb->nBins - 1ULL)
                    >> hb->nBinsLog);
}

static void pbs_hb_apply_incomplete(
        PbsHistBuilder* hb,
        size_t count,
        uint64_t lower,
        uint64_t upper)
{
    if (count == 0) {
        return;
    }
    if (hb->hasIncomplete) {
        hb->incomplete.upper = upper;
        hb->incomplete.count += (uint32_t)count;
    } else {
        hb->hasIncomplete    = 1;
        hb->incomplete.lower = lower;
        hb->incomplete.upper = upper;
        hb->incomplete.count = (uint32_t)count;
    }
    hb->nApplied += count;
}

static int pbs_hb_complete_bin(PbsHistBuilder* hb, size_t binIdx)
{
    if (!hb->hasIncomplete) {
        return 0;
    }
    ZL_ASSERT_GE(binIdx, hb->nextAvailBinIdx);
    hb->nextAvailBinIdx   = binIdx + 1u;
    hb->dst[hb->dstLen++] = hb->incomplete;
    hb->hasIncomplete     = 0;
    return 1;
}

static void pbs_hb_apply_const_run(
        PbsHistBuilder* hb,
        uint64_t val,
        size_t runLen)
{
    size_t const start = hb->nApplied;
    size_t const mid   = start + runLen / 2u;
    size_t const end   = start + runLen;
    size_t binIdx      = pbs_hb_bin_idx(hb, mid);
    if (binIdx > hb->nextAvailBinIdx) {
        size_t const spare = binIdx - 1u;
        if (!pbs_hb_complete_bin(hb, spare)) {
            binIdx = spare;
        }
    }
    pbs_hb_apply_incomplete(hb, runLen, val, val);
    if (end >= pbs_hb_c_count(hb, binIdx)) {
        pbs_hb_complete_bin(hb, binIdx);
    }
}

#define PBS_DEFINE_HIST_FN(W, T)                                                   static size_t pbs_build_histogram_##W(                                                 const T* sorted, size_t n, PbsHistBin* outBins)                        {                                                                                  unsigned nBinsLog = 0;                                                         while ((size_t)(1u << (nBinsLog + 1u)) <= n                                           && nBinsLog < PBS_MAX_BINS_LOG) {                                           nBinsLog++;                                                                }                                                                              PbsHistBuilder hb = {                                                              .n               = (uint64_t)n,                                                .nBins           = (uint64_t)(1u << nBinsLog),                                 .nBinsLog        = nBinsLog,                                                   .nApplied        = 0,                                                          .nextAvailBinIdx = 0,                                                          .hasIncomplete   = 0,                                                          .dst             = outBins,                                                    .dstLen          = 0,                                                      };                                                                             size_t pos = 0;                                                                while (pos < n) {                                                                  size_t const rem          = n - pos;                                           size_t const targetBinIdx = pbs_hb_bin_idx(&hb, hb.nApplied);                  size_t const targetCCount = pbs_hb_c_count(&hb, targetBinIdx);                 size_t const targetI      = targetCCount - hb.nApplied;                        if (targetI >= rem) {                                                              pbs_hb_apply_incomplete(                                                               &hb, rem, (uint64_t)sorted[pos], (uint64_t)sorted[n - 1u]);                 if (targetI == rem) {                                                              pbs_hb_complete_bin(&hb, targetBinIdx);                                    }                                                                              break;                                                                     }                                                                              size_t l    = pos + targetI - 1u;                                              size_t r    = pos + targetI;                                                   T const tx  = sorted[l];                                                       while (l > pos && sorted[l - 1u] == tx) {                                          l--;                                                                       }                                                                              while (r < n && sorted[r] == tx) {                                                 r++;                                                                       }                                                                              if (l > pos) {                                                                     pbs_hb_apply_incomplete(                                                               &hb,                                                                           l - pos,                                                                       (uint64_t)sorted[pos],                                                         (uint64_t)sorted[l - 1u]);                                         }                                                                              pbs_hb_apply_const_run(&hb, (uint64_t)tx, r - l);                              pos = r;                                                                   }                                                                              if (hb.hasIncomplete) {                                                            pbs_hb_complete_bin(&hb, (size_t)hb.nBins - 1u);                           }                                                                              return hb.dstLen;                                                          }

PBS_DEFINE_HIST_FN(1, uint8_t)
PBS_DEFINE_HIST_FN(2, uint16_t)
PBS_DEFINE_HIST_FN(4, uint32_t)
PBS_DEFINE_HIST_FN(8, uint64_t)

static inline float pbs_bin_cost(
        float binMetaCost,
        uint64_t lower,
        uint64_t upper,
        uint32_t count,
        float totalCountLog2)
{
    float const c          = (float)count;
    float const ansCost    = totalCountLog2 - pbs_log2_approx(c);
    float const offsetCost = (float)pbs_offset_bits(upper - lower);
    return binMetaCost + (ansCost + offsetCost) * c;
}

static size_t pbs_optimize_bins(
        const PbsHistBin* hist,
        size_t nHist,
        unsigned eltBits,
        PbsHistBin* mergedOut)
{
    uint32_t cCounts[PBS_MAX_BINS + 1];
    float bestCosts[PBS_MAX_BINS + 1];
    size_t bestJs[PBS_MAX_BINS];

    cCounts[0]   = 0;
    bestCosts[0] = 0.0f;
    for (size_t i = 0; i < nHist; i++) {
        cCounts[i + 1] = cCounts[i] + hist[i].count;
    }
    uint32_t const totalCount  = cCounts[nHist];
    float const totalCountLog2 = pbs_log2_approx((float)totalCount);
    float const binMetaCost    = 24.0f + (float)eltBits;

    for (size_t i = 0; i < nHist; i++) {
        float bestCost       = FLT_MAX;
        size_t bestJ         = 0;
        uint64_t const upper = hist[i].upper;
        uint32_t const ccI   = cCounts[i + 1];
        for (size_t j = i + 1; j-- > 0;) {
            float const cost = bestCosts[j]
                    + pbs_bin_cost(
                            binMetaCost,
                            hist[j].lower,
                            upper,
                            ccI - cCounts[j],
                            totalCountLog2);
            if (cost < bestCost) {
                bestCost = cost;
                bestJ    = j;
            }
        }
        bestCosts[i + 1] = bestCost;
        bestJs[i]        = bestJ;
    }

    float const bestCost      = bestCosts[nHist];
    float const singleBinCost = pbs_bin_cost(
            binMetaCost,
            hist[0].lower,
            hist[nHist - 1].upper,
            totalCount,
            totalCountLog2);
    if (singleBinCost < bestCost + 0.1f * (float)totalCount) {
        mergedOut[0].lower = hist[0].lower;
        mergedOut[0].upper = hist[nHist - 1].upper;
        mergedOut[0].count = totalCount;
        return 1;
    }

    int allTrivial = 1;
    float trivCost = 0.0f;
    for (size_t i = 0; i < nHist; i++) {
        if (hist[i].lower != hist[i].upper) {
            allTrivial = 0;
            break;
        }
        trivCost += pbs_bin_cost(
                binMetaCost,
                hist[i].lower,
                hist[i].upper,
                hist[i].count,
                totalCountLog2);
    }
    if (allTrivial && trivCost < bestCost + 0.1f * (float)totalCount) {
        memcpy(mergedOut, hist, nHist * sizeof(PbsHistBin));
        return nHist;
    }

    size_t revStart[PBS_MAX_BINS];
    size_t revEnd[PBS_MAX_BINS];
    size_t nMerged = 0;
    size_t cur     = nHist - 1u;
    while (1) {
        size_t const j    = bestJs[cur];
        revStart[nMerged] = j;
        revEnd[nMerged]   = cur;
        nMerged++;
        if (j == 0) {
            break;
        }
        cur = j - 1u;
    }
    for (size_t k = 0; k < nMerged; k++) {
        size_t const rk    = nMerged - 1u - k;
        size_t const s     = revStart[rk];
        size_t const e     = revEnd[rk];
        mergedOut[k].lower = hist[s].lower;
        mergedOut[k].upper = hist[e].upper;
        mergedOut[k].count = cCounts[e + 1u] - cCounts[s];
    }
    return nMerged;
}

#define PBS_SHAR_STEP(BASE, V, LOWS, STEP)                                         do {                                                                               size_t const cand_ = (BASE) + (STEP);                                          (BASE)             = ((LOWS)[cand_] <= (V)) ? cand_ : (BASE);              } while (0)

#define PBS_SHAR_SEARCH_ONE(T, B_OUT, V, LOWS, REM, DEPTH)                         do {                                                                               T const v_    = (V);                                                           size_t b_     = ((LOWS)[REM] <= v_) ? (REM) : 0u;                              if ((DEPTH) >= 8) {                                                                PBS_SHAR_STEP(b_, v_, LOWS, 128u);                                         }                                                                              if ((DEPTH) >= 7) {                                                                PBS_SHAR_STEP(b_, v_, LOWS, 64u);                                          }                                                                              if ((DEPTH) >= 6) {                                                                PBS_SHAR_STEP(b_, v_, LOWS, 32u);                                          }                                                                              if ((DEPTH) >= 5) {                                                                PBS_SHAR_STEP(b_, v_, LOWS, 16u);                                          }                                                                              if ((DEPTH) >= 4) {                                                                PBS_SHAR_STEP(b_, v_, LOWS, 8u);                                           }                                                                              if ((DEPTH) >= 3) {                                                                PBS_SHAR_STEP(b_, v_, LOWS, 4u);                                           }                                                                              if ((DEPTH) >= 2) {                                                                PBS_SHAR_STEP(b_, v_, LOWS, 2u);                                           }                                                                              if ((DEPTH) >= 1) {                                                                PBS_SHAR_STEP(b_, v_, LOWS, 1u);                                           }                                                                              (B_OUT) = b_;                                                              } while (0)

#define PBS_SHAR_LOOP_DEPTH(T, LOWS, LATENTS, N, REM, DEPTH, OBITS, BDST, TBITS)     do {                                                                                 size_t i_          = 0;                                                          uint64_t tbits_    = 0;                                                          for (; i_ + 4u <= (N); i_ += 4u) {                                                   size_t b0_, b1_, b2_, b3_;                                                       PBS_SHAR_SEARCH_ONE(T, b0_, (LATENTS)[i_ + 0u], LOWS, REM, DEPTH);               PBS_SHAR_SEARCH_ONE(T, b1_, (LATENTS)[i_ + 1u], LOWS, REM, DEPTH);               PBS_SHAR_SEARCH_ONE(T, b2_, (LATENTS)[i_ + 2u], LOWS, REM, DEPTH);               PBS_SHAR_SEARCH_ONE(T, b3_, (LATENTS)[i_ + 3u], LOWS, REM, DEPTH);               (BDST)[i_ + 0u] = (uint8_t)b0_;                                                  (BDST)[i_ + 1u] = (uint8_t)b1_;                                                  (BDST)[i_ + 2u] = (uint8_t)b2_;                                                  (BDST)[i_ + 3u] = (uint8_t)b3_;                                                  tbits_ += (uint64_t)(OBITS)[b0_] + (uint64_t)(OBITS)[b1_]                                + (uint64_t)(OBITS)[b2_] + (uint64_t)(OBITS)[b3_];                   }                                                                                for (; i_ < (N); i_++) {                                                             size_t b0_;                                                                      PBS_SHAR_SEARCH_ONE(T, b0_, (LATENTS)[i_], LOWS, REM, DEPTH);                    (BDST)[i_] = (uint8_t)b0_;                                                       tbits_ += (uint64_t)(OBITS)[b0_];                                            }                                                                                *(TBITS) = tbits_;                                                           } while (0)

#define PBS_DEFINE_SHAR_FIND(W, T)                                                 static uint64_t pbs_assign_bins_##W(                                                   const T* lowers,                                                               size_t numBins,                                                                const T* latents,                                                              size_t n,                                                                      const uint8_t* offsetBits,                                                     uint8_t* binDst)                                                       {                                                                                  ZL_ASSERT_GE(numBins, 2u);                                                     unsigned const depth = 31u - (unsigned)ZL_clz32((uint32_t)numBins);            size_t const rem     = numBins - ((size_t)1u << depth);                        uint64_t totalBits   = 0;                                                      switch (depth) {                                                                   case 1:                                                                            PBS_SHAR_LOOP_DEPTH(                                                                   T, lowers, latents, n, rem, 1, offsetBits, binDst, &totalBits);                 break;                                                                     case 2:                                                                            PBS_SHAR_LOOP_DEPTH(                                                                   T, lowers, latents, n, rem, 2, offsetBits, binDst, &totalBits);                 break;                                                                     case 3:                                                                            PBS_SHAR_LOOP_DEPTH(                                                                   T, lowers, latents, n, rem, 3, offsetBits, binDst, &totalBits);                 break;                                                                     case 4:                                                                            PBS_SHAR_LOOP_DEPTH(                                                                   T, lowers, latents, n, rem, 4, offsetBits, binDst, &totalBits);                 break;                                                                     case 5:                                                                            PBS_SHAR_LOOP_DEPTH(                                                                   T, lowers, latents, n, rem, 5, offsetBits, binDst, &totalBits);                 break;                                                                     case 6:                                                                            PBS_SHAR_LOOP_DEPTH(                                                                   T, lowers, latents, n, rem, 6, offsetBits, binDst, &totalBits);                 break;                                                                     case 7:                                                                            PBS_SHAR_LOOP_DEPTH(                                                                   T, lowers, latents, n, rem, 7, offsetBits, binDst, &totalBits);                 break;                                                                     default:                                                                           PBS_SHAR_LOOP_DEPTH(                                                                   T, lowers, latents, n, 0u, 8, offsetBits, binDst, &totalBits);                 break;                                                                 }                                                                              return totalBits;                                                          }

PBS_DEFINE_SHAR_FIND(2, uint16_t)
PBS_DEFINE_SHAR_FIND(4, uint32_t)
PBS_DEFINE_SHAR_FIND(8, uint64_t)

#define PBS_DEFINE_RADIX_SORT(W, T, BYTES)                                         static const T* pbs_radix_sort_##W(                                                    const T* latents,                                                              T* bufA,                                                                       T* bufB,                                                                       size_t n,                                                                      uint32_t counts[BYTES][256])                                           {                                                                                  const T* curSrc = latents;                                                     T* curDst       = bufA;                                                        T* altDst       = bufB;                                                        for (unsigned byteIdx = 0; byteIdx < (BYTES); byteIdx++) {                         unsigned const shift    = byteIdx * 8u;                                        unsigned const firstB   = (unsigned)((latents[0] >> shift) & 0xFFu);             if (counts[byteIdx][firstB] == n) {                                                continue;                                                                  }                                                                              uint32_t offsets[256];                                                         uint32_t acc = 0;                                                              for (size_t b = 0; b < 256; b++) {                                                 offsets[b] = acc;                                                              acc += counts[byteIdx][b];                                                 }                                                                              for (size_t i = 0; i < n; i++) {                                                   T const v              = curSrc[i];                                            unsigned const bucket  = (unsigned)((v >> shift) & 0xFFu);                     curDst[offsets[bucket]++] = v;                                             }                                                                              curSrc    = curDst;                                                            T* tmpDst = curDst;                                                            curDst    = altDst;                                                            altDst    = tmpDst;                                                        }                                                                              return curSrc;                                                             }

PBS_DEFINE_RADIX_SORT(2, uint16_t, 2)
PBS_DEFINE_RADIX_SORT(4, uint32_t, 4)
PBS_DEFINE_RADIX_SORT(8, uint64_t, 8)

#define PBS_DEFINE_BITPACK(W, T)                                                   static void pbs_pack_offsets_##W(                                                      uint8_t* dst,                                                                  size_t packedBytes,                                                            const T* latents,                                                              size_t n,                                                                      const T* lowers,                                                               const uint8_t* offsetBits,                                                     const uint8_t* binIds,                                                         int singleBin,                                                                 unsigned maxOb)                                                        {                                                                                  uint8_t* const dstEnd = dst + packedBytes;                                     uint64_t bitbuf       = 0;                                                     unsigned bitcnt       = 0;                                                     size_t i              = 0;                                                     if (singleBin) {                                                                   T const lo        = lowers[0];                                                 unsigned const ob = offsetBits[0];                                             if (ob <= 56u && packedBytes >= 8u) {                                              uint8_t* const dstSafe = dstEnd - 8u;                                          while (i < n && dst <= dstSafe) {                                                  uint64_t const off = (uint64_t)(latents[i] - lo);                              bitbuf |= off << bitcnt;                                                       bitcnt += ob;                                                                  ZL_writeLE64(dst, bitbuf);                                                     unsigned const adv = bitcnt >> 3;                                              dst += adv;                                                                    bitbuf >>= (adv << 3);                                                         bitcnt &= 7u;                                                                  i++;                                                                       }                                                                          }                                                                              for (; i < n; i++) {                                                               uint64_t const off = (uint64_t)(latents[i] - lo);                              bitbuf |= off << bitcnt;                                                       if (bitcnt + ob >= 64u) {                                                          if (dst + 8 <= dstEnd) {                                                           ZL_writeLE64(dst, bitbuf);                                                     dst += 8;                                                                  } else {                                                                           for (int k = 0; k < 8 && dst < dstEnd; k++) {                                      *dst++ = (uint8_t)(bitbuf >> (k * 8));                                     }                                                                          }                                                                              bitbuf = (bitcnt == 0u) ? 0ULL : (off >> (64u - bitcnt));                      bitcnt = (bitcnt + ob) - 64u;                                              } else {                                                                           bitcnt += ob;                                                              }                                                                          }                                                                          } else {                                                                           if (maxOb <= 56u && packedBytes >= 8u) {                                           uint8_t* const dstSafe = dstEnd - 8u;                                          while (i < n && dst <= dstSafe) {                                                  unsigned const b   = binIds[i];                                                unsigned const ob  = offsetBits[b];                                            uint64_t const off = (uint64_t)(latents[i] - lowers[b]);                       bitbuf |= off << bitcnt;                                                       bitcnt += ob;                                                                  ZL_writeLE64(dst, bitbuf);                                                     unsigned const adv = bitcnt >> 3;                                              dst += adv;                                                                    bitbuf >>= (adv << 3);                                                         bitcnt &= 7u;                                                                  i++;                                                                       }                                                                          }                                                                              for (; i < n; i++) {                                                               unsigned const b   = binIds[i];                                                unsigned const ob  = offsetBits[b];                                            uint64_t const off = (uint64_t)(latents[i] - lowers[b]);                       bitbuf |= off << bitcnt;                                                       if (bitcnt + ob >= 64u) {                                                          if (dst + 8 <= dstEnd) {                                                           ZL_writeLE64(dst, bitbuf);                                                     dst += 8;                                                                  } else {                                                                           for (int k = 0; k < 8 && dst < dstEnd; k++) {                                      *dst++ = (uint8_t)(bitbuf >> (k * 8));                                     }                                                                          }                                                                              bitbuf = (bitcnt == 0u) ? 0ULL : (off >> (64u - bitcnt));                      bitcnt = (bitcnt + ob) - 64u;                                              } else {                                                                           bitcnt += ob;                                                              }                                                                          }                                                                          }                                                                              while (bitcnt > 0u && dst < dstEnd) {                                              *dst++ = (uint8_t)(bitbuf & 0xFFu);                                            bitbuf >>= 8;                                                                  bitcnt = (bitcnt >= 8u) ? (bitcnt - 8u) : 0u;                              }                                                                          }

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

    PbsHistBin hist[PBS_MAX_BINS];
    size_t nHist = 0;

    switch (eltWidth) {
        case 1: {
            const uint8_t* const p = (const uint8_t*)srcPtr;
            uint32_t counts[256];
            memset(counts, 0, sizeof(counts));
            if (latentMode == ZL_PcoBinSplit_signed) {
                uint8_t* const l = (uint8_t*)ZL_Encoder_getScratchSpace(eictx, n);
                ZL_ERR_IF_NULL(l, allocation);
                for (size_t i = 0; i < n; i++) {
                    uint8_t const v = (uint8_t)(p[i] ^ 0x80u);
                    l[i]            = v;
                    counts[v]++;
                }
                latents1 = l;
            } else {
                for (size_t i = 0; i < n; i++) {
                    counts[p[i]]++;
                }
                latents1 = p;
            }
            if (counts[latents1[0]] == n) {
                hist[0].lower = latents1[0];
                hist[0].upper = latents1[0];
                hist[0].count = (uint32_t)n;
                nHist         = 1;
            } else {
                uint8_t* const sorted =
                        (uint8_t*)ZL_Encoder_getScratchSpace(eictx, n);
                ZL_ERR_IF_NULL(sorted, allocation);
                size_t pos = 0;
                for (unsigned v = 0; v < 256u; v++) {
                    uint32_t const c = counts[v];
                    if (c > 0u) {
                        memset(sorted + pos, (int)v, c);
                        pos += c;
                    }
                }
                nHist = pbs_build_histogram_1(sorted, n, hist);
            }
            break;
        }
        case 2: {
            const uint16_t* const p = (const uint16_t*)srcPtr;
            if (latentMode == ZL_PcoBinSplit_unsigned) {
                latents2 = p;
            } else {
                uint16_t* const l = (uint16_t*)ZL_Encoder_getScratchSpace(
                        eictx, n * sizeof(uint16_t));
                ZL_ERR_IF_NULL(l, allocation);
                if (latentMode == ZL_PcoBinSplit_signed) {
                    for (size_t i = 0; i < n; i++) {
                        l[i] = (uint16_t)(p[i] ^ 0x8000u);
                    }
                } else {
                    for (size_t i = 0; i < n; i++) {
                        uint16_t const x = p[i];
                        uint16_t const m =
                                (uint16_t)((0u - (uint32_t)(x >> 15u)) | 0x8000u);
                        l[i] = (uint16_t)(x ^ m);
                    }
                }
                latents2 = l;
            }
            if (n >= 2048u) {
                uint32_t* const counts = (uint32_t*)ZL_Encoder_getScratchSpace(
                        eictx, 65536u * sizeof(uint32_t));
                ZL_ERR_IF_NULL(counts, allocation);
                memset(counts, 0, 65536u * sizeof(uint32_t));
                for (size_t i = 0; i < n; i++) {
                    counts[latents2[i]]++;
                }
                if (counts[latents2[0]] == n) {
                    hist[0].lower = latents2[0];
                    hist[0].upper = latents2[0];
                    hist[0].count = (uint32_t)n;
                    nHist         = 1;
                } else {
                    uint16_t* const sorted = (uint16_t*)ZL_Encoder_getScratchSpace(
                            eictx, n * sizeof(uint16_t));
                    ZL_ERR_IF_NULL(sorted, allocation);
                    size_t pos = 0;
                    for (uint32_t v = 0; v < 65536u; v++) {
                        uint32_t const c = counts[v];
                        for (uint32_t k = 0; k < c; k++) {
                            sorted[pos++] = (uint16_t)v;
                        }
                    }
                    nHist = pbs_build_histogram_2(sorted, n, hist);
                }
            } else {
                uint32_t counts[2][256];
                memset(counts, 0, sizeof(counts));
                for (size_t i = 0; i < n; i++) {
                    uint16_t const v = latents2[i];
                    counts[0][v & 0xFFu]++;
                    counts[1][(v >> 8) & 0xFFu]++;
                }
                uint16_t const v0 = latents2[0];
                if (counts[0][v0 & 0xFFu] == n && counts[1][(v0 >> 8) & 0xFFu] == n) {
                    hist[0].lower = v0;
                    hist[0].upper = v0;
                    hist[0].count = (uint32_t)n;
                    nHist         = 1;
                } else {
                    uint16_t* const bufA = (uint16_t*)ZL_Encoder_getScratchSpace(
                            eictx, n * sizeof(uint16_t));
                    uint16_t* const bufB = (uint16_t*)ZL_Encoder_getScratchSpace(
                            eictx, n * sizeof(uint16_t));
                    ZL_ERR_IF(bufA == NULL || bufB == NULL, allocation);
                    const uint16_t* const sorted =
                            pbs_radix_sort_2(latents2, bufA, bufB, n, counts);
                    nHist = pbs_build_histogram_2(sorted, n, hist);
                }
            }
            break;
        }
        case 4: {
            const uint32_t* const p = (const uint32_t*)srcPtr;
            uint32_t counts[4][256];
            memset(counts, 0, sizeof(counts));
            if (latentMode == ZL_PcoBinSplit_unsigned) {
                latents4 = p;
                for (size_t i = 0; i < n; i++) {
                    uint32_t const v = p[i];
                    counts[0][v & 0xFFu]++;
                    counts[1][(v >> 8) & 0xFFu]++;
                    counts[2][(v >> 16) & 0xFFu]++;
                    counts[3][(v >> 24) & 0xFFu]++;
                }
            } else {
                uint32_t* const l = (uint32_t*)ZL_Encoder_getScratchSpace(
                        eictx, n * sizeof(uint32_t));
                ZL_ERR_IF_NULL(l, allocation);
                if (latentMode == ZL_PcoBinSplit_signed) {
                    for (size_t i = 0; i < n; i++) {
                        uint32_t const v = p[i] ^ 0x80000000u;
                        l[i]             = v;
                        counts[0][v & 0xFFu]++;
                        counts[1][(v >> 8) & 0xFFu]++;
                        counts[2][(v >> 16) & 0xFFu]++;
                        counts[3][(v >> 24) & 0xFFu]++;
                    }
                } else {
                    for (size_t i = 0; i < n; i++) {
                        uint32_t const x = p[i];
                        uint32_t const m = (0u - (x >> 31u)) | 0x80000000u;
                        uint32_t const v = x ^ m;
                        l[i]             = v;
                        counts[0][v & 0xFFu]++;
                        counts[1][(v >> 8) & 0xFFu]++;
                        counts[2][(v >> 16) & 0xFFu]++;
                        counts[3][(v >> 24) & 0xFFu]++;
                    }
                }
                latents4 = l;
            }
            uint32_t const v0 = latents4[0];
            if (counts[0][v0 & 0xFFu] == n && counts[1][(v0 >> 8) & 0xFFu] == n
                && counts[2][(v0 >> 16) & 0xFFu] == n
                && counts[3][(v0 >> 24) & 0xFFu] == n) {
                hist[0].lower = v0;
                hist[0].upper = v0;
                hist[0].count = (uint32_t)n;
                nHist         = 1;
            } else {
                uint32_t* const bufA = (uint32_t*)ZL_Encoder_getScratchSpace(
                        eictx, n * sizeof(uint32_t));
                uint32_t* const bufB = (uint32_t*)ZL_Encoder_getScratchSpace(
                        eictx, n * sizeof(uint32_t));
                ZL_ERR_IF(bufA == NULL || bufB == NULL, allocation);
                const uint32_t* const sorted =
                        pbs_radix_sort_4(latents4, bufA, bufB, n, counts);
                nHist = pbs_build_histogram_4(sorted, n, hist);
            }
            break;
        }
        case 8: {
            const uint64_t* const p = (const uint64_t*)srcPtr;
            uint32_t counts[8][256];
            memset(counts, 0, sizeof(counts));
            if (latentMode == ZL_PcoBinSplit_unsigned) {
                latents8 = p;
                for (size_t i = 0; i < n; i++) {
                    uint64_t const v = p[i];
                    counts[0][v & 0xFFu]++;
                    counts[1][(v >> 8) & 0xFFu]++;
                    counts[2][(v >> 16) & 0xFFu]++;
                    counts[3][(v >> 24) & 0xFFu]++;
                    counts[4][(v >> 32) & 0xFFu]++;
                    counts[5][(v >> 40) & 0xFFu]++;
                    counts[6][(v >> 48) & 0xFFu]++;
                    counts[7][(v >> 56) & 0xFFu]++;
                }
            } else {
                uint64_t* const l = (uint64_t*)ZL_Encoder_getScratchSpace(
                        eictx, n * sizeof(uint64_t));
                ZL_ERR_IF_NULL(l, allocation);
                if (latentMode == ZL_PcoBinSplit_signed) {
                    for (size_t i = 0; i < n; i++) {
                        uint64_t const v = p[i] ^ 0x8000000000000000ULL;
                        l[i]             = v;
                        counts[0][v & 0xFFu]++;
                        counts[1][(v >> 8) & 0xFFu]++;
                        counts[2][(v >> 16) & 0xFFu]++;
                        counts[3][(v >> 24) & 0xFFu]++;
                        counts[4][(v >> 32) & 0xFFu]++;
                        counts[5][(v >> 40) & 0xFFu]++;
                        counts[6][(v >> 48) & 0xFFu]++;
                        counts[7][(v >> 56) & 0xFFu]++;
                    }
                } else {
                    for (size_t i = 0; i < n; i++) {
                        uint64_t const x = p[i];
                        uint64_t const m =
                                (0ULL - (x >> 63u)) | 0x8000000000000000ULL;
                        uint64_t const v = x ^ m;
                        l[i]             = v;
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
                latents8 = l;
            }
            uint64_t const v0 = latents8[0];
            int allConst      = 1;
            for (unsigned b = 0; b < 8u; b++) {
                if (counts[b][(v0 >> (b * 8u)) & 0xFFu] != n) {
                    allConst = 0;
                    break;
                }
            }
            if (allConst) {
                hist[0].lower = v0;
                hist[0].upper = v0;
                hist[0].count = (uint32_t)n;
                nHist         = 1;
            } else {
                uint64_t* const bufA = (uint64_t*)ZL_Encoder_getScratchSpace(
                        eictx, n * sizeof(uint64_t));
                uint64_t* const bufB = (uint64_t*)ZL_Encoder_getScratchSpace(
                        eictx, n * sizeof(uint64_t));
                ZL_ERR_IF(bufA == NULL || bufB == NULL, allocation);
                const uint64_t* const sorted =
                        pbs_radix_sort_8(latents8, bufA, bufB, n, counts);
                nHist = pbs_build_histogram_8(sorted, n, hist);
            }
            break;
        }
    }

    PbsHistBin bins[PBS_MAX_BINS];
    size_t const numBins = pbs_optimize_bins(hist, nHist, eltBits, bins);

    uint64_t lowers[PBS_MAX_BINS];
    uint8_t lowers1[PBS_MAX_BINS];
    uint16_t lowers2[PBS_MAX_BINS];
    uint32_t lowers4[PBS_MAX_BINS];
    uint8_t offsetBits[PBS_MAX_BINS];
    int allZeroOffsets = 1;
    unsigned maxOb     = 0;
    for (size_t b = 0; b < numBins; b++) {
        uint64_t const lo = bins[b].lower;
        lowers[b]         = lo;
        lowers1[b]        = (uint8_t)lo;
        lowers2[b]        = (uint16_t)lo;
        lowers4[b]        = (uint32_t)lo;
        unsigned const ob = pbs_offset_bits(bins[b].upper - lo);
        offsetBits[b]     = (uint8_t)ob;
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
                for (size_t i = 0; i < n; i++) {
                    uint8_t const b = valToBin[latents1[i]];
                    binDst[i]       = b;
                    totalBits += offsetBits[b];
                }
                break;
            }
            case 2: {
                if (n >= 4096u) {
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
