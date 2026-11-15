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

static inline uint64_t pbs_to_latent(
        uint64_t x,
        unsigned eltBits,
        int latentMode)
{
    uint64_t const mask    = (eltBits == 64) ? UINT64_MAX : ((1ULL << eltBits) - 1ULL);
    uint64_t const signBit = 1ULL << (eltBits - 1u);
    if (latentMode == ZL_PcoBinSplit_signed) {
        return x ^ signBit;
    }
    if (latentMode == ZL_PcoBinSplit_float && eltBits >= 16) {
        return ((x & signBit) != 0) ? ((~x) & mask) : (x ^ signBit);
    }
    return x;
}

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

static uint64_t* pbs_radix_sort(
        uint64_t* bufA,
        uint64_t* bufB,
        size_t n,
        unsigned eltBytes)
{
    uint64_t* src = bufA;
    uint64_t* dst = bufB;
    for (unsigned byteIdx = 0; byteIdx < eltBytes; byteIdx++) {
        uint32_t counts[256];
        memset(counts, 0, sizeof(counts));
        unsigned const shift = byteIdx * 8u;
        for (size_t i = 0; i < n; i++) {
            counts[(src[i] >> shift) & 0xFFu]++;
        }
        uint32_t const firstCount = counts[(src[0] >> shift) & 0xFFu];
        if (firstCount == n) {
            continue;
        }
        uint32_t offsets[256];
        uint32_t acc = 0;
        for (size_t b = 0; b < 256; b++) {
            offsets[b] = acc;
            acc += counts[b];
        }
        for (size_t i = 0; i < n; i++) {
            uint64_t const v       = src[i];
            unsigned const bucket  = (unsigned)((v >> shift) & 0xFFu);
            dst[offsets[bucket]++] = v;
        }
        uint64_t* const tmp = src;
        src                 = dst;
        dst                 = tmp;
    }
    return src;
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
    hb->nextAvailBinIdx = binIdx + 1u;
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

static size_t pbs_build_histogram(
        const uint64_t* sorted,
        size_t n,
        PbsHistBin* outBins)
{
    unsigned nBinsLog = 0;
    while ((size_t)(1u << (nBinsLog + 1u)) <= n && nBinsLog < PBS_MAX_BINS_LOG) {
        nBinsLog++;
    }
    PbsHistBuilder hb = {
        .n               = (uint64_t)n,
        .nBins           = (uint64_t)(1u << nBinsLog),
        .nBinsLog        = nBinsLog,
        .nApplied        = 0,
        .nextAvailBinIdx = 0,
        .hasIncomplete   = 0,
        .dst             = outBins,
        .dstLen          = 0,
    };

    size_t pos = 0;
    while (pos < n) {
        size_t const rem          = n - pos;
        size_t const targetBinIdx = pbs_hb_bin_idx(&hb, hb.nApplied);
        size_t const targetCCount = pbs_hb_c_count(&hb, targetBinIdx);
        size_t const targetI      = targetCCount - hb.nApplied;

        if (targetI >= rem) {
            pbs_hb_apply_incomplete(&hb, rem, sorted[pos], sorted[n - 1]);
            if (targetI == rem) {
                pbs_hb_complete_bin(&hb, targetBinIdx);
            }
            break;
        }

        size_t l           = pos + targetI - 1u;
        size_t r           = pos + targetI;
        uint64_t const tx  = sorted[l];
        while (l > pos && sorted[l - 1u] == tx) {
            l--;
        }
        while (r < n && sorted[r] == tx) {
            r++;
        }

        if (l > pos) {
            pbs_hb_apply_incomplete(&hb, l - pos, sorted[pos], sorted[l - 1u]);
        }
        pbs_hb_apply_const_run(&hb, tx, r - l);
        pos = r;
    }
    if (hb.hasIncomplete) {
        pbs_hb_complete_bin(&hb, (size_t)hb.nBins - 1u);
    }
    return hb.dstLen;
}

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
    // Approximate per-bin metadata + entropy table overhead in bits.
    float const binMetaCost = 24.0f + (float)eltBits;

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

    float const bestCost = bestCosts[nHist];
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
        size_t const j     = bestJs[cur];
        revStart[nMerged]  = j;
        revEnd[nMerged]    = cur;
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

static inline size_t pbs_find_bin(
        const uint64_t* lowers,
        size_t numBins,
        uint64_t v)
{
    size_t lo = 0;
    size_t hi = numBins;
    while (lo + 1u < hi) {
        size_t const mid = (lo + hi) >> 1;
        if (lowers[mid] <= v) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo;
}

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

    uint64_t* const latents =
            (uint64_t*)ZL_Encoder_getScratchSpace(eictx, n * sizeof(uint64_t));
    uint64_t* const sortBufA =
            (uint64_t*)ZL_Encoder_getScratchSpace(eictx, n * sizeof(uint64_t));
    uint64_t* const sortBufB =
            (uint64_t*)ZL_Encoder_getScratchSpace(eictx, n * sizeof(uint64_t));
    ZL_ERR_IF(
            latents == NULL || sortBufA == NULL || sortBufB == NULL,
            allocation);

    const void* const srcPtr = ZL_Input_ptr(in);
    switch (eltWidth) {
        case 1: {
            const uint8_t* const p = (const uint8_t*)srcPtr;
            for (size_t i = 0; i < n; i++) {
                latents[i] = pbs_to_latent(p[i], 8u, latentMode);
            }
            break;
        }
        case 2: {
            const uint16_t* const p = (const uint16_t*)srcPtr;
            for (size_t i = 0; i < n; i++) {
                latents[i] = pbs_to_latent(p[i], 16u, latentMode);
            }
            break;
        }
        case 4: {
            const uint32_t* const p = (const uint32_t*)srcPtr;
            for (size_t i = 0; i < n; i++) {
                latents[i] = pbs_to_latent(p[i], 32u, latentMode);
            }
            break;
        }
        case 8: {
            const uint64_t* const p = (const uint64_t*)srcPtr;
            for (size_t i = 0; i < n; i++) {
                latents[i] = pbs_to_latent(p[i], 64u, latentMode);
            }
            break;
        }
    }
    memcpy(sortBufA, latents, n * sizeof(uint64_t));
    const uint64_t* const sorted =
            pbs_radix_sort(sortBufA, sortBufB, n, (unsigned)eltWidth);

    PbsHistBin hist[PBS_MAX_BINS];
    size_t const nHist = pbs_build_histogram(sorted, n, hist);

    PbsHistBin bins[PBS_MAX_BINS];
    size_t const numBins = pbs_optimize_bins(hist, nHist, eltBits, bins);

    uint64_t lowers[PBS_MAX_BINS];
    uint8_t offsetBits[PBS_MAX_BINS];
    int allZeroOffsets = 1;
    for (size_t b = 0; b < numBins; b++) {
        lowers[b]     = bins[b].lower;
        unsigned ob   = pbs_offset_bits(bins[b].upper - bins[b].lower);
        offsetBits[b] = (uint8_t)ob;
        if (ob != 0u) {
            allZeroOffsets = 0;
        }
    }

    int const singleBin = (numBins == 1u);
    ZL_Output* const binOut =
            ZL_Encoder_createTypedStream(eictx, 0, singleBin ? 0 : n, 1);
    ZL_ERR_IF_NULL(binOut, allocation);

    uint64_t totalBits = 0;
    if (singleBin) {
        totalBits = (uint64_t)n * (uint64_t)offsetBits[0];
    } else {
        uint8_t* const binDst = (uint8_t*)ZL_Output_ptr(binOut);
        for (size_t i = 0; i < n; i++) {
            size_t const b = pbs_find_bin(lowers, numBins, latents[i]);
            binDst[i]      = (uint8_t)b;
            totalBits += offsetBits[b];
        }
    }

    size_t const packedBytes = (size_t)((totalBits + 7ULL) >> 3);
    ZL_Output* const offOut =
            ZL_Encoder_createTypedStream(eictx, 1, packedBytes, 1);
    ZL_ERR_IF_NULL(offOut, allocation);

    if (packedBytes > 0) {
        uint8_t* dst             = (uint8_t*)ZL_Output_ptr(offOut);
        uint8_t* const dstEnd    = dst + packedBytes;
        uint64_t bitbuf          = 0;
        unsigned bitcnt          = 0;
        if (singleBin) {
            uint64_t const lo = lowers[0];
            unsigned const ob = offsetBits[0];
            for (size_t i = 0; i < n; i++) {
                uint64_t const off = latents[i] - lo;
                bitbuf |= off << bitcnt;
                if (bitcnt + ob >= 64u) {
                    if (dst + 8 <= dstEnd) {
                        ZL_writeLE64(dst, bitbuf);
                        dst += 8;
                    } else {
                        for (int k = 0; k < 8 && dst < dstEnd; k++) {
                            *dst++ = (uint8_t)(bitbuf >> (k * 8));
                        }
                    }
                    bitbuf = (bitcnt == 0u) ? 0ULL : (off >> (64u - bitcnt));
                    bitcnt = (bitcnt + ob) - 64u;
                } else {
                    bitcnt += ob;
                }
            }
        } else {
            const uint8_t* const binIds = (const uint8_t*)ZL_Output_ptr(binOut);
            for (size_t i = 0; i < n; i++) {
                unsigned const b   = binIds[i];
                unsigned const ob  = offsetBits[b];
                uint64_t const off = latents[i] - lowers[b];
                bitbuf |= off << bitcnt;
                if (bitcnt + ob >= 64u) {
                    if (dst + 8 <= dstEnd) {
                        ZL_writeLE64(dst, bitbuf);
                        dst += 8;
                    } else {
                        for (int k = 0; k < 8 && dst < dstEnd; k++) {
                            *dst++ = (uint8_t)(bitbuf >> (k * 8));
                        }
                    }
                    bitbuf = (bitcnt == 0u) ? 0ULL : (off >> (64u - bitcnt));
                    bitcnt = (bitcnt + ob) - 64u;
                } else {
                    bitcnt += ob;
                }
            }
        }
        while (bitcnt > 0u && dst < dstEnd) {
            *dst++ = (uint8_t)(bitbuf & 0xFFu);
            bitbuf >>= 8;
            bitcnt = (bitcnt >= 8u) ? (bitcnt - 8u) : 0u;
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
