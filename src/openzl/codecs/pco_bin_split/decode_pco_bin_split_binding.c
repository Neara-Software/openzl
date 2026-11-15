// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/pco_bin_split/decode_pco_bin_split_binding.h"

#include <stdint.h>
#include <string.h>

#include "openzl/codecs/zl_pco_bin_split.h"
#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
#include "openzl/shared/mem.h"
#include "openzl/shared/varint.h"

enum {
    PBS_MAX_BINS  = 256,
    PBS_BATCH_N   = 256,
};

static inline uint64_t pbs_from_latent(
        uint64_t l,
        unsigned eltBits,
        int latentMode)
{
    uint64_t const mask    = (eltBits == 64) ? UINT64_MAX : ((1ULL << eltBits) - 1ULL);
    uint64_t const signBit = 1ULL << (eltBits - 1u);
    if (latentMode == ZL_PcoBinSplit_signed) {
        return l ^ signBit;
    }
    if (latentMode == ZL_PcoBinSplit_float && eltBits >= 16) {
        return ((l & signBit) != 0) ? (l ^ signBit) : ((~l) & mask);
    }
    return l;
}

static inline void pbs_store_elt(
        void* dst,
        size_t idx,
        size_t eltWidth,
        uint64_t val)
{
    switch (eltWidth) {
        case 1:
            ((uint8_t*)dst)[idx] = (uint8_t)val;
            break;
        case 2:
            ((uint16_t*)dst)[idx] = (uint16_t)val;
            break;
        case 4:
            ((uint32_t*)dst)[idx] = (uint32_t)val;
            break;
        case 8:
            ((uint64_t*)dst)[idx] = val;
            break;
    }
}

static inline uint64_t pbs_read_bits(
        const uint8_t* src,
        size_t packedBytes,
        uint64_t bitPos,
        unsigned ob)
{
    if (ob == 0u) {
        return 0ULL;
    }
    size_t const bytePos = (size_t)(bitPos >> 3);
    unsigned const shift = (unsigned)(bitPos & 7u);
    uint64_t const mask  = (ob == 64u) ? UINT64_MAX : ((1ULL << ob) - 1ULL);

    uint64_t word = 0;
    if (ZL_LIKELY(bytePos + 8u <= packedBytes)) {
        word = ZL_readLE64(src + bytePos) >> shift;
        if (ZL_UNLIKELY(shift + ob > 64u)) {
            word |= ((uint64_t)src[bytePos + 8u]) << (64u - shift);
        }
    } else {
        uint8_t tmp[9];
        memset(tmp, 0, sizeof(tmp));
        size_t const avail = packedBytes - bytePos;
        memcpy(tmp, src + bytePos, avail < 9u ? avail : 9u);
        word = ZL_readLE64(tmp) >> shift;
        if (shift + ob > 64u) {
            word |= ((uint64_t)tmp[8]) << (64u - shift);
        }
    }
    return word & mask;
}

ZL_Report DI_pco_bin_split(ZL_Decoder* dictx, const ZL_Input* ins[])
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dictx);
    const ZL_Input* const binIn = ins[0];
    const ZL_Input* const offIn = ins[1];
    ZL_ASSERT_EQ(ZL_Input_type(binIn), ZL_Type_numeric);
    ZL_ASSERT_EQ(ZL_Input_type(offIn), ZL_Type_serial);
    ZL_ERR_IF_NE(ZL_Input_eltWidth(binIn), 1, corruption);

    ZL_RBuffer const hdr = ZL_Decoder_getCodecHeader(dictx);
    ZL_ERR_IF_LT(hdr.size, 2, corruption);
    const uint8_t* ptr       = (const uint8_t*)hdr.start;
    const uint8_t* const end = ptr + hdr.size;

    uint8_t const flags       = *ptr++;
    unsigned const wLog2      = flags & 3u;
    int const latentMode      = (int)((flags >> 2) & 3u);
    int const singleBin       = (int)((flags >> 4) & 1u);
    int const allZeroOffsets  = (int)((flags >> 5) & 1u);
    ZL_ERR_IF_GT(latentMode, 2, corruption);

    size_t const eltWidth = (size_t)1u << wLog2;
    unsigned const eltBits = (unsigned)(eltWidth * 8u);

    ZL_TRY_LET(uint64_t, nU64, ZL_varintDecode(&ptr, end));
    size_t const n = (size_t)nU64;
    ZL_ERR_IF_NE((uint64_t)n, nU64, corruption);

    ZL_Output* const out = ZL_Decoder_create1OutStream(dictx, n, eltWidth);
    ZL_ERR_IF_NULL(out, allocation);

    if (n == 0) {
        ZL_ERR_IF_NE(ZL_Input_numElts(binIn), 0, corruption);
        ZL_ERR_IF_NE(ZL_Input_numElts(offIn), 0, corruption);
        ZL_ERR_IF_ERR(ZL_Output_commit(out, 0));
        return ZL_returnValue(1);
    }

    size_t numBins = 1u;
    if (!singleBin) {
        ZL_ERR_IF_GE(ptr, end, corruption);
        numBins = (size_t)(*ptr++) + 1u;
    }

    uint64_t lowers[PBS_MAX_BINS];
    uint8_t offsetBits[PBS_MAX_BINS];
    uint64_t prevLower = 0;
    for (size_t b = 0; b < numBins; b++) {
        ZL_TRY_LET(uint64_t, deltaLo, ZL_varintDecode(&ptr, end));
        lowers[b] = prevLower + deltaLo;
        prevLower = lowers[b];
        if (allZeroOffsets) {
            offsetBits[b] = 0;
        } else {
            ZL_ERR_IF_GE(ptr, end, corruption);
            uint8_t const ob = *ptr++;
            ZL_ERR_IF_GT(ob, eltBits, corruption);
            offsetBits[b] = ob;
        }
    }
    ZL_ERR_IF_NE(ptr, end, corruption);

    ZL_ERR_IF_NE(ZL_Input_numElts(binIn), singleBin ? 0 : n, corruption);
    const uint8_t* const offPtr = (const uint8_t*)ZL_Input_ptr(offIn);
    size_t const packedBytes    = ZL_Input_numElts(offIn);
    void* const dst             = ZL_Output_ptr(out);

    if (singleBin) {
        uint64_t const lo     = lowers[0];
        unsigned const ob     = offsetBits[0];
        uint64_t const needBits = (uint64_t)n * (uint64_t)ob;
        ZL_ERR_IF_GT((needBits + 7ULL) >> 3, (uint64_t)packedBytes, corruption);
        if (ob == 0u) {
            uint64_t const val = pbs_from_latent(lo, eltBits, latentMode);
            for (size_t i = 0; i < n; i++) {
                pbs_store_elt(dst, i, eltWidth, val);
            }
        } else {
            uint64_t bitPos = 0;
            for (size_t i = 0; i < n; i++) {
                uint64_t const off = pbs_read_bits(offPtr, packedBytes, bitPos, ob);
                bitPos += ob;
                pbs_store_elt(
                        dst,
                        i,
                        eltWidth,
                        pbs_from_latent(lo + off, eltBits, latentMode));
            }
        }
        ZL_ERR_IF_ERR(ZL_Output_commit(out, n));
        return ZL_returnValue(1);
    }

    const uint8_t* const binIds = (const uint8_t*)ZL_Input_ptr(binIn);
    if (allZeroOffsets) {
        ZL_ERR_IF_NE(packedBytes, 0, corruption);
        for (size_t i = 0; i < n; i++) {
            uint8_t const b = binIds[i];
            ZL_ERR_IF_GE((size_t)b, numBins, corruption);
            pbs_store_elt(
                    dst,
                    i,
                    eltWidth,
                    pbs_from_latent(lowers[b], eltBits, latentMode));
        }
        ZL_ERR_IF_ERR(ZL_Output_commit(out, n));
        return ZL_returnValue(1);
    }

    uint64_t bitBase = 0;
    size_t i         = 0;
    while (i < n) {
        size_t const batchN = (n - i < PBS_BATCH_N) ? (n - i) : PBS_BATCH_N;
        uint64_t bLowers[PBS_BATCH_N];
        uint8_t bObits[PBS_BATCH_N];
        uint32_t bBitOff[PBS_BATCH_N];
        uint32_t batchBits = 0;
        for (size_t k = 0; k < batchN; k++) {
            uint8_t const b = binIds[i + k];
            ZL_ERR_IF_GE((size_t)b, numBins, corruption);
            bLowers[k]  = lowers[b];
            bObits[k]   = offsetBits[b];
            bBitOff[k]  = batchBits;
            batchBits += bObits[k];
        }
        ZL_ERR_IF_GT(
                (bitBase + (uint64_t)batchBits + 7ULL) >> 3,
                (uint64_t)packedBytes,
                corruption);
        for (size_t k = 0; k < batchN; k++) {
            uint64_t const off = pbs_read_bits(
                    offPtr, packedBytes, bitBase + bBitOff[k], bObits[k]);
            pbs_store_elt(
                    dst,
                    i + k,
                    eltWidth,
                    pbs_from_latent(bLowers[k] + off, eltBits, latentMode));
        }
        bitBase += batchBits;
        i += batchN;
    }

    ZL_ERR_IF_ERR(ZL_Output_commit(out, n));
    return ZL_returnValue(1);
}
