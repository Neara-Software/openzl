// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/pco_bin_split/decode_pco_bin_split_binding.h"

#include <stdint.h>
#include <string.h>

#include "openzl/codecs/zl_pco_bin_split.h"
#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
#include "openzl/shared/mem.h"
#include "openzl/shared/portability.h"
#include "openzl/shared/varint.h"

enum {
    PBS_MAX_BINS = 256,
    PBS_BATCH_N  = 256,
};

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

#define PBS_FROM_LATENT_UNSIGNED(l, eltBits) (l)
#define PBS_FROM_LATENT_SIGNED(l, eltBits) ((l) ^ (1ULL << ((eltBits) - 1u)))
#define PBS_FROM_LATENT_FLOAT(l, eltBits)     (((l) & (1ULL << ((eltBits) - 1u))) != 0              ? ((l) ^ (1ULL << ((eltBits) - 1u)))              : (~(l)))

#define PBS_DEFINE_DECODER(NAME, TYPE, ELT_BITS, FROM_LATENT)                \
    static ZL_Report pbs_decode_##NAME(                                      \
            ZL_Decoder* dictx,                                               \
            TYPE* dst,                                                       \
            size_t n,                                                        \
            int singleBin,                                                   \
            int allZeroOffsets,                                              \
            size_t numBins,                                                  \
            const uint64_t lowers[PBS_MAX_BINS],                             \
            const uint8_t offsetBits[PBS_MAX_BINS],                          \
            const uint64_t masks[PBS_MAX_BINS],                              \
            unsigned maxOb,                                                  \
            const uint8_t* binIds,                                           \
            const uint8_t* offPtr,                                           \
            size_t packedBytes)                                              \
    {                                                                        \
        ZL_RESULT_DECLARE_SCOPE_REPORT(dictx);                               \
        if (singleBin) {                                                     \
            uint64_t const lo       = lowers[0];                             \
            unsigned const ob       = offsetBits[0];                         \
            uint64_t const mask     = masks[0];                              \
            uint64_t const needBits = (uint64_t)n * (uint64_t)ob;            \
            ZL_ERR_IF_GT(                                                    \
                    (needBits + 7ULL) >> 3,                                  \
                    (uint64_t)packedBytes,                                   \
                    corruption);                                             \
            if (ob == 0u) {                                                  \
                TYPE const val = (TYPE)(FROM_LATENT(lo, (ELT_BITS)));        \
                for (size_t i = 0; i < n; i++) {                             \
                    dst[i] = val;                                            \
                }                                                            \
                return ZL_returnValue(1);                                    \
            }                                                                \
            size_t i        = 0;                                             \
            uint64_t bitPos = 0;                                             \
            if (ob <= 57u && packedBytes >= 8u) {                            \
                uint64_t const safeLimit =                                   \
                        (uint64_t)(packedBytes - 8u) * 8ULL + 7ULL;          \
                size_t fastN = 0;                                            \
                if (needBits - ob <= safeLimit) {                            \
                    fastN = n;                                               \
                } else {                                                     \
                    fastN = (size_t)(safeLimit / ob) + 1u;                   \
                    if (fastN > n) {                                         \
                        fastN = n;                                           \
                    }                                                        \
                    while (fastN > 0u                                        \
                           && (uint64_t)(fastN - 1u) * ob > safeLimit) {     \
                        fastN--;                                             \
                    }                                                        \
                }                                                            \
                for (; i + 8u <= fastN; i += 8u) {                           \
                    uint64_t const bp0 = bitPos;                             \
                    uint64_t const bp1 = bitPos + (uint64_t)ob * 1u;         \
                    uint64_t const bp2 = bitPos + (uint64_t)ob * 2u;         \
                    uint64_t const bp3 = bitPos + (uint64_t)ob * 3u;         \
                    uint64_t const bp4 = bitPos + (uint64_t)ob * 4u;         \
                    uint64_t const bp5 = bitPos + (uint64_t)ob * 5u;         \
                    uint64_t const bp6 = bitPos + (uint64_t)ob * 6u;         \
                    uint64_t const bp7 = bitPos + (uint64_t)ob * 7u;         \
                    uint64_t const off0 =                                    \
                            (ZL_readLE64(offPtr + (bp0 >> 3)) >> (bp0 & 7u)) \
                            & mask;                                          \
                    uint64_t const off1 =                                    \
                            (ZL_readLE64(offPtr + (bp1 >> 3)) >> (bp1 & 7u)) \
                            & mask;                                          \
                    uint64_t const off2 =                                    \
                            (ZL_readLE64(offPtr + (bp2 >> 3)) >> (bp2 & 7u)) \
                            & mask;                                          \
                    uint64_t const off3 =                                    \
                            (ZL_readLE64(offPtr + (bp3 >> 3)) >> (bp3 & 7u)) \
                            & mask;                                          \
                    uint64_t const off4 =                                    \
                            (ZL_readLE64(offPtr + (bp4 >> 3)) >> (bp4 & 7u)) \
                            & mask;                                          \
                    uint64_t const off5 =                                    \
                            (ZL_readLE64(offPtr + (bp5 >> 3)) >> (bp5 & 7u)) \
                            & mask;                                          \
                    uint64_t const off6 =                                    \
                            (ZL_readLE64(offPtr + (bp6 >> 3)) >> (bp6 & 7u)) \
                            & mask;                                          \
                    uint64_t const off7 =                                    \
                            (ZL_readLE64(offPtr + (bp7 >> 3)) >> (bp7 & 7u)) \
                            & mask;                                          \
                    dst[i + 0u] = (TYPE)(FROM_LATENT(lo + off0, (ELT_BITS)));\
                    dst[i + 1u] = (TYPE)(FROM_LATENT(lo + off1, (ELT_BITS)));\
                    dst[i + 2u] = (TYPE)(FROM_LATENT(lo + off2, (ELT_BITS)));\
                    dst[i + 3u] = (TYPE)(FROM_LATENT(lo + off3, (ELT_BITS)));\
                    dst[i + 4u] = (TYPE)(FROM_LATENT(lo + off4, (ELT_BITS)));\
                    dst[i + 5u] = (TYPE)(FROM_LATENT(lo + off5, (ELT_BITS)));\
                    dst[i + 6u] = (TYPE)(FROM_LATENT(lo + off6, (ELT_BITS)));\
                    dst[i + 7u] = (TYPE)(FROM_LATENT(lo + off7, (ELT_BITS)));\
                    bitPos += (uint64_t)ob * 8u;                             \
                }                                                            \
                for (; i < fastN; i++) {                                     \
                    uint64_t const off =                                     \
                            (ZL_readLE64(offPtr + (bitPos >> 3))             \
                             >> (bitPos & 7u))                               \
                            & mask;                                          \
                    dst[i] = (TYPE)(FROM_LATENT(lo + off, (ELT_BITS)));      \
                    bitPos += ob;                                            \
                }                                                            \
            }                                                                \
            for (; i < n; i++) {                                             \
                uint64_t const off =                                         \
                        pbs_read_bits(offPtr, packedBytes, bitPos, ob);      \
                bitPos += ob;                                                \
                dst[i] = (TYPE)(FROM_LATENT(lo + off, (ELT_BITS)));          \
            }                                                                \
            return ZL_returnValue(1);                                        \
        }                                                                    \
                                                                             \
        if (allZeroOffsets) {                                                \
            ZL_ERR_IF_NE(packedBytes, 0, corruption);                        \
            TYPE decodedLowers[PBS_MAX_BINS];                                \
            for (size_t b = 0; b < numBins; b++) {                           \
                decodedLowers[b] =                                           \
                        (TYPE)(FROM_LATENT(lowers[b], (ELT_BITS)));          \
            }                                                                \
            if (numBins == PBS_MAX_BINS) {                                   \
                for (size_t i = 0; i < n; i++) {                             \
                    dst[i] = decodedLowers[binIds[i]];                       \
                }                                                            \
            } else {                                                         \
                for (size_t i = 0; i < n; i++) {                             \
                    uint8_t const b = binIds[i];                             \
                    ZL_ERR_IF_GE((size_t)b, numBins, corruption);            \
                    dst[i] = decodedLowers[b];                               \
                }                                                            \
            }                                                                \
            return ZL_returnValue(1);                                        \
        }                                                                    \
                                                                             \
        uint64_t const safeLimit = (packedBytes >= 8u)                       \
                ? ((uint64_t)(packedBytes - 8u) * 8ULL + 7ULL)               \
                : 0ULL;                                                      \
        uint64_t bitBase         = 0;                                        \
        size_t i                 = 0;                                        \
        while (i < n) {                                                      \
            size_t const batchN =                                            \
                    (n - i < PBS_BATCH_N) ? (n - i) : PBS_BATCH_N;           \
            uint64_t bLowers[PBS_BATCH_N];                                   \
            uint64_t bMasks[PBS_BATCH_N];                                    \
            uint32_t bBitOff[PBS_BATCH_N];                                   \
            uint32_t batchBits = 0;                                          \
            if (numBins == PBS_MAX_BINS) {                                   \
                for (size_t k = 0; k < batchN; k++) {                        \
                    uint8_t const b = binIds[i + k];                         \
                    bLowers[k]      = lowers[b];                             \
                    bMasks[k]       = masks[b];                              \
                    bBitOff[k]      = batchBits;                             \
                    batchBits += offsetBits[b];                              \
                }                                                            \
            } else {                                                         \
                uint8_t badBin = 0;                                          \
                for (size_t k = 0; k < batchN; k++) {                        \
                    uint8_t const b = binIds[i + k];                         \
                    badBin |= (uint8_t)(b >= numBins);                       \
                    uint8_t const safeB = (b < numBins) ? b : 0u;            \
                    bLowers[k]          = lowers[safeB];                     \
                    bMasks[k]           = masks[safeB];                      \
                    bBitOff[k]          = batchBits;                         \
                    batchBits += offsetBits[safeB];                          \
                }                                                            \
                ZL_ERR_IF(badBin, corruption);                               \
            }                                                                \
            ZL_ERR_IF_GT(                                                    \
                    (bitBase + (uint64_t)batchBits + 7ULL) >> 3,             \
                    (uint64_t)packedBytes,                                   \
                    corruption);                                             \
            TYPE* const outPtr = dst + i;                                    \
            size_t k           = 0;                                          \
            if (ZL_LIKELY(maxOb <= 57u && packedBytes >= 8u)) {              \
                size_t fastN = batchN;                                       \
                if (ZL_UNLIKELY(bitBase + (uint64_t)batchBits > safeLimit)) {\
                    while (fastN > 0u                                        \
                           && (bitBase + bBitOff[fastN - 1u]) > safeLimit) { \
                        fastN--;                                             \
                    }                                                        \
                }                                                            \
                for (; k + 8u <= fastN; k += 8u) {                           \
                    uint64_t const bp0 = bitBase + bBitOff[k + 0u];          \
                    uint64_t const bp1 = bitBase + bBitOff[k + 1u];          \
                    uint64_t const bp2 = bitBase + bBitOff[k + 2u];          \
                    uint64_t const bp3 = bitBase + bBitOff[k + 3u];          \
                    uint64_t const bp4 = bitBase + bBitOff[k + 4u];          \
                    uint64_t const bp5 = bitBase + bBitOff[k + 5u];          \
                    uint64_t const bp6 = bitBase + bBitOff[k + 6u];          \
                    uint64_t const bp7 = bitBase + bBitOff[k + 7u];          \
                    uint64_t const off0 =                                    \
                            (ZL_readLE64(offPtr + (bp0 >> 3)) >> (bp0 & 7u)) \
                            & bMasks[k + 0u];                                \
                    uint64_t const off1 =                                    \
                            (ZL_readLE64(offPtr + (bp1 >> 3)) >> (bp1 & 7u)) \
                            & bMasks[k + 1u];                                \
                    uint64_t const off2 =                                    \
                            (ZL_readLE64(offPtr + (bp2 >> 3)) >> (bp2 & 7u)) \
                            & bMasks[k + 2u];                                \
                    uint64_t const off3 =                                    \
                            (ZL_readLE64(offPtr + (bp3 >> 3)) >> (bp3 & 7u)) \
                            & bMasks[k + 3u];                                \
                    uint64_t const off4 =                                    \
                            (ZL_readLE64(offPtr + (bp4 >> 3)) >> (bp4 & 7u)) \
                            & bMasks[k + 4u];                                \
                    uint64_t const off5 =                                    \
                            (ZL_readLE64(offPtr + (bp5 >> 3)) >> (bp5 & 7u)) \
                            & bMasks[k + 5u];                                \
                    uint64_t const off6 =                                    \
                            (ZL_readLE64(offPtr + (bp6 >> 3)) >> (bp6 & 7u)) \
                            & bMasks[k + 6u];                                \
                    uint64_t const off7 =                                    \
                            (ZL_readLE64(offPtr + (bp7 >> 3)) >> (bp7 & 7u)) \
                            & bMasks[k + 7u];                                \
                    outPtr[k + 0u] = (TYPE)(FROM_LATENT(                     \
                            bLowers[k + 0u] + off0, (ELT_BITS)));            \
                    outPtr[k + 1u] = (TYPE)(FROM_LATENT(                     \
                            bLowers[k + 1u] + off1, (ELT_BITS)));            \
                    outPtr[k + 2u] = (TYPE)(FROM_LATENT(                     \
                            bLowers[k + 2u] + off2, (ELT_BITS)));            \
                    outPtr[k + 3u] = (TYPE)(FROM_LATENT(                     \
                            bLowers[k + 3u] + off3, (ELT_BITS)));            \
                    outPtr[k + 4u] = (TYPE)(FROM_LATENT(                     \
                            bLowers[k + 4u] + off4, (ELT_BITS)));            \
                    outPtr[k + 5u] = (TYPE)(FROM_LATENT(                     \
                            bLowers[k + 5u] + off5, (ELT_BITS)));            \
                    outPtr[k + 6u] = (TYPE)(FROM_LATENT(                     \
                            bLowers[k + 6u] + off6, (ELT_BITS)));            \
                    outPtr[k + 7u] = (TYPE)(FROM_LATENT(                     \
                            bLowers[k + 7u] + off7, (ELT_BITS)));            \
                }                                                            \
                for (; k < fastN; k++) {                                     \
                    uint64_t const bp  = bitBase + bBitOff[k];               \
                    uint64_t const off =                                     \
                            (ZL_readLE64(offPtr + (bp >> 3)) >> (bp & 7u))   \
                            & bMasks[k];                                     \
                    outPtr[k] = (TYPE)(FROM_LATENT(                          \
                            bLowers[k] + off, (ELT_BITS)));                  \
                }                                                            \
            }                                                                \
            for (; k < batchN; k++) {                                        \
                unsigned const ob  = offsetBits[binIds[i + k]];              \
                uint64_t const off = pbs_read_bits(                          \
                        offPtr, packedBytes, bitBase + bBitOff[k], ob);      \
                outPtr[k] =                                                  \
                        (TYPE)(FROM_LATENT(bLowers[k] + off, (ELT_BITS)));   \
            }                                                                \
            bitBase += batchBits;                                            \
            i += batchN;                                                     \
        }                                                                    \
        return ZL_returnValue(1);                                            \
    }

PBS_DEFINE_DECODER(u8, uint8_t, 8u, PBS_FROM_LATENT_UNSIGNED)
PBS_DEFINE_DECODER(s8, uint8_t, 8u, PBS_FROM_LATENT_SIGNED)
PBS_DEFINE_DECODER(u16, uint16_t, 16u, PBS_FROM_LATENT_UNSIGNED)
PBS_DEFINE_DECODER(s16, uint16_t, 16u, PBS_FROM_LATENT_SIGNED)
PBS_DEFINE_DECODER(f16, uint16_t, 16u, PBS_FROM_LATENT_FLOAT)
PBS_DEFINE_DECODER(u32, uint32_t, 32u, PBS_FROM_LATENT_UNSIGNED)
PBS_DEFINE_DECODER(s32, uint32_t, 32u, PBS_FROM_LATENT_SIGNED)
PBS_DEFINE_DECODER(f32, uint32_t, 32u, PBS_FROM_LATENT_FLOAT)
PBS_DEFINE_DECODER(u64, uint64_t, 64u, PBS_FROM_LATENT_UNSIGNED)
PBS_DEFINE_DECODER(s64, uint64_t, 64u, PBS_FROM_LATENT_SIGNED)
PBS_DEFINE_DECODER(f64, uint64_t, 64u, PBS_FROM_LATENT_FLOAT)

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

    uint8_t const flags      = *ptr++;
    unsigned const wLog2     = flags & 3u;
    int const latentMode     = (int)((flags >> 2) & 3u);
    int const singleBin      = (int)((flags >> 4) & 1u);
    int const allZeroOffsets = (int)((flags >> 5) & 1u);
    ZL_ERR_IF_GT(latentMode, 2, corruption);

    size_t const eltWidth  = (size_t)1u << wLog2;
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
    uint64_t masks[PBS_MAX_BINS];
    uint64_t prevLower = 0;
    unsigned maxOb     = 0;
    for (size_t b = 0; b < numBins; b++) {
        ZL_TRY_LET(uint64_t, deltaLo, ZL_varintDecode(&ptr, end));
        lowers[b] = prevLower + deltaLo;
        prevLower = lowers[b];
        if (allZeroOffsets) {
            offsetBits[b] = 0;
            masks[b]      = 0;
        } else {
            ZL_ERR_IF_GE(ptr, end, corruption);
            uint8_t const ob = *ptr++;
            ZL_ERR_IF_GT(ob, eltBits, corruption);
            offsetBits[b] = ob;
            masks[b] = (ob == 0u) ? 0ULL
                    : (ob == 64u) ? UINT64_MAX
                                  : ((1ULL << ob) - 1ULL);
            if (ob > maxOb) {
                maxOb = ob;
            }
        }
    }
    ZL_ERR_IF_NE(ptr, end, corruption);

    ZL_ERR_IF_NE(ZL_Input_numElts(binIn), singleBin ? 0 : n, corruption);
    const uint8_t* const offPtr = (const uint8_t*)ZL_Input_ptr(offIn);
    size_t const packedBytes    = ZL_Input_numElts(offIn);
    const uint8_t* const binIds = (const uint8_t*)ZL_Input_ptr(binIn);
    void* const dst             = ZL_Output_ptr(out);

    unsigned const dispatchKey = (wLog2 << 2)
            | (unsigned)((wLog2 == 0u && latentMode == ZL_PcoBinSplit_float)
                                 ? ZL_PcoBinSplit_unsigned
                                 : latentMode);
    ZL_Report decRes = ZL_returnValue(1);
    switch (dispatchKey) {
        case (0u << 2) | ZL_PcoBinSplit_unsigned:
            decRes = pbs_decode_u8(
                    dictx, (uint8_t*)dst, n, singleBin, allZeroOffsets, numBins,
                    lowers, offsetBits, masks, maxOb, binIds, offPtr, packedBytes);
            break;
        case (0u << 2) | ZL_PcoBinSplit_signed:
            decRes = pbs_decode_s8(
                    dictx, (uint8_t*)dst, n, singleBin, allZeroOffsets, numBins,
                    lowers, offsetBits, masks, maxOb, binIds, offPtr, packedBytes);
            break;
        case (1u << 2) | ZL_PcoBinSplit_unsigned:
            decRes = pbs_decode_u16(
                    dictx, (uint16_t*)dst, n, singleBin, allZeroOffsets, numBins,
                    lowers, offsetBits, masks, maxOb, binIds, offPtr, packedBytes);
            break;
        case (1u << 2) | ZL_PcoBinSplit_signed:
            decRes = pbs_decode_s16(
                    dictx, (uint16_t*)dst, n, singleBin, allZeroOffsets, numBins,
                    lowers, offsetBits, masks, maxOb, binIds, offPtr, packedBytes);
            break;
        case (1u << 2) | ZL_PcoBinSplit_float:
            decRes = pbs_decode_f16(
                    dictx, (uint16_t*)dst, n, singleBin, allZeroOffsets, numBins,
                    lowers, offsetBits, masks, maxOb, binIds, offPtr, packedBytes);
            break;
        case (2u << 2) | ZL_PcoBinSplit_unsigned:
            decRes = pbs_decode_u32(
                    dictx, (uint32_t*)dst, n, singleBin, allZeroOffsets, numBins,
                    lowers, offsetBits, masks, maxOb, binIds, offPtr, packedBytes);
            break;
        case (2u << 2) | ZL_PcoBinSplit_signed:
            decRes = pbs_decode_s32(
                    dictx, (uint32_t*)dst, n, singleBin, allZeroOffsets, numBins,
                    lowers, offsetBits, masks, maxOb, binIds, offPtr, packedBytes);
            break;
        case (2u << 2) | ZL_PcoBinSplit_float:
            decRes = pbs_decode_f32(
                    dictx, (uint32_t*)dst, n, singleBin, allZeroOffsets, numBins,
                    lowers, offsetBits, masks, maxOb, binIds, offPtr, packedBytes);
            break;
        case (3u << 2) | ZL_PcoBinSplit_unsigned:
            decRes = pbs_decode_u64(
                    dictx, (uint64_t*)dst, n, singleBin, allZeroOffsets, numBins,
                    lowers, offsetBits, masks, maxOb, binIds, offPtr, packedBytes);
            break;
        case (3u << 2) | ZL_PcoBinSplit_signed:
            decRes = pbs_decode_s64(
                    dictx, (uint64_t*)dst, n, singleBin, allZeroOffsets, numBins,
                    lowers, offsetBits, masks, maxOb, binIds, offPtr, packedBytes);
            break;
        case (3u << 2) | ZL_PcoBinSplit_float:
            decRes = pbs_decode_f64(
                    dictx, (uint64_t*)dst, n, singleBin, allZeroOffsets, numBins,
                    lowers, offsetBits, masks, maxOb, binIds, offPtr, packedBytes);
            break;
        default:
            ZL_ASSERT_FAIL("Unhandled pco_bin_split dispatch key");
            return ZL_REPORT_ERROR(GENERIC);
    }
    ZL_ERR_IF_ERR(decRes);
    ZL_ERR_IF_ERR(ZL_Output_commit(out, n));
    return ZL_returnValue(1);
}
