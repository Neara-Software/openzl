// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/entropy/decode_huffman_kernel.h"

#include <stddef.h> // size_t

#include <string.h>

#include "openzl/codecs/entropy/common_huffman_kernel.h"
#include "openzl/codecs/entropy/deprecated/common_entropy.h"
#include "openzl/fse/bitstream.h"
#include "openzl/shared/mem.h"
#include "openzl/shared/utils.h"
#include "openzl/zl_errors.h"

bool ZS_largeHuffmanValidWeights(
        const uint8_t* weights,
        size_t nbWeights,
        int tableLog)
{
    if (tableLog < 0 || tableLog > ZS_kLargeHuffmanMaxNbBits) {
        return false;
    }
    bool weightTooLarge = false;
    uint64_t sum        = 0;
    size_t i            = 0;
    while (i + 8 <= nbWeights) {
        if (ZL_read64(weights + i) == 0) {
            i += 8;
            continue;
        }
        for (size_t k = 0; k < 8; ++k) {
            int const w = weights[i + k];
            weightTooLarge |= (w > tableLog);
            sum += (((uint64_t)1) << (w & 63)) >> 1;
        }
        i += 8;
    }
    for (; i < nbWeights; ++i) {
        int const w = weights[i];
        weightTooLarge |= (w > tableLog);
        sum += (((uint64_t)1) << (w & 63)) >> 1;
    }
    // Guaranteed that there are at least 2 non-zero weights
    // by the fact that we're adding (1 << weight) >> 1, weights are <= the
    // tableLog, that the total is 1 << tableLog.
    return sum == ((uint64_t)1 << tableLog) && !weightTooLarge;
}

void ZS_largeHuffmanBuildDTable(
        ZS_Huf16DElt* dtable,
        const uint8_t* weights,
        size_t nbWeights,
        int tableLog)
{
    ZL_ASSERT(ZS_largeHuffmanValidWeights(weights, nbWeights, tableLog));

    int const maxSymbolValue1                    = (int)nbWeights;
    int rankStart[ZS_kLargeHuffmanMaxNbBits + 1] = { 0 };
    {
        int s = 0;
        while (s + 8 <= maxSymbolValue1) {
            if (ZL_read64(weights + s) == 0) {
                s += 8;
                continue;
            }
            for (int k = 0; k < 8; ++k) {
                uint8_t const w = weights[s + k];
                ZL_ASSERT_LT(w, ZL_ARRAY_SIZE(rankStart));
                if (w != 0) {
                    ++rankStart[w];
                }
            }
            s += 8;
        }
        for (; s < maxSymbolValue1; ++s) {
            uint8_t const w = weights[s];
            ZL_ASSERT_LT(w, ZL_ARRAY_SIZE(rankStart));
            if (w != 0) {
                ++rankStart[w];
            }
        }
    }
    rankStart[0] = 0;
    for (int r = 1, nextRankStart = 0; r < tableLog + 1; ++r) {
        int const current = nextRankStart;
        nextRankStart += rankStart[r] << (r - 1);
        rankStart[r] = current;
        ZL_ASSERT_LE(nextRankStart, (1 << tableLog));
    }

    int s = 0;
    while (s < maxSymbolValue1) {
        if (s + 8 <= maxSymbolValue1 && ZL_read64(weights + s) == 0) {
            s += 8;
            continue;
        }
        int const weight = weights[s];
        if (weight == 0) {
            ++s;
            continue;
        }
        int const length = 1 << (weight - 1);
        int const start  = rankStart[weight];
        ZL_ASSERT_LE(start + length, (1 << tableLog));
        ZS_Huf16DElt const elt = {
            .nbBits = (uint16_t)(tableLog + 1 - weight),
            .symbol = (uint16_t)s,
        };
        rankStart[weight] = start + length;
        if (length == 1) {
            dtable[start] = elt;
        } else {
            uint32_t raw32;
            memcpy(&raw32, &elt, sizeof(raw32));
            uint64_t const pair64 =
                    (uint64_t)raw32 | ((uint64_t)raw32 << 32);
            ZS_Huf16DElt* dst = dtable + start;
            int i             = 0;
            for (; i + 8 <= length; i += 8) {
                memcpy(dst + i + 0, &pair64, sizeof(pair64));
                memcpy(dst + i + 2, &pair64, sizeof(pair64));
                memcpy(dst + i + 4, &pair64, sizeof(pair64));
                memcpy(dst + i + 6, &pair64, sizeof(pair64));
            }
            for (; i < length; i += 2) {
                memcpy(dst + i, &pair64, sizeof(pair64));
            }
        }
        ++s;
    }
}

#define ZS_LARGE_HUF_STACK_TABLELOG 12
#define ZS_LARGE_HUF_STACK_WEIGHTS 4096

static ZS_Huf16DElt* ZS_largeHuffmanCreateDTable_internal(
        ZL_RC* src,
        int* tableLogPtr,
        ZS_Huf16DElt* stackDTable,
        size_t stackDTableElts)
{
    uint8_t stackWeights[ZS_LARGE_HUF_STACK_WEIGHTS];
    uint8_t* heapWeights = NULL;
    ZS_Huf16DElt* dtable = NULL;
    bool dtableOwned     = false;
    if (ZL_RC_avail(src) < 3) {
        goto _error;
    }
    int const tableLog = (int)ZL_RC_pop(src);
    if (tableLog > ZS_kLargeHuffmanMaxNbBits || tableLog == 0) {
        goto _error;
    }
    uint16_t const maxSymbolValue = ZL_RC_popCE16(src);
    size_t const maxSymbolValue1  = (size_t)maxSymbolValue + 1;
    uint8_t* weights              = stackWeights;
    if (maxSymbolValue1 > sizeof(stackWeights)) {
        heapWeights = (uint8_t*)malloc(maxSymbolValue1);
        if (heapWeights == NULL) {
            goto _error;
        }
        weights = heapWeights;
    }
    {
        ZL_Report const ret =
                ZS_Entropy_decodeDefault(weights, maxSymbolValue1, src, 1);
        if (ZL_isError(ret) || ZL_validResult(ret) != maxSymbolValue1) {
            goto _error;
        }
    }

    if (!ZS_largeHuffmanValidWeights(weights, maxSymbolValue1, tableLog)) {
        goto _error;
    }

    size_t const tableElts = (size_t)1 << tableLog;
    if (stackDTable != NULL && tableElts <= stackDTableElts) {
        dtable = stackDTable;
    } else {
        dtable = (ZS_Huf16DElt*)malloc(tableElts * sizeof(ZS_Huf16DElt));
        if (dtable == NULL) {
            goto _error;
        }
        dtableOwned = true;
    }

    ZS_largeHuffmanBuildDTable(dtable, weights, maxSymbolValue1, tableLog);

    *tableLogPtr = tableLog;

    free(heapWeights);
    return dtable;

_error:
    free(heapWeights);
    if (dtableOwned) {
        free(dtable);
    }
    return NULL;
}

ZS_Huf16DElt* ZS_largeHuffmanCreateDTable(ZL_RC* src, int* tableLogPtr)
{
    return ZS_largeHuffmanCreateDTable_internal(src, tableLogPtr, NULL, 0);
}

ZL_FORCE_INLINE ZS_Huf16DElt
ZS_loadHuf16DElt(ZS_Huf16DElt const* dtable, size_t idx)
{
    ZS_Huf16DElt elt;
    uint32_t const raw = ZL_read32(dtable + idx);
    memcpy(&elt, &raw, sizeof(elt));
    return elt;
}

ZL_FORCE_INLINE uint16_t ZS_largeHuffmanDecodeSymbol(
        BIT_DStream_t* dstream,
        ZS_Huf16DElt const* dtable,
        int const tableLog)
{
    size_t const val = BIT_lookBitsFast(
            dstream, (unsigned)tableLog); /* note : dtLog >= 1 */
    ZS_Huf16DElt const elt = ZS_loadHuf16DElt(dtable, val);
    BIT_skipBits(dstream, elt.nbBits);
    return elt.symbol;
}

ZL_FORCE_INLINE void ZS_largeHuffmanDecode_body(
        uint16_t* restrict dst,
        size_t dstSize,
        BIT_DStream_t* dstream,
        ZS_Huf16DElt const* restrict dtable,
        int tableLog,
        int const kUnroll)
{
    uint16_t* const dstEnd = dst + dstSize;
    uint16_t* restrict ptr = dst;
    if (dstSize >= (size_t)kUnroll) {
        uint16_t* const fastLimit = dstEnd - kUnroll;
        if (ZL_64bits() && (ptr < fastLimit)) {
            unsigned const rshift     = (unsigned)(64 - tableLog) & 63u;
            uint64_t bc               = dstream->bitContainer;
            unsigned cons             = dstream->bitsConsumed;
            const char* restrict bptr = dstream->ptr;
            const char* const blim    = dstream->limitPtr;
            while ((ptr < fastLimit) & (bptr >= blim)) {
                bptr -= cons >> 3;
                cons &= 7;
                bc = MEM_readLE64(bptr);
                for (int u = 0; u < kUnroll; ++u) {
                    size_t const idx = (size_t)((bc << (cons & 63)) >> rshift);
                    ZS_Huf16DElt const elt = dtable[idx];
                    cons += elt.nbBits;
                    ptr[u] = elt.symbol;
                }
                ptr += kUnroll;
            }
            dstream->bitContainer = bc;
            dstream->bitsConsumed = cons;
            dstream->ptr          = bptr;
        }
        while ((BIT_reloadDStream(dstream) == BIT_DStream_unfinished)
               & (ptr < fastLimit)) {
            for (int u = 0; u < kUnroll; ++u) {
                *ptr++ = ZS_largeHuffmanDecodeSymbol(dstream, dtable, tableLog);
            }
        }
    }
    BIT_reloadDStream(dstream);
    while (ptr < dstEnd) {
        *ptr++ = ZS_largeHuffmanDecodeSymbol(dstream, dtable, tableLog);
    }
    ZL_ASSERT_EQ(ptr, dstEnd);
}

ZL_FORCE_INLINE void ZS_largeHuffmanDecodeX4_fast64_impl(
        uint16_t* ptrs[4],
        uint16_t* const dsts[5],
        BIT_DStream_t dstreams[4],
        ZS_Huf16DElt const* restrict dtable,
        unsigned const rshift,
        int const kUnroll)
{
    ptrdiff_t const stride = dsts[1] - dsts[0];
    uint16_t* const limit  = dsts[4] - kUnroll;
    if ((stride <= 0) || ((dsts[2] - dsts[1]) != stride)
        || ((dsts[3] - dsts[2]) != stride) || ((dsts[4] - dsts[3]) > stride)
        || (ptrs[3] >= limit)) {
        return;
    }
    const char* restrict bptr0 = dstreams[0].ptr;
    const char* restrict bptr1 = dstreams[1].ptr;
    const char* restrict bptr2 = dstreams[2].ptr;
    const char* restrict bptr3 = dstreams[3].ptr;
    const char* const blim0    = dstreams[0].limitPtr;
    const char* const blim1    = dstreams[1].limitPtr;
    const char* const blim2    = dstreams[2].limitPtr;
    const char* const blim3    = dstreams[3].limitPtr;
    if ((bptr0 < blim0) | (bptr1 < blim1) | (bptr2 < blim2) | (bptr3 < blim3)) {
        return;
    }

    unsigned cons0         = dstreams[0].bitsConsumed;
    unsigned cons1         = dstreams[1].bitsConsumed;
    unsigned cons2         = dstreams[2].bitsConsumed;
    unsigned cons3         = dstreams[3].bitsConsumed;
    size_t const ustride   = (size_t)stride;
    uint16_t* restrict p0  = ptrs[0];
    uint16_t* const limit0 = dsts[0] + (limit - dsts[3]);

    while ((p0 < limit0) && (bptr0 >= blim0) && (bptr1 >= blim1)
           && (bptr2 >= blim2) && (bptr3 >= blim3)) {
        size_t const dstIters =
                ((size_t)(limit0 - p0) + (size_t)(kUnroll - 1)) / (size_t)kUnroll;
        size_t minBytes = (size_t)(bptr0 - blim0);
        minBytes        = ZL_MIN(minBytes, (size_t)(bptr1 - blim1));
        minBytes        = ZL_MIN(minBytes, (size_t)(bptr2 - blim2));
        minBytes        = ZL_MIN(minBytes, (size_t)(bptr3 - blim3));
        size_t const srcIters = 1 + minBytes / 8;
        size_t iters          = ZL_MIN(dstIters, srcIters);
        do {
            bptr0 -= cons0 >> 3;
            cons0 &= 7;
            uint64_t const bc0 = MEM_readLE64(bptr0);

            bptr1 -= cons1 >> 3;
            cons1 &= 7;
            uint64_t const bc1 = MEM_readLE64(bptr1);

            bptr2 -= cons2 >> 3;
            cons2 &= 7;
            uint64_t const bc2 = MEM_readLE64(bptr2);

            bptr3 -= cons3 >> 3;
            cons3 &= 7;
            uint64_t const bc3 = MEM_readLE64(bptr3);

            uint16_t* const q1 = p0 + ustride;
            uint16_t* const q2 = p0 + 2 * ustride;
            uint16_t* const q3 = q1 + 2 * ustride;

            for (int u = 0; u < kUnroll; ++u) {
                ZS_Huf16DElt const e0 = dtable[(size_t)((bc0 << (cons0 & 63)) >> rshift)];
                cons0 += e0.nbBits;
                p0[u] = e0.symbol;

                ZS_Huf16DElt const e1 = dtable[(size_t)((bc1 << (cons1 & 63)) >> rshift)];
                cons1 += e1.nbBits;
                q1[u] = e1.symbol;

                ZS_Huf16DElt const e2 = dtable[(size_t)((bc2 << (cons2 & 63)) >> rshift)];
                cons2 += e2.nbBits;
                q2[u] = e2.symbol;

                ZS_Huf16DElt const e3 = dtable[(size_t)((bc3 << (cons3 & 63)) >> rshift)];
                cons3 += e3.nbBits;
                q3[u] = e3.symbol;
            }
            p0 += kUnroll;
        } while (--iters);
    }

    ptrdiff_t const done     = p0 - dsts[0];
    ptrs[0]                  = dsts[0] + done;
    ptrs[1]                  = dsts[1] + done;
    ptrs[2]                  = dsts[2] + done;
    ptrs[3]                  = dsts[3] + done;
    dstreams[0].ptr          = bptr0;
    dstreams[0].bitsConsumed = cons0;
    dstreams[0].bitContainer = MEM_readLE64(bptr0);
    dstreams[1].ptr          = bptr1;
    dstreams[1].bitsConsumed = cons1;
    dstreams[1].bitContainer = MEM_readLE64(bptr1);
    dstreams[2].ptr          = bptr2;
    dstreams[2].bitsConsumed = cons2;
    dstreams[2].bitContainer = MEM_readLE64(bptr2);
    dstreams[3].ptr          = bptr3;
    dstreams[3].bitsConsumed = cons3;
    dstreams[3].bitContainer = MEM_readLE64(bptr3);
}

ZL_FORCE_NOINLINE void ZS_largeHuffmanDecodeX4_fast64_u4(
        uint16_t* ptrs[4],
        uint16_t* const dsts[5],
        BIT_DStream_t dstreams[4],
        ZS_Huf16DElt const* restrict dtable,
        unsigned const rshift)
{
    ZS_largeHuffmanDecodeX4_fast64_impl(ptrs, dsts, dstreams, dtable, rshift, 4);
}

ZL_FORCE_NOINLINE void ZS_largeHuffmanDecodeX4_fast64_u3(
        uint16_t* ptrs[4],
        uint16_t* const dsts[5],
        BIT_DStream_t dstreams[4],
        ZS_Huf16DElt const* restrict dtable,
        unsigned const rshift)
{
    ZS_largeHuffmanDecodeX4_fast64_impl(ptrs, dsts, dstreams, dtable, rshift, 3);
}

ZL_FORCE_NOINLINE void ZS_largeHuffmanDecodeX4_fast64_u2(
        uint16_t* ptrs[4],
        uint16_t* const dsts[5],
        BIT_DStream_t dstreams[4],
        ZS_Huf16DElt const* restrict dtable,
        unsigned const rshift)
{
    ZS_largeHuffmanDecodeX4_fast64_impl(ptrs, dsts, dstreams, dtable, rshift, 2);
}

ZL_FORCE_INLINE ZL_Report ZS_largeHuffmanDecodeX4_body(
        uint16_t* dsts[5],
        BIT_DStream_t dstreams[4],
        ZS_Huf16DElt const* restrict dtable,
        int tableLog,
        int const kUnroll)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(NULL);
    uint16_t* ptrs[4];
    memcpy(ptrs, dsts, sizeof(ptrs));
    if ((size_t)(dsts[4] - dsts[3]) >= (size_t)kUnroll) {
        uint16_t* const limit = dsts[4] - kUnroll;
        int endSignal         = 1;
        if (ZL_64bits()) {
            unsigned const rshift = (unsigned)(64 - tableLog) & 63u;
            if (kUnroll == 4) {
                ZS_largeHuffmanDecodeX4_fast64_u4(ptrs, dsts, dstreams, dtable, rshift);
            } else if (kUnroll == 3) {
                ZS_largeHuffmanDecodeX4_fast64_u3(ptrs, dsts, dstreams, dtable, rshift);
            } else if (kUnroll == 2) {
                ZS_largeHuffmanDecodeX4_fast64_u2(ptrs, dsts, dstreams, dtable, rshift);
            }
            for (int i = 0; i < 4; ++i) {
                endSignal &=
                        (BIT_reloadDStream(&dstreams[i])
                         == BIT_DStream_unfinished);
            }
        }
        for (; endSignal & (ptrs[3] < limit);) {
            for (int u = 0; u < kUnroll; ++u) {
                for (int i = 0; i < 4; ++i) {
                    *ptrs[i]++ = ZS_largeHuffmanDecodeSymbol(
                            &dstreams[i], dtable, tableLog);
                }
            }
            for (int i = 0; i < 4; ++i) {
                endSignal &=
                        (BIT_reloadDStream(&dstreams[i])
                         == BIT_DStream_unfinished);
            }
        }
    }
    for (int i = 0; i < 4; ++i) {
        if (ptrs[i] > dsts[i + 1]) {
            ZL_ERR(GENERIC);
        }
        ZS_largeHuffmanDecode_body(
                ptrs[i],
                (size_t)(dsts[i + 1] - ptrs[i]),
                &dstreams[i],
                dtable,
                tableLog,
                kUnroll);
    }
    return ZL_returnValue((size_t)(dsts[4] - dsts[0]));
}

ZL_Report ZS_largeHuffmanDecodeUsingDTableX4(
        uint16_t* dst,
        size_t capacity,
        ZL_RC* src,
        ZS_Huf16DElt const* dtable,
        int tableLog)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(NULL);
    uint16_t* dsts[5];
    BIT_DStream_t dstreams[4];
    uint16_t* dstEnd = dst;
    for (int i = 0; i < 4; ++i) {
        dsts[i] = dstEnd;
        if (ZL_RC_avail(src) < 8) {
            ZL_ERR(GENERIC);
        }
        uint32_t const dstSize = ZL_RC_popCE32(src);
        uint32_t const srcSize = ZL_RC_popCE32(src);
        if (ZL_RC_avail(src) < srcSize
            || capacity < (size_t)(dstSize + (dstEnd - dst))) {
            ZL_ERR(GENERIC);
        }
        if (ERR_isError(
                    BIT_initDStream(&dstreams[i], ZL_RC_ptr(src), srcSize))) {
            ZL_ERR(GENERIC);
        }
        ZL_RC_advance(src, srcSize);
        dstEnd += dstSize;
    }
    dsts[4] = dstEnd;

    if (ZL_64bits()) {
        if (tableLog <= 14) {
            return ZS_largeHuffmanDecodeX4_body(
                    dsts, dstreams, dtable, tableLog, 4);
        } else if (tableLog <= 18) {
            return ZS_largeHuffmanDecodeX4_body(
                    dsts, dstreams, dtable, tableLog, 3);
        } else {
            return ZS_largeHuffmanDecodeX4_body(
                    dsts, dstreams, dtable, tableLog, 2);
        }
    } else {
        if (tableLog <= 14) {
            return ZS_largeHuffmanDecodeX4_body(
                    dsts, dstreams, dtable, tableLog, 2);
        } else {
            return ZS_largeHuffmanDecodeX4_body(
                    dsts, dstreams, dtable, tableLog, 1);
        }
    }
}

ZL_Report ZS_largeHuffmanDecodeUsingDTable(
        uint16_t* dst,
        size_t capacity,
        ZL_RC* src,
        ZS_Huf16DElt const* dtable,
        int tableLog)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(NULL);
    if (ZL_RC_avail(src) < 8) {
        ZL_ERR(GENERIC);
    }
    uint32_t const dstSize = ZL_RC_popCE32(src);
    uint32_t const srcSize = ZL_RC_popCE32(src);
    if (ZL_RC_avail(src) < srcSize || capacity < dstSize) {
        ZL_ERR(GENERIC);
    }
    BIT_DStream_t dstream;
    if (ERR_isError(BIT_initDStream(&dstream, ZL_RC_ptr(src), srcSize))) {
        ZL_ERR(GENERIC);
    }

    ZL_RC_advance(src, srcSize);

    if (ZL_64bits()) {
        if (tableLog <= 14) {
            ZS_largeHuffmanDecode_body(
                    dst, dstSize, &dstream, dtable, tableLog, 4);
        } else if (tableLog <= 18) {
            ZS_largeHuffmanDecode_body(
                    dst, dstSize, &dstream, dtable, tableLog, 3);
        } else {
            ZS_largeHuffmanDecode_body(
                    dst, dstSize, &dstream, dtable, tableLog, 2);
        }
    } else {
        if (tableLog <= 14) {
            ZS_largeHuffmanDecode_body(
                    dst, dstSize, &dstream, dtable, tableLog, 2);
        } else {
            ZS_largeHuffmanDecode_body(
                    dst, dstSize, &dstream, dtable, tableLog, 1);
        }
    }

    return ZL_returnValue(dstSize);
}

ZL_Report ZS_largeHuffmanDecode(uint16_t* dst, size_t capacity, ZL_RC* src)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(NULL);
    if (ZL_RC_avail(src) < 1) {
        ZL_ERR(GENERIC);
    }
    ZS_HufTransformPrefix_e const header =
            (ZS_HufTransformPrefix_e)ZL_RC_pop(src);
    if (header == ZS_HufTransformPrefix_constant) {
        ZL_TRY_LET_CONST(uint64_t, nelts, ZL_RC_popVarint(src));
        if (ZL_RC_avail(src) < sizeof(uint16_t)) {
            ZL_ERR(GENERIC);
        }
        uint16_t const value = ZL_RC_popCE16(src);
        if (capacity < nelts) {
            ZL_ERR(GENERIC);
        }
        for (size_t i = 0; i < nelts; ++i) {
            dst[i] = value;
        }
        return ZL_returnValue((size_t)nelts);
    }
    if (header == ZS_HufTransformPrefix_lit) {
        ZL_TRY_LET_CONST(uint64_t, nelts, ZL_RC_popVarint(src));
        if (capacity < nelts) {
            ZL_ERR(GENERIC);
        }
        if (ZL_RC_avail(src) < nelts * sizeof(uint16_t)) {
            ZL_ERR(GENERIC);
        }
        for (size_t i = 0; i < nelts; ++i) {
            dst[i] = ZL_RC_popCE16(src);
        }
        return ZL_returnValue((size_t)nelts);
    }
    if (header != ZS_HufTransformPrefix_huf) {
        ZL_ERR(GENERIC);
    }

    int tableLog;
    ZS_Huf16DElt stackDTable[1u << ZS_LARGE_HUF_STACK_TABLELOG];
    ZS_Huf16DElt* const dtable = ZS_largeHuffmanCreateDTable_internal(
            src, &tableLog, stackDTable, ZL_ARRAY_SIZE(stackDTable));
    if (dtable == NULL) {
        ZL_ERR(GENERIC);
    }

    // Decompress
    if (ZL_RC_avail(src) < 1) {
        if (dtable != stackDTable) {
            free(dtable);
        }
        ZL_ERR(GENERIC);
    }

    bool const x4 = (bool)ZL_RC_pop(src);
    ZL_Report dstSize;
    if (x4) {
        dstSize = ZS_largeHuffmanDecodeUsingDTableX4(
                dst, capacity, src, dtable, tableLog);
    } else {
        dstSize = ZS_largeHuffmanDecodeUsingDTable(
                dst, capacity, src, dtable, tableLog);
    }

    if (dtable != stackDTable) {
        free(dtable);
    }

    return dstSize;
}
