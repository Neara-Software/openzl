// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/ctx_split/decode_ctx_split_binding.h"

#include <stdint.h>
#include <string.h>

#include "openzl/codecs/ctx_split/common_ctx_split.h"
#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
#include "openzl/decompress/dctx2.h"
#include "openzl/decompress/dictx.h"
#include "openzl/shared/overflow.h"
#include "openzl/shared/varint.h"
#include "openzl/zl_output.h"

static ZL_Output* ctx_createOutStream(
        ZL_Decoder* dictx,
        size_t eltsCapacity,
        size_t eltWidth,
        ZL_Type st)
{
    size_t dstBytes = 0;
    if (eltWidth == 0 || ZL_overflowMulST(eltsCapacity, eltWidth, &dstBytes)) {
        return NULL;
    }
    if (dictx == NULL || dictx->nbRegens < 1) {
        return NULL;
    }
    return ZL_codemodDataAsOutput(DCTX_newStream(
            dictx->dctx, dictx->regensID[0], st, eltWidth, eltsCapacity));
}

#define CTX_DEFINE_DECODE_OPS(W, T)                                            \
    static int ctx_gather_mode0_##W(                                           \
            T* restrict dst,                                                   \
            size_t totalElts,                                                  \
            const void* const srcPtrs[CTX_SPLIT_MAX_K],                        \
            const size_t srcCounts[CTX_SPLIT_MAX_K],                           \
            uint64_t minVal,                                                   \
            unsigned shift,                                                    \
            uint32_t maxCls)                                                   \
    {                                                                          \
        const T* cur[CTX_SPLIT_MAX_K];                                         \
        size_t rem[CTX_SPLIT_MAX_K];                                           \
        for (size_t c = 0; c < CTX_SPLIT_MAX_K; c++) {                         \
            cur[c] = (const T*)srcPtrs[c];                                     \
            rem[c] = srcCounts[c];                                             \
        }                                                                      \
        uint64_t prev = 0;                                                     \
        for (size_t i = 0; i < totalElts; i++) {                               \
            uint32_t const c = ctx_cls_mode0(prev, minVal, shift, maxCls);     \
            if (ZL_UNLIKELY(rem[c] == 0)) {                                    \
                return 0;                                                      \
            }                                                                  \
            rem[c]--;                                                          \
            T const val = *cur[c]++;                                           \
            dst[i]      = val;                                                 \
            prev        = (uint64_t)val;                                       \
        }                                                                      \
        return 1;                                                              \
    }                                                                          \
                                                                               \
    static int ctx_gather_mode1_##W(                                           \
            T* restrict dst,                                                   \
            size_t totalElts,                                                  \
            const void* const srcPtrs[CTX_SPLIT_MAX_K],                        \
            const size_t srcCounts[CTX_SPLIT_MAX_K],                           \
            uint32_t maxCls)                                                   \
    {                                                                          \
        const T* cur[CTX_SPLIT_MAX_K];                                         \
        size_t rem[CTX_SPLIT_MAX_K];                                           \
        for (size_t c = 0; c < CTX_SPLIT_MAX_K; c++) {                         \
            cur[c] = (const T*)srcPtrs[c];                                     \
            rem[c] = srcCounts[c];                                             \
        }                                                                      \
        uint64_t prev = 0;                                                     \
        for (size_t i = 0; i < totalElts; i++) {                               \
            uint32_t const c = ctx_cls_mode1(prev, maxCls);                    \
            if (ZL_UNLIKELY(rem[c] == 0)) {                                    \
                return 0;                                                      \
            }                                                                  \
            rem[c]--;                                                          \
            T const val = *cur[c]++;                                           \
            dst[i]      = val;                                                 \
            prev        = (uint64_t)val;                                       \
        }                                                                      \
        return 1;                                                              \
    }

CTX_DEFINE_DECODE_OPS(8, uint8_t)
CTX_DEFINE_DECODE_OPS(16, uint16_t)
CTX_DEFINE_DECODE_OPS(32, uint32_t)
CTX_DEFINE_DECODE_OPS(64, uint64_t)

static int ctx_isValidTypeWidth(ZL_Type st, size_t eltWidth)
{
    if (st == ZL_Type_serial) {
        return eltWidth == 1;
    }
    if (st == ZL_Type_numeric) {
        return eltWidth == 1 || eltWidth == 2 || eltWidth == 4 || eltWidth == 8;
    }
    return 0;
}

ZL_Report DI_ctx_split(
        ZL_Decoder* dictx,
        const ZL_Input* compulsorySrcs[],
        size_t nbCompulsorySrcs,
        const ZL_Input* variableSrcs[],
        size_t nbVariableSrcs)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dictx);
    ZL_ASSERT_NN(dictx);
    (void)compulsorySrcs;
    ZL_ERR_IF_NE(nbCompulsorySrcs, 0, corruption);

    ZL_RBuffer const hdr = ZL_Decoder_getCodecHeader(dictx);
    ZL_ERR_IF_LT(hdr.size, 1, corruption);

    const uint8_t* ptr       = (const uint8_t*)hdr.start;
    const uint8_t* const end = ptr + hdr.size;

    uint8_t const flagsByte = *ptr++;
    unsigned const mode     = (unsigned)(flagsByte >> 4);
    unsigned const nCtxLog  = (unsigned)(flagsByte & 0x0Fu);

    ZL_ERR_IF_GT(mode, CTX_SPLIT_MODE_PASSTHROUGH, corruption);
    ZL_ERR_IF_LT(nCtxLog, CTX_SPLIT_MIN_NCTX_LOG, corruption);
    ZL_ERR_IF_GT(nCtxLog, CTX_SPLIT_MAX_NCTX_LOG, corruption);

    if (mode == CTX_SPLIT_MODE_PASSTHROUGH) {
        ZL_ERR_IF_NE(ptr, end, corruption);
        ZL_ERR_IF_NE(nbVariableSrcs, 1, corruption);
        const ZL_Input* const src0 = variableSrcs[0];
        ZL_ERR_IF_NULL(src0, corruption);

        ZL_Type const srcType = ZL_Input_type(src0);
        size_t const eltWidth = ZL_Input_eltWidth(src0);
        size_t const nbElts   = ZL_Input_numElts(src0);
        ZL_ERR_IF_NOT(ctx_isValidTypeWidth(srcType, eltWidth), corruption);

        size_t totalBytes = 0;
        ZL_ERR_IF(ZL_overflowMulST(nbElts, eltWidth, &totalBytes), corruption);

        ZL_Output* const out =
                ctx_createOutStream(dictx, nbElts, eltWidth, srcType);
        ZL_ERR_IF_NULL(out, allocation);
        if (totalBytes > 0) {
            memcpy(ZL_Output_ptr(out), ZL_Input_ptr(src0), totalBytes);
        }
        ZL_ERR_IF_ERR(ZL_Output_commit(out, nbElts));
        return ZL_returnSuccess();
    }

    /* mode == 0 (LINEAR) or mode == 1 (LOG2) */
    ZL_ERR_IF_GE(ptr, end, corruption);
    uint8_t const activeMask = *ptr++;
    uint32_t const K         = 1u << nCtxLog;
    uint32_t const maxCls    = K - 1u;

    ZL_ERR_IF_EQ(activeMask, 0, corruption);
    if (K < 8u) {
        ZL_ERR_IF_NE((unsigned)activeMask >> K, 0u, corruption);
    }

    size_t const expectedStreams = (size_t)ZL_popcount64((uint64_t)activeMask);
    ZL_ERR_IF_NE(nbVariableSrcs, expectedStreams, corruption);

    unsigned shift  = 0;
    uint64_t minVal = 0;
    if (mode == CTX_SPLIT_MODE_LINEAR) {
        ZL_ERR_IF_LT((size_t)(end - ptr), 2u, corruption);
        shift = (unsigned)(*ptr++);
        ZL_TRY_LET(uint64_t, decodedMin, ZL_varintDecode(&ptr, end));
        minVal = decodedMin;
    }
    ZL_ERR_IF_NE(ptr, end, corruption);

    const ZL_Input* const firstSrc = variableSrcs[0];
    ZL_ERR_IF_NULL(firstSrc, corruption);
    ZL_Type const srcType = ZL_Input_type(firstSrc);
    size_t const eltWidth = ZL_Input_eltWidth(firstSrc);
    ZL_ERR_IF_NOT(ctx_isValidTypeWidth(srcType, eltWidth), corruption);

    if (mode == CTX_SPLIT_MODE_LINEAR) {
        ZL_ERR_IF_GE(shift, (unsigned)(eltWidth * 8u), corruption);
        if (eltWidth < 8u) {
            ZL_ERR_IF_NE(minVal >> (eltWidth * 8u), 0u, corruption);
        }
    }

    const void* srcPtrs[CTX_SPLIT_MAX_K] = { NULL };
    size_t srcCounts[CTX_SPLIT_MAX_K]    = { 0 };
    size_t totalElts                     = 0;
    size_t sIdx                          = 0;

    for (uint32_t c = 0; c < K; c++) {
        if ((activeMask & (1u << c)) != 0) {
            const ZL_Input* const s = variableSrcs[sIdx++];
            ZL_ERR_IF_NULL(s, corruption);
            ZL_ERR_IF_NE(ZL_Input_type(s), srcType, corruption);
            ZL_ERR_IF_NE(ZL_Input_eltWidth(s), eltWidth, corruption);
            size_t const cnt = ZL_Input_numElts(s);
            ZL_ERR_IF_EQ(cnt, 0, corruption);
            ZL_ERR_IF(ZL_overflowAddST(totalElts, cnt, &totalElts), corruption);
            srcPtrs[c]   = ZL_Input_ptr(s);
            srcCounts[c] = cnt;
        }
    }
    ZL_ASSERT_EQ(sIdx, expectedStreams);

    size_t totalBytes = 0;
    ZL_ERR_IF(ZL_overflowMulST(totalElts, eltWidth, &totalBytes), corruption);

    ZL_Output* const out =
            ctx_createOutStream(dictx, totalElts, eltWidth, srcType);
    ZL_ERR_IF_NULL(out, allocation);

    void* const dst = ZL_Output_ptr(out);
    int ok          = 0;
    if (mode == CTX_SPLIT_MODE_LINEAR) {
        switch (eltWidth) {
            case 1:
                ok = ctx_gather_mode0_8(
                        (uint8_t*)dst, totalElts, srcPtrs, srcCounts, minVal,
                        shift, maxCls);
                break;
            case 2:
                ok = ctx_gather_mode0_16(
                        (uint16_t*)dst, totalElts, srcPtrs, srcCounts, minVal,
                        shift, maxCls);
                break;
            case 4:
                ok = ctx_gather_mode0_32(
                        (uint32_t*)dst, totalElts, srcPtrs, srcCounts, minVal,
                        shift, maxCls);
                break;
            case 8:
                ok = ctx_gather_mode0_64(
                        (uint64_t*)dst, totalElts, srcPtrs, srcCounts, minVal,
                        shift, maxCls);
                break;
        }
    } else {
        switch (eltWidth) {
            case 1:
                ok = ctx_gather_mode1_8(
                        (uint8_t*)dst, totalElts, srcPtrs, srcCounts, maxCls);
                break;
            case 2:
                ok = ctx_gather_mode1_16(
                        (uint16_t*)dst, totalElts, srcPtrs, srcCounts, maxCls);
                break;
            case 4:
                ok = ctx_gather_mode1_32(
                        (uint32_t*)dst, totalElts, srcPtrs, srcCounts, maxCls);
                break;
            case 8:
                ok = ctx_gather_mode1_64(
                        (uint64_t*)dst, totalElts, srcPtrs, srcCounts, maxCls);
                break;
        }
    }
    ZL_ERR_IF_NOT(ok, corruption);

    ZL_ERR_IF_ERR(ZL_Output_commit(out, totalElts));
    return ZL_returnSuccess();
}
