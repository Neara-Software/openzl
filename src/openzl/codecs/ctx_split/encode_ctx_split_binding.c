// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/ctx_split/encode_ctx_split_binding.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "openzl/codecs/ctx_split/common_ctx_split.h"
#include "openzl/codecs/zl_ctx_split.h"
#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
#include "openzl/shared/varint.h"

ZL_FORCE_INLINE uint8_t ctx_sym_log2(uint64_t v)
{
    if (v < 4u) {
        return (uint8_t)v;
    }
    unsigned const hb = ctx_bitlen64(v) - 1u; /* >= 2 */
    unsigned const s  = (hb << 2) | (unsigned)((v >> (hb - 2u)) & 3u);
    return s > 255u ? (uint8_t)255u : (uint8_t)s;
}

#define CTX_DEFINE_ENCODER_OPS(W, T)                                           \
    static void ctx_minmax_##W(                                                \
            const T* restrict src,                                             \
            size_t n,                                                          \
            uint64_t* outMin,                                                  \
            uint64_t* outMax)                                                  \
    {                                                                          \
        T mn = src[0];                                                         \
        T mx = src[0];                                                         \
        for (size_t i = 1; i < n; i++) {                                       \
            T const v = src[i];                                                \
            if (v < mn)                                                        \
                mn = v;                                                        \
            if (v > mx)                                                        \
                mx = v;                                                        \
        }                                                                      \
        *outMin = (uint64_t)mn;                                                \
        *outMax = (uint64_t)mx;                                                \
    }                                                                          \
                                                                               \
    static void ctx_accum_block_##W(                                           \
            const T* restrict src,                                             \
            size_t startIdx,                                                   \
            size_t len,                                                        \
            uint64_t minVal,                                                   \
            unsigned shift,                                                    \
            uint32_t maxCls,                                                   \
            unsigned symShift,                                                 \
            int exact256,                                                      \
            uint32_t h0[CTX_SPLIT_MAX_K][256],                                 \
            uint32_t h1[CTX_SPLIT_MAX_K][256])                                 \
    {                                                                          \
        uint64_t prev = (startIdx == 0) ? 0u : (uint64_t)src[startIdx - 1];    \
        const T* const p = src + startIdx;                                     \
        if (exact256) {                                                        \
            for (size_t i = 0; i < len; i++) {                                 \
                uint64_t const v  = (uint64_t)p[i];                            \
                uint32_t const c0 = ctx_cls_mode0(prev, minVal, shift, maxCls);\
                uint32_t const c1 = ctx_cls_mode1(prev, maxCls);               \
                uint8_t const sym = (uint8_t)(v - minVal);                     \
                h0[c0][sym]++;                                                 \
                h1[c1][sym]++;                                                 \
                prev = v;                                                      \
            }                                                                  \
        } else {                                                               \
            for (size_t i = 0; i < len; i++) {                                 \
                uint64_t const v  = (uint64_t)p[i];                            \
                uint32_t const c0 = ctx_cls_mode0(prev, minVal, shift, maxCls);\
                uint32_t const c1 = ctx_cls_mode1(prev, maxCls);               \
                uint8_t const s0  = (uint8_t)((v - minVal) >> symShift);       \
                uint8_t const s1  = ctx_sym_log2(v);                           \
                h0[c0][s0]++;                                                  \
                h1[c1][s1]++;                                                  \
                prev = v;                                                      \
            }                                                                  \
        }                                                                      \
    }                                                                          \
                                                                               \
    static void ctx_count_mode0_##W(                                           \
            const T* restrict src,                                             \
            size_t n,                                                          \
            uint64_t minVal,                                                   \
            unsigned shift,                                                    \
            uint32_t maxCls,                                                   \
            size_t counts[CTX_SPLIT_MAX_K])                                    \
    {                                                                          \
        uint64_t prev = 0;                                                     \
        for (size_t i = 0; i < n; i++) {                                       \
            uint64_t const v = (uint64_t)src[i];                               \
            counts[ctx_cls_mode0(prev, minVal, shift, maxCls)]++;              \
            prev = v;                                                          \
        }                                                                      \
    }                                                                          \
                                                                               \
    static void ctx_count_mode1_##W(                                           \
            const T* restrict src,                                             \
            size_t n,                                                          \
            uint32_t maxCls,                                                   \
            size_t counts[CTX_SPLIT_MAX_K])                                    \
    {                                                                          \
        uint64_t prev = 0;                                                     \
        for (size_t i = 0; i < n; i++) {                                       \
            uint64_t const v = (uint64_t)src[i];                               \
            counts[ctx_cls_mode1(prev, maxCls)]++;                             \
            prev = v;                                                          \
        }                                                                      \
    }                                                                          \
                                                                               \
    static void ctx_scatter_mode0_##W(                                         \
            const T* restrict src,                                             \
            size_t n,                                                          \
            uint64_t minVal,                                                   \
            unsigned shift,                                                    \
            uint32_t maxCls,                                                   \
            void* const dstPtrs[CTX_SPLIT_MAX_K])                              \
    {                                                                          \
        T* cur[CTX_SPLIT_MAX_K];                                               \
        for (size_t c = 0; c < CTX_SPLIT_MAX_K; c++) {                         \
            cur[c] = (T*)dstPtrs[c];                                           \
        }                                                                      \
        uint64_t prev = 0;                                                     \
        for (size_t i = 0; i < n; i++) {                                       \
            T const val      = src[i];                                         \
            uint32_t const c = ctx_cls_mode0(prev, minVal, shift, maxCls);     \
            *cur[c]++        = val;                                            \
            prev             = (uint64_t)val;                                  \
        }                                                                      \
    }                                                                          \
                                                                               \
    static void ctx_scatter_mode1_##W(                                         \
            const T* restrict src,                                             \
            size_t n,                                                          \
            uint32_t maxCls,                                                   \
            void* const dstPtrs[CTX_SPLIT_MAX_K])                              \
    {                                                                          \
        T* cur[CTX_SPLIT_MAX_K];                                               \
        for (size_t c = 0; c < CTX_SPLIT_MAX_K; c++) {                         \
            cur[c] = (T*)dstPtrs[c];                                           \
        }                                                                      \
        uint64_t prev = 0;                                                     \
        for (size_t i = 0; i < n; i++) {                                       \
            T const val      = src[i];                                         \
            uint32_t const c = ctx_cls_mode1(prev, maxCls);                    \
            *cur[c]++        = val;                                            \
            prev             = (uint64_t)val;                                  \
        }                                                                      \
    }

CTX_DEFINE_ENCODER_OPS(8, uint8_t)
CTX_DEFINE_ENCODER_OPS(16, uint16_t)
CTX_DEFINE_ENCODER_OPS(32, uint32_t)
CTX_DEFINE_ENCODER_OPS(64, uint64_t)

static double ctx_hist_entropy_bits(
        const uint32_t hist[256],
        uint32_t* outTotal,
        size_t* outNonZero)
{
    uint32_t total  = 0;
    size_t nonZero  = 0;
    double sumCLogC = 0.0;
    for (size_t s = 0; s < 256; s++) {
        uint32_t const cnt = hist[s];
        if (cnt > 0) {
            total += cnt;
            nonZero++;
            if (cnt > 1) {
                sumCLogC += (double)cnt * log2((double)cnt);
            }
        }
    }
    *outTotal   = total;
    *outNonZero = nonZero;
    if (total <= 1) {
        return 0.0;
    }
    double const bits = (double)total * log2((double)total) - sumCLogC;
    return bits > 0.0 ? bits : 0.0;
}

static double ctx_score_mode_gain_bytes(
        const uint32_t h[CTX_SPLIT_MAX_K][256],
        uint32_t K,
        size_t nbElts,
        size_t sampleN,
        size_t classTotalsOut[CTX_SPLIT_MAX_K],
        size_t* outActiveK)
{
    uint32_t baseHist[256];
    memset(baseHist, 0, sizeof(baseHist));

    double splitBits   = 0.0;
    size_t activeK     = 0;
    size_t splitNonZero = 0;
    for (uint32_t c = 0; c < K; c++) {
        uint32_t nc = 0;
        size_t nz   = 0;
        splitBits += ctx_hist_entropy_bits(h[c], &nc, &nz);
        classTotalsOut[c] = (size_t)nc;
        if (nc > 0) {
            activeK++;
            splitNonZero += nz;
            for (size_t s = 0; s < 256; s++) {
                baseHist[s] += h[c][s];
            }
        }
    }
    *outActiveK = activeK;
    if (activeK <= 1) {
        return -(double)CTX_SPLIT_STREAM_OVH_BYTES;
    }

    uint32_t baseTotal    = 0;
    size_t baseNonZero    = 0;
    double const baseBits =
            ctx_hist_entropy_bits(baseHist, &baseTotal, &baseNonZero);
    ZL_ASSERT_EQ((size_t)baseTotal, sampleN);

    /* Subtract finite-sample plug-in entropy bias before scaling up from sampleN */
    double const biasBits = (splitNonZero > baseNonZero)
            ? (double)(splitNonZero - baseNonZero)
            : 0.0;
    double const netSampleBits = (baseBits - splitBits) - biasBits;
    if (netSampleBits <= 0.0) {
        return -(double)(CTX_SPLIT_STREAM_OVH_BYTES * activeK);
    }

    double const scale     = (double)nbElts / (double)sampleN;
    double const savedBits = netSampleBits * scale;
    return (savedBits / 8.0) - (double)(CTX_SPLIT_STREAM_OVH_BYTES * activeK);
}

static ZL_Report ctx_emit_passthrough(
        ZL_Encoder* eictx,
        const void* src,
        size_t nbElts,
        size_t eltWidth,
        unsigned nCtxLog)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(eictx);
    uint8_t const flags =
            (uint8_t)(((unsigned)CTX_SPLIT_MODE_PASSTHROUGH << 4) | nCtxLog);
    ZL_Encoder_sendCodecHeader(eictx, &flags, 1);

    ZL_Output* const out =
            ZL_Encoder_createTypedStream(eictx, 0, nbElts, eltWidth);
    ZL_ERR_IF_NULL(out, allocation);
    if (nbElts > 0) {
        memcpy(ZL_Output_ptr(out), src, nbElts * eltWidth);
    }
    ZL_ERR_IF_ERR(ZL_Output_commit(out, nbElts));
    return ZL_returnSuccess();
}

ZL_Report
EI_ctx_split(ZL_Encoder* eictx, const ZL_Input* ins[], size_t nbIns)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(eictx);
    ZL_ASSERT_NN(eictx);
    ZL_ASSERT_EQ(nbIns, 1);
    (void)nbIns;

    const ZL_Input* const in = ins[0];
    ZL_ASSERT_NN(in);
    ZL_Type const inType  = ZL_Input_type(in);
    size_t const eltWidth = ZL_Input_eltWidth(in);
    size_t const nbElts   = ZL_Input_numElts(in);
    const void* const src = ZL_Input_ptr(in);

    ZL_ASSERT(inType == ZL_Type_serial || inType == ZL_Type_numeric);
    if (inType == ZL_Type_serial) {
        ZL_ERR_IF_NE(eltWidth, 1, node_unexpected_input_type);
    } else {
        ZL_ERR_IF_NOT(
                eltWidth == 1 || eltWidth == 2 || eltWidth == 4
                        || eltWidth == 8,
                node_unexpected_input_type);
    }

    unsigned nCtxLog = CTX_SPLIT_DEFAULT_NCTX_LOG;
    ZL_IntParam const lp =
            ZL_Encoder_getLocalIntParam(eictx, ZL_CTX_SPLIT_NCTX_LOG_PID);
    if (lp.paramId == ZL_CTX_SPLIT_NCTX_LOG_PID) {
        if (lp.paramValue < CTX_SPLIT_MIN_NCTX_LOG) {
            nCtxLog = CTX_SPLIT_MIN_NCTX_LOG;
        } else if (lp.paramValue > CTX_SPLIT_MAX_NCTX_LOG) {
            nCtxLog = CTX_SPLIT_MAX_NCTX_LOG;
        } else {
            nCtxLog = (unsigned)lp.paramValue;
        }
    }

    if (nbElts < CTX_SPLIT_MIN_ELTS) {
        return ctx_emit_passthrough(eictx, src, nbElts, eltWidth, nCtxLog);
    }

    uint64_t minVal = 0;
    uint64_t maxVal = 0;
    switch (eltWidth) {
        case 1:
            ctx_minmax_8((const uint8_t*)src, nbElts, &minVal, &maxVal);
            break;
        case 2:
            ctx_minmax_16((const uint16_t*)src, nbElts, &minVal, &maxVal);
            break;
        case 4:
            ctx_minmax_32((const uint32_t*)src, nbElts, &minVal, &maxVal);
            break;
        case 8:
            ctx_minmax_64((const uint64_t*)src, nbElts, &minVal, &maxVal);
            break;
        default:
            ZL_ERR(node_unexpected_input_type);
    }

    if (minVal == maxVal) {
        return ctx_emit_passthrough(eictx, src, nbElts, eltWidth, nCtxLog);
    }

    uint32_t const K       = 1u << nCtxLog;
    uint32_t const maxCls  = K - 1u;
    uint64_t const range   = maxVal - minVal;
    unsigned const rangeBl = ctx_bitlen64(range);
    unsigned const shift   = (rangeBl > nCtxLog) ? (rangeBl - nCtxLog) : 0u;
    int const exact256     = (range <= 255u) ? 1 : 0;
    unsigned const symSh   = (rangeBl > 8u) ? (rangeBl - 8u) : 0u;

    uint32_t h0[CTX_SPLIT_MAX_K][256];
    uint32_t h1[CTX_SPLIT_MAX_K][256];
    memset(h0, 0, sizeof(h0));
    memset(h1, 0, sizeof(h1));

    size_t sampleN     = 0;
    int const fullScan = (nbElts <= CTX_SPLIT_FULL_SCAN_LIMIT) ? 1 : 0;
    if (fullScan) {
        sampleN = nbElts;
        switch (eltWidth) {
            case 1:
                ctx_accum_block_8(
                        (const uint8_t*)src, 0, nbElts, minVal, shift, maxCls,
                        symSh, exact256, h0, h1);
                break;
            case 2:
                ctx_accum_block_16(
                        (const uint16_t*)src, 0, nbElts, minVal, shift, maxCls,
                        symSh, exact256, h0, h1);
                break;
            case 4:
                ctx_accum_block_32(
                        (const uint32_t*)src, 0, nbElts, minVal, shift, maxCls,
                        symSh, exact256, h0, h1);
                break;
            case 8:
                ctx_accum_block_64(
                        (const uint64_t*)src, 0, nbElts, minVal, shift, maxCls,
                        symSh, exact256, h0, h1);
                break;
        }
    } else {
        size_t const span = nbElts - CTX_SPLIT_SAMPLE_BLOCK_LEN;
        for (size_t b = 0; b < CTX_SPLIT_SAMPLE_BLOCKS; b++) {
            size_t const startIdx =
                    (b * span) / (CTX_SPLIT_SAMPLE_BLOCKS - 1u);
            switch (eltWidth) {
                case 1:
                    ctx_accum_block_8(
                            (const uint8_t*)src, startIdx,
                            CTX_SPLIT_SAMPLE_BLOCK_LEN, minVal, shift, maxCls,
                            symSh, exact256, h0, h1);
                    break;
                case 2:
                    ctx_accum_block_16(
                            (const uint16_t*)src, startIdx,
                            CTX_SPLIT_SAMPLE_BLOCK_LEN, minVal, shift, maxCls,
                            symSh, exact256, h0, h1);
                    break;
                case 4:
                    ctx_accum_block_32(
                            (const uint32_t*)src, startIdx,
                            CTX_SPLIT_SAMPLE_BLOCK_LEN, minVal, shift, maxCls,
                            symSh, exact256, h0, h1);
                    break;
                case 8:
                    ctx_accum_block_64(
                            (const uint64_t*)src, startIdx,
                            CTX_SPLIT_SAMPLE_BLOCK_LEN, minVal, shift, maxCls,
                            symSh, exact256, h0, h1);
                    break;
            }
            sampleN += CTX_SPLIT_SAMPLE_BLOCK_LEN;
        }
    }

    size_t counts0[CTX_SPLIT_MAX_K] = { 0 };
    size_t counts1[CTX_SPLIT_MAX_K] = { 0 };
    size_t active0 = 0;
    size_t active1 = 0;
    double const gain0 = ctx_score_mode_gain_bytes(
            h0, K, nbElts, sampleN, counts0, &active0);
    double const gain1 = ctx_score_mode_gain_bytes(
            h1, K, nbElts, sampleN, counts1, &active1);

    unsigned const mode =
            (gain0 >= gain1) ? CTX_SPLIT_MODE_LINEAR : CTX_SPLIT_MODE_LOG2;
    double const bestGain = (gain0 >= gain1) ? gain0 : gain1;
    if (bestGain < (double)CTX_SPLIT_MIN_GAIN_BYTES) {
        return ctx_emit_passthrough(eictx, src, nbElts, eltWidth, nCtxLog);
    }

    size_t classCounts[CTX_SPLIT_MAX_K];
    memset(classCounts, 0, sizeof(classCounts));
    if (fullScan) {
        const size_t* const chosen =
                (mode == CTX_SPLIT_MODE_LINEAR) ? counts0 : counts1;
        for (uint32_t c = 0; c < K; c++) {
            classCounts[c] = chosen[c];
        }
    } else if (mode == CTX_SPLIT_MODE_LINEAR) {
        switch (eltWidth) {
            case 1:
                ctx_count_mode0_8(
                        (const uint8_t*)src, nbElts, minVal, shift, maxCls,
                        classCounts);
                break;
            case 2:
                ctx_count_mode0_16(
                        (const uint16_t*)src, nbElts, minVal, shift, maxCls,
                        classCounts);
                break;
            case 4:
                ctx_count_mode0_32(
                        (const uint32_t*)src, nbElts, minVal, shift, maxCls,
                        classCounts);
                break;
            case 8:
                ctx_count_mode0_64(
                        (const uint64_t*)src, nbElts, minVal, shift, maxCls,
                        classCounts);
                break;
        }
    } else {
        switch (eltWidth) {
            case 1:
                ctx_count_mode1_8(
                        (const uint8_t*)src, nbElts, maxCls, classCounts);
                break;
            case 2:
                ctx_count_mode1_16(
                        (const uint16_t*)src, nbElts, maxCls, classCounts);
                break;
            case 4:
                ctx_count_mode1_32(
                        (const uint32_t*)src, nbElts, maxCls, classCounts);
                break;
            case 8:
                ctx_count_mode1_64(
                        (const uint64_t*)src, nbElts, maxCls, classCounts);
                break;
        }
    }

    uint8_t activeMask = 0;
    size_t activeK     = 0;
    for (uint32_t c = 0; c < K; c++) {
        if (classCounts[c] > 0) {
            activeMask |= (uint8_t)(1u << c);
            activeK++;
        }
    }
    if (activeK <= 1) {
        return ctx_emit_passthrough(eictx, src, nbElts, eltWidth, nCtxLog);
    }

    uint8_t hdr[3 + ZL_VARINT_LENGTH_64];
    size_t hdrLen = 0;
    hdr[hdrLen++] = (uint8_t)((mode << 4) | nCtxLog);
    hdr[hdrLen++] = activeMask;
    if (mode == CTX_SPLIT_MODE_LINEAR) {
        hdr[hdrLen++] = (uint8_t)shift;
        hdrLen += ZL_varintEncode64Fast(minVal, hdr + hdrLen);
    }
    ZL_Encoder_sendCodecHeader(eictx, hdr, hdrLen);

    ZL_Output* outs[CTX_SPLIT_MAX_K] = { NULL };
    void* dstPtrs[CTX_SPLIT_MAX_K]   = { NULL };
    for (uint32_t c = 0; c < K; c++) {
        if (classCounts[c] > 0) {
            outs[c] = ZL_Encoder_createTypedStream(
                    eictx, 0, classCounts[c], eltWidth);
            ZL_ERR_IF_NULL(outs[c], allocation);
            dstPtrs[c] = ZL_Output_ptr(outs[c]);
        }
    }

    if (mode == CTX_SPLIT_MODE_LINEAR) {
        switch (eltWidth) {
            case 1:
                ctx_scatter_mode0_8(
                        (const uint8_t*)src, nbElts, minVal, shift, maxCls,
                        dstPtrs);
                break;
            case 2:
                ctx_scatter_mode0_16(
                        (const uint16_t*)src, nbElts, minVal, shift, maxCls,
                        dstPtrs);
                break;
            case 4:
                ctx_scatter_mode0_32(
                        (const uint32_t*)src, nbElts, minVal, shift, maxCls,
                        dstPtrs);
                break;
            case 8:
                ctx_scatter_mode0_64(
                        (const uint64_t*)src, nbElts, minVal, shift, maxCls,
                        dstPtrs);
                break;
        }
    } else {
        switch (eltWidth) {
            case 1:
                ctx_scatter_mode1_8(
                        (const uint8_t*)src, nbElts, maxCls, dstPtrs);
                break;
            case 2:
                ctx_scatter_mode1_16(
                        (const uint16_t*)src, nbElts, maxCls, dstPtrs);
                break;
            case 4:
                ctx_scatter_mode1_32(
                        (const uint32_t*)src, nbElts, maxCls, dstPtrs);
                break;
            case 8:
                ctx_scatter_mode1_64(
                        (const uint64_t*)src, nbElts, maxCls, dstPtrs);
                break;
        }
    }

    for (uint32_t c = 0; c < K; c++) {
        if (outs[c] != NULL) {
            ZL_ERR_IF_ERR(ZL_Output_commit(outs[c], classCounts[c]));
        }
    }

    return ZL_returnSuccess();
}
