// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/compress/selectors/selector_brute_force.h"

#include "openzl/codecs/zl_brute_force_selector.h"
#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
#include "openzl/compress/implicit_conversion.h"
#include "openzl/zl_compress.h"
#include "openzl/zl_compressor.h"
#include "openzl/zl_selector.h"


// Serial inputs are sliced at multiples of 3 * 4096 bytes so that records of
// width 1, 2, 3, 4, 6, 8, 12, 16 or 24 bytes stay aligned.
#define ZL_BRUTE_FORCE_SERIAL_ALIGN ((size_t)3 * 4096)

/// Tries every compatible successor on a slice from the middle of the input
/// and returns a list marking all but the @p topk smallest as skipped, or NULL
/// when the probe cannot be run (the caller then tries every successor).
static bool* probeSkipList(
        const ZL_Selector* selCtx,
        const ZL_Input* in,
        const ZL_GraphID* graphs,
        size_t nbGraphs,
        size_t probe,
        size_t topk)
{
    const ZL_Type type   = ZL_Input_type(in);
    const size_t width   = type == ZL_Type_serial ? ZL_BRUTE_FORCE_SERIAL_ALIGN
                                                  : ZL_Input_eltWidth(in);
    const size_t nbUnits = ZL_Input_contentSize(in) / width;
    const size_t unitsIn = (probe + width - 1) / width;
    if (unitsIn == 0 || unitsIn * 2 > nbUnits) {
        return NULL;
    }
    const char* start =
            (const char*)ZL_Input_ptr(in) + ((nbUnits - unitsIn) / 2) * width;
    ZL_TypedRef* slice = NULL;
    if (type == ZL_Type_serial) {
        slice = ZL_TypedRef_createSerial(start, unitsIn * width);
    } else if (type == ZL_Type_struct) {
        slice = ZL_TypedRef_createStruct(start, width, unitsIn);
    } else {
        slice = ZL_TypedRef_createNumeric(start, width, unitsIn);
    }
    size_t* probeSizes = (size_t*)ZL_Selector_getScratchSpace(
            selCtx, nbGraphs * sizeof(size_t));
    bool* skip =
            (bool*)ZL_Selector_getScratchSpace(selCtx, nbGraphs * sizeof(bool));
    if (slice == NULL || probeSizes == NULL || skip == NULL) {
        ZL_TypedRef_free(slice);
        return NULL;
    }
    for (size_t i = 0; i < nbGraphs; ++i) {
        probeSizes[i] = (size_t)-1;
        if (!ICONV_isCompatible(
                    type, ZL_Selector_getInput0MaskForGraph(selCtx, graphs[i]))) {
            continue;
        }
        ZL_GraphReport gr = ZL_Selector_tryGraph(selCtx, slice, graphs[i]);
        if (!ZL_isError(gr.finalCompressedSize)) {
            probeSizes[i] = ZL_validResult(gr.finalCompressedSize);
        }
    }
    ZL_TypedRef_free(slice);
    // A successor is kept when fewer than topk others probe strictly smaller
    // (ties broken by index).
    for (size_t i = 0; i < nbGraphs; ++i) {
        size_t better = 0;
        for (size_t j = 0; j < nbGraphs; ++j) {
            if (probeSizes[j] < probeSizes[i]
                || (probeSizes[j] == probeSizes[i] && j < i)) {
                ++better;
            }
        }
        skip[i] = probeSizes[i] == (size_t)-1 || better >= topk;
    }
    return skip;
}

ZL_GraphID SI_selector_brute_force(
        const ZL_Selector* selCtx,
        const ZL_Input* inputStream,
        const ZL_GraphID* customGraphs,
        size_t nbCustomGraphs)
{
    ZL_ASSERT_NN(selCtx);
    ZL_ASSERT_NN(inputStream);
    ZL_ASSERT(nbCustomGraphs == 0 || customGraphs != NULL);

    const ZL_Type inputType = ZL_Input_type(inputStream);

    // brute force all graphs
    size_t bestSize = ZL_Input_contentSize(inputStream);
    if (inputType == ZL_Type_string) {
        bestSize += ZL_Input_numElts(inputStream) * sizeof(uint32_t);
    }
    // Optional int local params, read from the graph that runs this selector
    // (for ACE that is the zl.ace wrapper graph). Encoder-only choices; the
    // frame format is unchanged.
    const ZL_IntParam slackParam =
            ZL_Selector_getLocalIntParam(selCtx, ZL_BRUTE_FORCE_SLACK_PARAM);
    const int slack = slackParam.paramId == ZL_LP_INVALID_PARAMID
            ? 0
            : slackParam.paramValue;
    const ZL_IntParam lambdaParam =
            ZL_Selector_getLocalIntParam(selCtx, ZL_BRUTE_FORCE_LAMBDA_PARAM);
    const int lambda = lambdaParam.paramId == ZL_LP_INVALID_PARAMID
            ? 0
            : lambdaParam.paramValue;
    size_t* sizes = NULL;
    if ((slack > 0 || lambda > 0) && nbCustomGraphs > 0) {
        sizes = (size_t*)ZL_Selector_getScratchSpace(
                selCtx, nbCustomGraphs * sizeof(size_t));
    }
    // Optional probe pass: rank the successors on a slice of the input and
    // fully try only the best ones.
    const ZL_IntParam probeParam =
            ZL_Selector_getLocalIntParam(selCtx, ZL_BRUTE_FORCE_PROBE_PARAM);
    const size_t probe = probeParam.paramId == ZL_LP_INVALID_PARAMID
            ? 0
            : (size_t)probeParam.paramValue;
    const ZL_IntParam topkParam = ZL_Selector_getLocalIntParam(
            selCtx, ZL_BRUTE_FORCE_PROBE_TOPK_PARAM);
    const size_t topk = topkParam.paramId == ZL_LP_INVALID_PARAMID
            ? 1
            : (size_t)topkParam.paramValue;
    bool* skip = NULL;
    if (probe > 0 && nbCustomGraphs > topk && inputType != ZL_Type_string
        && ZL_Input_contentSize(inputStream) >= 4 * probe) {
        skip = probeSkipList(selCtx, inputStream, customGraphs, nbCustomGraphs, probe, topk);
    }
    int64_t bestIdx = -1;
    for (size_t i = 0; i < nbCustomGraphs; ++i) {
        if (sizes != NULL) {
            sizes[i] = (size_t)-1;
        }
        if (skip != NULL && skip[i]) {
            continue;
        }
        // Skip successors that can't be fed the input type, since the
        // successor list is user-provided and isn't validated at registration.
        if (!ICONV_isCompatible(
                    inputType,
                    ZL_Selector_getInput0MaskForGraph(
                            selCtx, customGraphs[i]))) {
            continue;
        }
        ZL_GraphReport gr =
                ZL_Selector_tryGraph(selCtx, inputStream, customGraphs[i]);
        if (ZL_isError(gr.finalCompressedSize)) {
            continue;
        }
        size_t currSize = ZL_validResult(gr.finalCompressedSize);
        if (sizes != NULL) {
            sizes[i] = currSize;
        }
        if (currSize < bestSize) {
            bestSize = currSize;
            bestIdx  = (int64_t)i;
        }
    }
    if (bestIdx == -1) {
        return ZL_GRAPH_STORE;
    }
    if (sizes == NULL) {
        return customGraphs[bestIdx];
    }
    if (lambda > 0) {
        // argmin size + lambda * predicted decode time, over the successors
        // that compress the input.
        const size_t srcSize = ZL_Input_contentSize(inputStream);
        const double kib     = (double)srcSize / 1024.0;
        double bestScore     = 0;
        int64_t pick         = -1;
        for (size_t i = 0; i < nbCustomGraphs; ++i) {
            if (sizes[i] == (size_t)-1
                || (sizes[i] >= srcSize && (int64_t)i != bestIdx)) {
                continue;
            }
            const ZL_IntParam rate = ZL_Selector_getLocalIntParam(
                    selCtx, ZL_BRUTE_FORCE_RATE_PARAM_BASE + (int)i);
            const double ms = rate.paramId == ZL_LP_INVALID_PARAMID
                    ? 0
                    : (double)rate.paramValue * kib * 1e-6;
            const double score = (double)sizes[i] + (double)lambda * ms;
            if (pick == -1 || score < bestScore) {
                bestScore = score;
                pick      = (int64_t)i;
            }
        }
        return customGraphs[pick];
    }
    // Slack: the successors are listed fastest-decoding first, so take the
    // first one whose size is within (1 + slack * 1e-4) of the smallest.
    const double limit = (double)bestSize * (1.0 + slack * 1e-4);
    for (int64_t i = 0; i < bestIdx; ++i) {
        if ((double)sizes[i] <= limit) {
            return customGraphs[i];
        }
    }
    return customGraphs[bestIdx];
}

ZL_RESULT_OF(ZL_GraphID)
ZL_Compressor_buildBruteForceSelectorGraph(
        ZL_Compressor* cgraph,
        const ZL_GraphID* successors,
        size_t numSuccessors)
{
    ZL_RESULT_DECLARE_SCOPE(ZL_GraphID, cgraph);
    ZL_ERR_IF_EQ(
            numSuccessors,
            0,
            parameter_invalid,
            "brute force selector requires at least one successor");
    const ZL_GraphParameters params = {
        .customGraphs   = successors,
        .nbCustomGraphs = numSuccessors,
    };
    return ZL_Compressor_parameterizeGraph(
            cgraph, ZL_GRAPH_BRUTE_FORCE, &params);
}
