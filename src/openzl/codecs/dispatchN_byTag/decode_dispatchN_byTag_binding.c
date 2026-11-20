// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/dispatchN_byTag/decode_dispatchN_byTag_binding.h" // DI_dispatchN_byTag

#include <string.h> // memcmp

#include "openzl/codecs/dispatchN_byTag/decode_dispatchN_byTag_kernel.h" // ZL_dispatchN_byTag_decode
#include "openzl/codecs/range_pack/decode_range_pack_kernel.h" // rangePackDecode
#include "openzl/common/allocation.h"                          // ZL_zeroes
#include "openzl/common/assertion.h"
#include "openzl/common/logging.h"            // ZL_DLOG
#include "openzl/decompress/dictx.h"          // DI_getFrameFormatVersion
#include "openzl/shared/mem.h"                // ZL_read*
#include "openzl/shared/numeric_operations.h" // NUMOP_*
#include "openzl/zl_data.h"                   // ZS2_Data_*

static size_t DI_readNumAt(const void* ptr, size_t eltWidth)
{
    switch (eltWidth) {
        case 1:
            return ZL_read8(ptr);
        case 2:
            return ZL_read16(ptr);
        case 4:
            return ZL_read32(ptr);
        default:
            ZL_ASSERT_EQ(eltWidth, 8);
            return (size_t)ZL_read64(ptr);
    }
}

/* Detects the "interleave" layout:
 * all segments have the same non-zero size,
 * and tags cycle through 0, 1, ..., nbSrcs-1, 0, 1, ...
 * Checked directly on the packed inputs, without expanding them.
 * @return the segment size, or 0 if the layout is different. */
static size_t DI_dispatchN_roundRobinSegSize(
        const ZL_Input* tags,
        const ZL_Input* segSizes,
        size_t nbSegments,
        size_t nbSrcs)
{
    if (nbSegments == 0 || nbSrcs == 0) {
        return 0;
    }
    const uint8_t* const tagPtr = ZL_Input_ptr(tags);
    size_t const tagWidth       = ZL_Input_eltWidth(tags);
    size_t const nbHead         = nbSegments < nbSrcs ? nbSegments : nbSrcs;
    for (size_t n = 0; n < nbHead; n++) {
        if (DI_readNumAt(tagPtr + n * tagWidth, tagWidth) != n) {
            return 0;
        }
    }
    /* constant array <=> array equal to itself shifted by 1 element */
    const uint8_t* const sizePtr = ZL_Input_ptr(segSizes);
    size_t const sizeWidth       = ZL_Input_eltWidth(segSizes);
    size_t const segSize         = DI_readNumAt(sizePtr, sizeWidth);
    if (segSize == 0
        || memcmp(sizePtr, sizePtr + sizeWidth, (nbSegments - 1) * sizeWidth)
                != 0) {
        return 0;
    }
    /* periodic array <=> array equal to itself shifted by 1 period */
    if (memcmp(tagPtr,
               tagPtr + nbHead * tagWidth,
               (nbSegments - nbHead) * tagWidth)
        != 0) {
        return 0;
    }
    return segSize;
}

// DI_dispatchN_byTag design notes:
// - Generates a single stream of type ZL_Type_serial
//   from segments collected from multiple streams of type ZL_Type_serial.
//   Segments are defined by:
//   - origin : stream 1
//   - sizes : stream 2
ZL_Report DI_dispatchN_byTag(
        ZL_Decoder* dictx,
        const ZL_Input* inFixed[],
        size_t nbInFixed,
        const ZL_Input* inVariable[],
        size_t nbInVariable)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dictx);
    ZL_DLOG(BLOCK, "DI_dispatchN_byTag (%zu inputs to join)", nbInVariable);
    ZL_ASSERT_NN(dictx);
    ZL_ASSERT_EQ(nbInFixed, 2);
    ZL_ASSERT_NN(inFixed);

    // Validate inputs
    for (size_t n = 0; n < nbInFixed; n++) {
        // Guaranteed by engine
        ZL_ASSERT_NN(inFixed[n]);
        ZL_ASSERT_EQ(ZL_Input_type(inFixed[n]), ZL_Type_numeric);
    }
    const ZL_Input* tags     = inFixed[dnbt_tags];
    const ZL_Input* segSizes = inFixed[dnbt_segSizes];
    size_t const nbSegments  = ZL_Input_numElts(segSizes);
    ZL_ERR_IF_NE(ZL_Input_numElts(tags), nbSegments, corruption);

    if (DI_getFrameFormatVersion(dictx) < 20) {
        ZL_ERR_IF_GE(nbInVariable, 256, temporaryLibraryLimitation);
        ZL_ERR_IF_GT(ZL_Input_eltWidth(tags), 1, temporaryLibraryLimitation);
    } else {
        ZL_ERR_IF_GE(nbInVariable, 1 << 16, temporaryLibraryLimitation);
        ZL_ERR_IF_GT(ZL_Input_eltWidth(tags), 2, temporaryLibraryLimitation);
    }

    size_t total = 0;
    for (size_t n = 0; n < nbInVariable; n++) {
        // Must be validated
        ZL_ASSERT_NN(inVariable[n]);
        ZL_ERR_IF_NE(ZL_Input_type(inVariable[n]), ZL_Type_serial, corruption);
        total += ZL_Input_numElts(inVariable[n]);
    }
    ZL_Output* const out = ZL_Decoder_create1OutStream(dictx, total, 1);
    ZL_ERR_IF_NULL(out, allocation);

    size_t const rrSegSize = DI_dispatchN_roundRobinSegSize(
            tags, segSizes, nbSegments, nbInVariable);
    if (rrSegSize != 0) {
        const void** const rrSrcs = ZL_Decoder_getScratchSpace(
                dictx, sizeof(*rrSrcs) * nbInVariable);
        ZL_ERR_IF_NULL(rrSrcs, allocation);
        size_t const nbRounds = nbSegments / nbInVariable;
        size_t const nbTail   = nbSegments % nbInVariable;
        for (size_t n = 0; n < nbInVariable; n++) {
            size_t const nbSegs = nbRounds + (n < nbTail);
            /* written as a division to avoid overflow */
            if (ZL_Input_numElts(inVariable[n]) % rrSegSize != 0
                || ZL_Input_numElts(inVariable[n]) / rrSegSize != nbSegs) {
                ZL_ERR(corruption,
                       "segment sizes incorrect : invalid total size for stream %zu",
                       n);
            }
            rrSrcs[n] = ZL_Input_ptr(inVariable[n]);
        }
        ZL_dispatchN_byTag_decodeRoundRobin(
                ZL_Output_ptr(out),
                rrSrcs,
                nbInVariable,
                rrSegSize,
                nbSegments);
        ZL_ERR_IF_ERR(ZL_Output_commit(out, total));
        return ZL_returnSuccess();
    }

    // Note : reserving scratch buffers should be a property offered by DICtx*.
    // Currently it doesn't exist, so allocate directly instead.
    // This should be updated once scratch buffer support is implemented.

    const void** const srcs =
            ZL_Decoder_getScratchSpace(dictx, sizeof(*srcs) * nbInVariable);
    size_t* const srcSizes = ZL_Decoder_getScratchSpace(
            dictx,
            sizeof(*srcSizes) * nbInVariable); // must be init to zero
    ZL_zeroes(srcSizes, sizeof(*srcSizes) * nbInVariable);
    size_t* const segmentSizes = ZL_Decoder_getScratchSpace(
            dictx, sizeof(*segmentSizes) * nbSegments);
    uint16_t* const bufIndex =
            ZL_Decoder_getScratchSpace(dictx, sizeof(*bufIndex) * nbSegments);

    if (!srcs || !srcSizes || !segmentSizes || !bufIndex) {
        ZL_ERR(allocation);
    }

    /* prepare arrays for raw transform */
    for (size_t n = 0; n < nbInVariable; n++) {
        srcs[n] = ZL_Input_ptr(inVariable[n]);
    }
    rangePackDecode(
            segmentSizes,
            sizeof(*segmentSizes),
            ZL_Input_ptr(segSizes),
            ZL_Input_eltWidth(segSizes),
            nbSegments,
            0);
    rangePackDecode(
            bufIndex,
            sizeof(*bufIndex),
            ZL_Input_ptr(tags),
            ZL_Input_eltWidth(tags),
            nbSegments,
            0);

    /* Check validity of tags */
    if (!NUMOP_underLimitU16(bufIndex, nbSegments, (unsigned)nbInVariable)) {
        ZL_ERR(corruption,
               "vector of tags incorrect : some value(s) > nb srcs");
    }

    /* Check validity of segment sizes */
    for (size_t n = 0; n < nbSegments; n++) {
        srcSizes[bufIndex[n]] += segmentSizes[n];
    }
    for (size_t n = 0; n < nbInVariable; n++) {
        if (srcSizes[n] != ZL_Input_numElts(inVariable[n])) {
            ZL_ERR(corruption,
                   "segment sizes incorrect : invalid total size for stream %zu",
                   n);
        }
    }

    size_t const r = ZL_dispatchN_byTag_decode(
            ZL_Output_ptr(out),
            total,
            srcs,
            nbInVariable,
            segmentSizes,
            bufIndex,
            nbSegments);

    ZL_ASSERT_EQ(r, total);

    ZL_ERR_IF_ERR(ZL_Output_commit(out, total));

    return ZL_returnSuccess();
}
