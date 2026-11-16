// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/compress/codec_output_cache.h"

#include <string.h>

#include "openzl/codecs/conversion/encode_conversion_binding.h"
#include "openzl/codecs/entropy/encode_entropy_binding.h"
#include "openzl/codecs/zl_dispatch.h"
#include "openzl/common/allocation.h"
#include "openzl/common/assertion.h"
#include "openzl/common/map.h"
#include "openzl/common/stream.h"
#include "openzl/compress/cctx.h"
#include "openzl/compress/cgraph.h"
#include "openzl/compress/cnode.h"
#include "openzl/compress/enc_interface.h"
#include "openzl/compress/graph_registry.h"
#include "openzl/compress/nodemgr.h"
#include "openzl/compress/private_nodes.h"
#include "openzl/compress/selectors/selector_brute_force.h"
#include "openzl/dict/dict_constants.h"
#define HUF_STATIC_LINKING_ONLY
#include "openzl/fse/huf.h"
#include "openzl/shared/histogram.h"
#include "openzl/shared/overflow.h"
#include "openzl/shared/xxhash.h"
#include "openzl/zl_reflection.h"

#define CODEC_CACHE_DEFAULT_MAX_BYTES ((size_t)256 << 20)
#define CODEC_CACHE_MAX_ENTRIES (1u << 24)
#define CODEC_CACHE_TRYGRAPH_MAX_ENTRIES (1u << 20)

typedef struct {
    ZL_Type type;
    size_t eltWidth;
    size_t numElts;
    size_t contentSize;
    const void* content;
    size_t nbIntMetadata;
    const Stream_IntMetadata* intMetadata;
} CodecCache_Input;

typedef struct {
    uint64_t contentHash64;
    ZL_IDType standardNodeID;
    uint32_t formatVersion;
    int32_t compressionLevel;
    int32_t decompressionLevel;
    CodecCache_Input input;
    size_t localParamsSize;
    const void* localParams;
} CodecCache_Key;

typedef struct {
    uint64_t hash64;
    size_t size;
    const void* data;
} CodecCache_BlobKey;

typedef struct {
    uint64_t combinedHash64;
    const ZL_Compressor* cgraph;
    ZL_IDType graphID;
    uint32_t formatVersion;
    int32_t compressionLevel;
    int32_t decompressionLevel;
    int32_t permissiveCompression;
    int32_t compressedChecksum;
    int32_t contentChecksum;
    int32_t storeOnExpansion;
    uint32_t minStreamSize;
    size_t rgpSize;
    const void* rgpData;
    size_t nbInputs;
    const CodecCache_Input* inputs;
} CodecCache_TryGraphKey;

static size_t CodecCache_Map_hash(const CodecCache_Key* key);
static bool CodecCache_Map_eq(
        const CodecCache_Key* lhs,
        const CodecCache_Key* rhs);

static size_t CodecCache_BlobMap_hash(const CodecCache_BlobKey* key);
static bool CodecCache_BlobMap_eq(
        const CodecCache_BlobKey* lhs,
        const CodecCache_BlobKey* rhs);

static size_t CodecCache_TryGraphMap_hash(const CodecCache_TryGraphKey* key);
static bool CodecCache_TryGraphMap_eq(
        const CodecCache_TryGraphKey* lhs,
        const CodecCache_TryGraphKey* rhs);

ZL_DECLARE_CUSTOM_MAP_TYPE(
        CodecCache_Map,
        CodecCache_Key,
        const CodecCache_Result*);

ZL_DECLARE_CUSTOM_MAP_TYPE(CodecCache_BlobMap, CodecCache_BlobKey, uint8_t);

ZL_DECLARE_CUSTOM_MAP_TYPE(
        CodecCache_TryGraphMap,
        CodecCache_TryGraphKey,
        size_t);

ZL_DECLARE_MAP_TYPE(CodecCache_PurityMap, ZL_IDType, uint8_t);

enum {
    CodecCache_GraphPurity_visiting = 0,
    CodecCache_GraphPurity_impure   = 1,
    CodecCache_GraphPurity_pure     = 2,
};

#define CODEC_CACHE_META_MAX_BLOB_BYTES ((size_t)2048)
#define CODEC_CACHE_INTERN_MIN_OUTPUT_BYTES ((size_t)4096)

struct ZL_CodecOutputCache_s {
    Arena* metaArena;
    Arena* blobArena;
    CodecCache_Map map;
    CodecCache_BlobMap blobMap;
    Arena* tryGraphArena;
    CodecCache_TryGraphMap tryGraphMap;
    Arena* purityArena;
    CodecCache_PurityMap graphPurityMap;
    const ZL_Compressor* purityCGraph;
    bool tryGraphInitialized;
    size_t maxBytes;
    size_t bytesStored;
    size_t tryGraphBytesStored;
    bool insertionsEnabled;
    bool statsEnabled;
    size_t hashReuses;
    CodecCache_Stats stats;
    CodecCache_Stats lastCompletedStats;
};

static Arena* CodecCache_payloadArena(ZL_CodecOutputCache* cache, size_t size)
{
    return size <= CODEC_CACHE_META_MAX_BLOB_BYTES ? cache->metaArena
                                                   : cache->blobArena;
}

struct CodecCache_Lookup_s {
    ZL_CodecOutputCache* cache;
    CodecCache_Key key;
    const CodecCache_Result* result;
};

struct CodecCache_TryGraphLookup_s {
    ZL_CodecOutputCache* cache;
    CodecCache_TryGraphKey key;
    const uint64_t* inputHashes;
};

static bool CodecCache_equalBytes(const void* lhs, const void* rhs, size_t size)
{
    return size == 0 || lhs == rhs || memcmp(lhs, rhs, size) == 0;
}

typedef struct {
    uint64_t contentHash64;
    uint64_t standardNodeID;
    uint32_t formatVersion;
    int32_t compressionLevel;
    int32_t decompressionLevel;
    uint32_t pad;
    uint64_t localParamsSize;
} CodecCache_KeyHashHeader;

static size_t CodecCache_Map_hash(const CodecCache_Key* key)
{
    const CodecCache_KeyHashHeader hdr = {
        .contentHash64      = key->contentHash64,
        .standardNodeID     = (uint64_t)key->standardNodeID,
        .formatVersion      = key->formatVersion,
        .compressionLevel   = key->compressionLevel,
        .decompressionLevel = key->decompressionLevel,
        .pad                = 0,
        .localParamsSize    = (uint64_t)key->localParamsSize,
    };
    if (key->localParamsSize == 0) {
        return (size_t)XXH3_64bits(&hdr, sizeof(hdr));
    }
    XXH3_state_t state;
    XXH3_INITSTATE(&state);
    XXH3_64bits_reset(&state);
    XXH3_64bits_update(&state, &hdr, sizeof(hdr));
    XXH3_64bits_update(&state, key->localParams, key->localParamsSize);
    return (size_t)XXH3_64bits_digest(&state);
}

static size_t CodecCache_BlobMap_hash(const CodecCache_BlobKey* key)
{
    return (size_t)(key->hash64 ^ (key->size * 0x9E3779B97F4A7C15ULL));
}

static bool CodecCache_BlobMap_eq(
        const CodecCache_BlobKey* lhs,
        const CodecCache_BlobKey* rhs)
{
    return lhs->hash64 == rhs->hash64 && lhs->size == rhs->size
            && CodecCache_equalBytes(lhs->data, rhs->data, lhs->size);
}

static bool CodecCache_Input_eq(
        const CodecCache_Input* lhs,
        const CodecCache_Input* rhs)
{
    if (lhs->type != rhs->type || lhs->eltWidth != rhs->eltWidth
        || lhs->numElts != rhs->numElts || lhs->contentSize != rhs->contentSize
        || lhs->nbIntMetadata != rhs->nbIntMetadata) {
        return false;
    }
    size_t metadataSize;
    if (ZL_overflowMulST(
                lhs->nbIntMetadata,
                sizeof(Stream_IntMetadata),
                &metadataSize)) {
        return false;
    }
    return CodecCache_equalBytes(lhs->content, rhs->content, lhs->contentSize)
            && CodecCache_equalBytes(
                    lhs->intMetadata, rhs->intMetadata, metadataSize);
}

static bool CodecCache_Map_eq(
        const CodecCache_Key* lhs,
        const CodecCache_Key* rhs)
{
    if (lhs->contentHash64 != rhs->contentHash64
        || lhs->standardNodeID != rhs->standardNodeID
        || lhs->formatVersion != rhs->formatVersion
        || lhs->compressionLevel != rhs->compressionLevel
        || lhs->decompressionLevel != rhs->decompressionLevel
        || lhs->localParamsSize != rhs->localParamsSize
        || !CodecCache_equalBytes(
                lhs->localParams, rhs->localParams, lhs->localParamsSize)) {
        return false;
    }
    return CodecCache_Input_eq(&lhs->input, &rhs->input);
}

static size_t CodecCache_TryGraphMap_hash(const CodecCache_TryGraphKey* key)
{
    return (size_t)key->combinedHash64;
}

static bool CodecCache_TryGraphMap_eq(
        const CodecCache_TryGraphKey* lhs,
        const CodecCache_TryGraphKey* rhs)
{
    if (lhs->combinedHash64 != rhs->combinedHash64 || lhs->cgraph != rhs->cgraph
        || lhs->graphID != rhs->graphID
        || lhs->formatVersion != rhs->formatVersion
        || lhs->compressionLevel != rhs->compressionLevel
        || lhs->decompressionLevel != rhs->decompressionLevel
        || lhs->permissiveCompression != rhs->permissiveCompression
        || lhs->compressedChecksum != rhs->compressedChecksum
        || lhs->contentChecksum != rhs->contentChecksum
        || lhs->storeOnExpansion != rhs->storeOnExpansion
        || lhs->minStreamSize != rhs->minStreamSize
        || lhs->rgpSize != rhs->rgpSize || lhs->nbInputs != rhs->nbInputs
        || !CodecCache_equalBytes(lhs->rgpData, rhs->rgpData, lhs->rgpSize)) {
        return false;
    }
    for (size_t i = 0; i < lhs->nbInputs; ++i) {
        if (!CodecCache_Input_eq(&lhs->inputs[i], &rhs->inputs[i])) {
            return false;
        }
    }
    return true;
}

static bool CodecCache_addSize(size_t* total, size_t amount)
{
    size_t result;
    if (ZL_overflowAddST(*total, amount, &result)) {
        return false;
    }
    *total = result;
    return true;
}

static bool CodecCache_addArraySize(size_t* total, size_t count, size_t eltSize)
{
    size_t size;
    return !ZL_overflowMulST(count, eltSize, &size)
            && CodecCache_addSize(total, size);
}

typedef struct {
    uint64_t type;
    uint64_t eltWidth;
    uint64_t numElts;
    uint64_t contentSize;
    uint64_t payloadHash;
    uint64_t nbIntMetadata;
} CodecCache_InputHashHeader;

static uint64_t CodecCache_hashInput(const CodecCache_Input* input)
{
    const uint64_t payloadHash = input->contentSize != 0
            ? XXH3_64bits(input->content, input->contentSize)
            : 0;
    const CodecCache_InputHashHeader hdr = {
        .type          = (uint64_t)input->type,
        .eltWidth      = (uint64_t)input->eltWidth,
        .numElts       = (uint64_t)input->numElts,
        .contentSize   = (uint64_t)input->contentSize,
        .payloadHash   = payloadHash,
        .nbIntMetadata = (uint64_t)input->nbIntMetadata,
    };
    if (input->nbIntMetadata == 0) {
        return XXH3_64bits(&hdr, sizeof(hdr));
    }
    XXH3_state_t state;
    XXH3_INITSTATE(&state);
    XXH3_64bits_reset(&state);
    XXH3_64bits_update(&state, &hdr, sizeof(hdr));
    XXH3_64bits_update(
            &state,
            input->intMetadata,
            input->nbIntMetadata * sizeof(input->intMetadata[0]));
    return XXH3_64bits_digest(&state);
}

static bool CodecCache_buildInput(
        CodecCache_Input* cacheInput,
        uint64_t* contentHash,
        ZL_CodecOutputCache* cache,
        Arena* scratchArena,
        const ZL_Data* input)
{
    *cacheInput = (CodecCache_Input){
        .type        = ZL_Data_type(input),
        .eltWidth    = ZL_Data_eltWidth(input),
        .numElts     = ZL_Data_numElts(input),
        .contentSize = ZL_Data_contentSize(input),
        .content     = ZL_Data_rPtr(input),
    };
    cacheInput->nbIntMetadata = STREAM_numIntMetadata(input);
    if (cacheInput->nbIntMetadata != 0) {
        size_t metadataSize;
        if (ZL_overflowMulST(
                    cacheInput->nbIntMetadata,
                    sizeof(Stream_IntMetadata),
                    &metadataSize)) {
            return false;
        }
        Stream_IntMetadata* const metadata =
                ALLOC_Arena_malloc(scratchArena, metadataSize);
        if (metadata == NULL
            || ZL_isError(STREAM_copyIntMetadata(
                    metadata, input, cacheInput->nbIntMetadata))) {
            return false;
        }
        cacheInput->intMetadata = metadata;
    }

    uint64_t memoizedHash;
    if (STREAM_getCodecCacheKeyHash(input, &memoizedHash)) {
        *contentHash = memoizedHash;
        if (cache != NULL) {
            CodecCache_recordHashReuse(cache);
        }
        return true;
    }

    *contentHash = CodecCache_hashInput(cacheInput);
    return true;
}

void CodecCache_ensureStreamKeyHash(
        ZL_CodecOutputCache* cache,
        ZL_Data* stream,
        Arena* scratchArena)
{
    (void)cache;
    if (stream == NULL || ZL_Data_type(stream) == ZL_Type_string) {
        return;
    }
    uint64_t existing;
    if (STREAM_getCodecCacheKeyHash(stream, &existing)) {
        return;
    }
    CodecCache_Input cacheInput;
    uint64_t computed;
    if (CodecCache_buildInput(
                &cacheInput, &computed, NULL, scratchArena, stream)) {
        STREAM_setCodecCacheKeyHash(stream, computed);
    }
}

static bool CodecCache_hasPointerCopyParam(
        ZL_IDType standardNodeID,
        const ZL_LocalParams* lparams)
{
    if (lparams == NULL || lparams->copyParams.nbCopyParams == 0) {
        return false;
    }
    if (standardNodeID == ZL_StandardNodeID_tokenize_struct
        || standardNodeID == ZL_StandardNodeID_tokenize_numeric
        || standardNodeID == ZL_StandardNodeID_tokenize_string
        || standardNodeID == ZL_PrivateStandardNodeID_tokenize_sorted
        || standardNodeID == ZL_PrivateStandardNodeID_tokenize_string_sorted) {
        return true;
    }
    for (size_t i = 0; i < lparams->copyParams.nbCopyParams; ++i) {
        const int pid = lparams->copyParams.copyParams[i].paramId;
        if (standardNodeID == ZL_StandardNodeID_dispatchN_byTag
            && pid == ZL_DISPATCH_PARSINGFN_PID) {
            return true;
        }
        if (standardNodeID == ZL_PrivateStandardNodeID_set_string_lens
            && pid == ZL_SETSTRINGLENS_PARSINGF_PID) {
            return true;
        }
    }
    return false;
}

static bool CodecCache_serializeAllowedRefParams(
        int* extraRefId,
        const void** extraRefData,
        size_t* extraRefSize,
        Arena* scratchArena,
        ZL_IDType standardNodeID,
        const ZL_LocalParams* lparams)
{
    *extraRefId   = 0;
    *extraRefData = NULL;
    *extraRefSize = 0;
    if (lparams == NULL || lparams->refParams.nbRefParams == 0) {
        return true;
    }

    const ZL_LocalRefParams* const rp = &lparams->refParams;
    if (standardNodeID == ZL_PrivateStandardNodeID_fse_v2
        || standardNodeID == ZL_PrivateStandardNodeID_huffman_struct_v2
        || standardNodeID == ZL_PrivateStandardNodeID_huffman_v2
        || standardNodeID == ZL_PrivateStandardNodeID_pivco_huffman) {
        const bool usesCTable =
                standardNodeID == ZL_PrivateStandardNodeID_huffman_v2
                || standardNodeID == ZL_PrivateStandardNodeID_pivco_huffman;
        const ZL_Histogram* hist = NULL;
        const void* ctable       = NULL;
        for (size_t i = 0; i < rp->nbRefParams; ++i) {
            const int pid = rp->refParams[i].paramId;
            if (pid == ENTROPY_HISTORAM_PID) {
                hist = (const ZL_Histogram*)rp->refParams[i].paramRef;
            } else if (pid == ENTROPY_HUF_CTABLE_PID) {
                ctable = rp->refParams[i].paramRef;
            } else {
                return false;
            }
        }
        size_t histBytes = 0;
        if (hist != NULL) {
            if (hist->maxSymbol > UINT16_MAX) {
                return false;
            }
            histBytes = offsetof(ZL_Histogram, count)
                    + ((size_t)hist->maxSymbol + 1) * sizeof(hist->count[0]);
        }
        if (!usesCTable || ctable == NULL) {
            if (histBytes != 0) {
                XXH128_hash_t* const digest =
                        ALLOC_Arena_malloc(scratchArena, sizeof(*digest));
                if (digest == NULL) {
                    return false;
                }
                *digest       = XXH3_128bits(hist, histBytes);
                *extraRefId   = ENTROPY_HISTORAM_PID;
                *extraRefData = digest;
                *extraRefSize = sizeof(*digest);
            }
            return true;
        }
        if (hist == NULL || hist->maxSymbol > HUF_SYMBOLVALUE_MAX) {
            return false;
        }
        const size_t ctableBytes = HUF_CTABLE_SIZE(hist->maxSymbol);
        XXH128_hash_t* const digests =
                ALLOC_Arena_malloc(scratchArena, 2 * sizeof(*digests));
        if (digests == NULL) {
            return false;
        }
        digests[0]    = XXH3_128bits(hist, histBytes);
        digests[1]    = XXH3_128bits(ctable, ctableBytes);
        *extraRefId   = ENTROPY_HUF_CTABLE_PID;
        *extraRefData = digests;
        *extraRefSize = 2 * sizeof(*digests);
        return true;
    }

    if (standardNodeID == ZL_StandardNodeID_dispatchN_byTag) {
        if (lparams->copyParams.nbCopyParams != 0 || rp->nbRefParams != 1
            || rp->refParams[0].paramId != ZL_DISPATCH_INSTRUCTIONS_PID
            || rp->refParams[0].paramRef == NULL) {
            return false;
        }
        const ZL_DispatchInstructions* const inst =
                (const ZL_DispatchInstructions*)rp->refParams[0].paramRef;
        if (inst->nbSegments != 0
            && (inst->segmentSizes == NULL || inst->tags == NULL)) {
            return false;
        }
        size_t bufSize = sizeof(inst->nbTags) + sizeof(inst->nbSegments);
        if (!CodecCache_addArraySize(
                    &bufSize, inst->nbSegments, sizeof(inst->segmentSizes[0]))
            || !CodecCache_addArraySize(
                    &bufSize, inst->nbSegments, sizeof(inst->tags[0]))) {
            return false;
        }
        uint8_t* const buf = ALLOC_Arena_malloc(scratchArena, bufSize);
        if (buf == NULL) {
            return false;
        }
        uint8_t* p = buf;
        memcpy(p, &inst->nbTags, sizeof(inst->nbTags));
        p += sizeof(inst->nbTags);
        memcpy(p, &inst->nbSegments, sizeof(inst->nbSegments));
        p += sizeof(inst->nbSegments);
        if (inst->nbSegments != 0) {
            const size_t segBytes =
                    inst->nbSegments * sizeof(inst->segmentSizes[0]);
            const size_t tagBytes = inst->nbSegments * sizeof(inst->tags[0]);
            memcpy(p, inst->segmentSizes, segBytes);
            p += segBytes;
            memcpy(p, inst->tags, tagBytes);
            p += tagBytes;
        }
        ZL_ASSERT_EQ((size_t)(p - buf), bufSize);
        *extraRefId   = ZL_DISPATCH_INSTRUCTIONS_PID;
        *extraRefData = buf;
        *extraRefSize = bufSize;
        return true;
    }

    return false;
}

static bool CodecCache_serializeLocalParamsWithExtra(
        void** serialized,
        size_t* serializedSize,
        Arena* scratchArena,
        const ZL_LocalParams* params,
        int extraRefId,
        const void* extraRefData,
        size_t extraRefSize)
{
    *serialized     = NULL;
    *serializedSize = 0;

    const size_t nbIntParams =
            params != NULL ? params->intParams.nbIntParams : 0;
    const size_t nbCopyParams =
            params != NULL ? params->copyParams.nbCopyParams : 0;
    const size_t nbExtraCopy = extraRefId != 0 ? 1 : 0;
    const size_t totalCopyParams = nbCopyParams + nbExtraCopy;

    if (nbIntParams == 0 && totalCopyParams == 0) {
        return true;
    }
    if (!CodecCache_addSize(serializedSize, sizeof(nbIntParams))
        || !CodecCache_addSize(serializedSize, sizeof(totalCopyParams))
        || !CodecCache_addArraySize(
                serializedSize, nbIntParams, sizeof(int) + sizeof(int))) {
        return false;
    }
    for (size_t i = 0; i < nbCopyParams; ++i) {
        if (!CodecCache_addSize(serializedSize, sizeof(int) + sizeof(size_t))
            || !CodecCache_addSize(
                    serializedSize, params->copyParams.copyParams[i].paramSize)) {
            return false;
        }
    }
    if (nbExtraCopy != 0) {
        if (!CodecCache_addSize(serializedSize, sizeof(int) + sizeof(size_t))
            || !CodecCache_addSize(serializedSize, extraRefSize)) {
            return false;
        }
    }
    *serialized = ALLOC_Arena_malloc(scratchArena, *serializedSize);
    if (*serialized == NULL) {
        return false;
    }
    uint8_t* dst = *serialized;
    memcpy(dst, &nbIntParams, sizeof(nbIntParams));
    dst += sizeof(nbIntParams);
    memcpy(dst, &totalCopyParams, sizeof(totalCopyParams));
    dst += sizeof(totalCopyParams);
    for (size_t i = 0; i < nbIntParams; ++i) {
        const int id    = params->intParams.intParams[i].paramId;
        const int value = params->intParams.intParams[i].paramValue;
        memcpy(dst, &id, sizeof(id));
        dst += sizeof(id);
        memcpy(dst, &value, sizeof(value));
        dst += sizeof(value);
    }
    for (size_t i = 0; i < nbCopyParams; ++i) {
        const ZL_CopyParam* const param = &params->copyParams.copyParams[i];
        memcpy(dst, &param->paramId, sizeof(param->paramId));
        dst += sizeof(param->paramId);
        memcpy(dst, &param->paramSize, sizeof(param->paramSize));
        dst += sizeof(param->paramSize);
        if (param->paramSize != 0) {
            memcpy(dst, param->paramPtr, param->paramSize);
            dst += param->paramSize;
        }
    }
    if (nbExtraCopy != 0) {
        memcpy(dst, &extraRefId, sizeof(extraRefId));
        dst += sizeof(extraRefId);
        memcpy(dst, &extraRefSize, sizeof(extraRefSize));
        dst += sizeof(extraRefSize);
        if (extraRefSize != 0) {
            memcpy(dst, extraRefData, extraRefSize);
            dst += extraRefSize;
        }
    }
    ZL_ASSERT_EQ((size_t)(dst - (uint8_t*)*serialized), *serializedSize);
    return true;
}

static bool CodecCache_resolveStandardNodeID(
        ZL_IDType* standardNodeID,
        const ZL_Compressor* compressor,
        ZL_NodeID node)
{
    while (!NM_isStandardNode(node)) {
        const CNode* const cnode = CGRAPH_getCNode(compressor, node);
        if (cnode == NULL) {
            return false;
        }
        const ZL_NodeID base = CNODE_getBaseNodeID(cnode);
        if (base.nid == ZL_NODE_ILLEGAL.nid) {
            return false;
        }
        node = base;
    }
    *standardNodeID = node.nid;
    return true;
}

static bool CodecCache_getStandardNodeID(
        ZL_IDType* standardNodeID,
        const ZL_Encoder* encoder,
        ZL_NodeID node)
{
    const ZL_Compressor* const compressor = CCTX_getCGraph(encoder->cctx);
    ZL_ASSERT_NN(compressor);
    ZL_ASSERT_EQ(CGRAPH_getCNode(compressor, node), encoder->cnode);
    return CodecCache_resolveStandardNodeID(standardNodeID, compressor, node);
}

static bool CodecCache_buildKey(
        CodecCache_Key* key,
        ZL_CodecOutputCache* cache,
        ZL_Encoder* encoder,
        ZL_NodeID node,
        const ZL_Data* input)
{
    ZL_IDType standardNodeID;
    if (!CodecCache_getStandardNodeID(&standardNodeID, encoder, node)) {
        CodecCache_recordSkip(cache, CodecCache_SkipReason_customCodec);
        return false;
    }
    if (CodecCache_hasPointerCopyParam(standardNodeID, encoder->lparams)) {
        CodecCache_recordSkip(cache, CodecCache_SkipReason_customCodec);
        return false;
    }
    if (CNODE_getDictIndex(encoder->cnode) != ZL_DICT_INDEX_NONE) {
        CodecCache_recordSkip(cache, CodecCache_SkipReason_dict);
        return false;
    }
    if (CNODE_getMParamObj(encoder->cnode) != NULL) {
        CodecCache_recordSkip(cache, CodecCache_SkipReason_mparam);
        return false;
    }
    int extraRefId         = 0;
    const void* extraRefPtr = NULL;
    size_t extraRefSize     = 0;
    if (encoder->lparams != NULL
        && encoder->lparams->refParams.nbRefParams != 0) {
        if (!CodecCache_serializeAllowedRefParams(
                    &extraRefId,
                    &extraRefPtr,
                    &extraRefSize,
                    encoder->wkspArena,
                    standardNodeID,
                    encoder->lparams)) {
            CodecCache_recordSkip(cache, CodecCache_SkipReason_refParam);
            return false;
        }
    }
    if (ZL_Data_type(input) == ZL_Type_string) {
        CodecCache_recordSkip(cache, CodecCache_SkipReason_string);
        return false;
    }

    CodecCache_Input cacheInput;
    uint64_t contentHash;
    if (!CodecCache_buildInput(
                &cacheInput, &contentHash, cache, encoder->wkspArena, input)) {
        return false;
    }
    void* localParams;
    size_t localParamsSize;
    if (!CodecCache_serializeLocalParamsWithExtra(
                &localParams,
                &localParamsSize,
                encoder->wkspArena,
                encoder->lparams,
                extraRefId,
                extraRefPtr,
                extraRefSize)) {
        return false;
    }

    *key = (CodecCache_Key){
        .contentHash64  = contentHash,
        .standardNodeID = standardNodeID,
        .formatVersion  = (uint32_t)ZL_Encoder_getCParam(
                encoder, ZL_CParam_formatVersion),
        .compressionLevel =
                ZL_Encoder_getCParam(encoder, ZL_CParam_compressionLevel),
        .decompressionLevel =
                ZL_Encoder_getCParam(encoder, ZL_CParam_decompressionLevel),
        .input           = cacheInput,
        .localParamsSize = localParamsSize,
        .localParams     = localParams,
    };
    return true;
}

ZL_CodecOutputCache* CodecCache_create(size_t maxBytes)
{
    ZL_CodecOutputCache* const cache = ZL_calloc(sizeof(*cache));
    if (cache == NULL) {
        return NULL;
    }
    cache->metaArena = ALLOC_StackArena_create();
    cache->blobArena = ALLOC_HeapArena_create();
    if (cache->metaArena == NULL || cache->blobArena == NULL) {
        ALLOC_Arena_freeArena(cache->blobArena);
        ALLOC_Arena_freeArena(cache->metaArena);
        ZL_free(cache);
        return NULL;
    }
    cache->maxBytes          = maxBytes;
    cache->insertionsEnabled = true;
    cache->map               = CodecCache_Map_createInArena(
            cache->metaArena, CODEC_CACHE_MAX_ENTRIES);
    cache->blobMap = CodecCache_BlobMap_createInArena(
            cache->metaArena, CODEC_CACHE_MAX_ENTRIES);
    return cache;
}

size_t CodecCache_getDefaultMaxBytes(void)
{
    return CODEC_CACHE_DEFAULT_MAX_BYTES;
}

void CodecCache_free(ZL_CodecOutputCache* cache)
{
    if (cache == NULL) {
        return;
    }
    ALLOC_Arena_freeArena(cache->purityArena);
    ALLOC_Arena_freeArena(cache->tryGraphArena);
    ALLOC_Arena_freeArena(cache->blobArena);
    ALLOC_Arena_freeArena(cache->metaArena);
    ZL_free(cache);
}

size_t CodecCache_sizeof(const ZL_CodecOutputCache* cache)
{
    if (cache == NULL) {
        return 0;
    }
    return sizeof(*cache) + ALLOC_Arena_memAllocated(cache->metaArena)
            + ALLOC_Arena_memAllocated(cache->blobArena)
            + ALLOC_Arena_memAllocated(cache->tryGraphArena)
            + ALLOC_Arena_memAllocated(cache->purityArena);
}

void CodecCache_resetTryGraph(ZL_CodecOutputCache* cache)
{
    if (cache == NULL || cache->tryGraphArena == NULL) {
        return;
    }
    ALLOC_Arena_freeAll(cache->tryGraphArena);
    cache->tryGraphInitialized = false;
    cache->tryGraphBytesStored = 0;
}

static bool CodecCache_ensureTryGraphInitialized(
        ZL_CodecOutputCache* cache,
        const ZL_Compressor* cgraph)
{
    if (!cache->tryGraphInitialized) {
        if (cache->tryGraphArena == NULL) {
            cache->tryGraphArena = ALLOC_StackArena_create();
            if (cache->tryGraphArena == NULL) {
                return false;
            }
        }
        cache->tryGraphMap = CodecCache_TryGraphMap_createInArena(
                cache->tryGraphArena, CODEC_CACHE_TRYGRAPH_MAX_ENTRIES);
        cache->tryGraphInitialized = true;
        cache->tryGraphBytesStored = 0;
    }
    if (cache->purityCGraph != cgraph) {
        if (cache->purityArena == NULL) {
            cache->purityArena = ALLOC_StackArena_create();
            if (cache->purityArena == NULL) {
                return false;
            }
        } else {
            ALLOC_Arena_freeAll(cache->purityArena);
        }
        cache->graphPurityMap = CodecCache_PurityMap_createInArena(
                cache->purityArena, CODEC_CACHE_TRYGRAPH_MAX_ENTRIES);
        cache->purityCGraph = cgraph;
    }
    return true;
}

static void CodecCache_resetCurrent(ZL_CodecOutputCache* cache)
{
    ZL_ASSERT_NN(cache);
    CodecCache_resetTryGraph(cache);
    ALLOC_Arena_freeAll(cache->blobArena);
    ALLOC_Arena_freeAll(cache->metaArena);
    cache->map = CodecCache_Map_createInArena(
            cache->metaArena, CODEC_CACHE_MAX_ENTRIES);
    cache->blobMap = CodecCache_BlobMap_createInArena(
            cache->metaArena, CODEC_CACHE_MAX_ENTRIES);
    cache->bytesStored       = 0;
    cache->insertionsEnabled = true;
    if (cache->statsEnabled) {
        cache->hashReuses = 0;
        memset(&cache->stats, 0, sizeof(cache->stats));
    }
}

void CodecCache_setStatsEnabled(ZL_CodecOutputCache* cache, bool enabled)
{
    ZL_ASSERT_NN(cache);
    cache->statsEnabled = enabled;
    cache->hashReuses   = 0;
    memset(&cache->stats, 0, sizeof(cache->stats));
    memset(&cache->lastCompletedStats, 0, sizeof(cache->lastCompletedStats));
}

void CodecCache_reset(ZL_CodecOutputCache* cache)
{
    if (cache == NULL) {
        return;
    }
    CodecCache_resetCurrent(cache);
    if (cache->statsEnabled) {
        memset(&cache->lastCompletedStats,
               0,
               sizeof(cache->lastCompletedStats));
    }
}

void CodecCache_resetPreservingCompletedStats(ZL_CodecOutputCache* cache)
{
    if (cache == NULL) {
        return;
    }
    CodecCache_resetCurrent(cache);
}

void CodecCache_setInsertionsEnabled(ZL_CodecOutputCache* cache, bool enabled)
{
    ZL_ASSERT_NN(cache);
    cache->insertionsEnabled = enabled;
}

void CodecCache_recordHashReuse(ZL_CodecOutputCache* cache)
{
    ZL_ASSERT_NN(cache);
    if (cache->statsEnabled) {
        ++cache->hashReuses;
    }
}

size_t CodecCache_getHashReuses(const ZL_CodecOutputCache* cache)
{
    ZL_ASSERT_NN(cache);
    return cache->statsEnabled ? cache->hashReuses : 0;
}

CodecCache_Stats CodecCache_getStats(const ZL_CodecOutputCache* cache)
{
    ZL_ASSERT_NN(cache);
    if (!cache->statsEnabled) {
        return (CodecCache_Stats){ 0 };
    }
    CodecCache_Stats stats = cache->stats;
    stats.bytesStored      = cache->bytesStored;
    stats.arenaBytes       = ALLOC_Arena_memAllocated(cache->metaArena)
            + ALLOC_Arena_memAllocated(cache->blobArena)
            + ALLOC_Arena_memAllocated(cache->tryGraphArena)
            + ALLOC_Arena_memAllocated(cache->purityArena);
    return stats;
}

void CodecCache_captureCompletedStats(ZL_CodecOutputCache* cache)
{
    ZL_ASSERT_NN(cache);
    if (cache->statsEnabled) {
        cache->lastCompletedStats = CodecCache_getStats(cache);
    }
}

CodecCache_Stats CodecCache_getLastCompletedStats(
        const ZL_CodecOutputCache* cache)
{
    ZL_ASSERT_NN(cache);
    if (!cache->statsEnabled) {
        return (CodecCache_Stats){ 0 };
    }
    return cache->lastCompletedStats;
}

void CodecCache_recordSkip(
        ZL_CodecOutputCache* cache,
        CodecCache_SkipReason reason)
{
    ZL_ASSERT_NN(cache);
    if (!cache->statsEnabled) {
        return;
    }
    switch (reason) {
        case CodecCache_SkipReason_customCodec:
            ++cache->stats.customCodecSkips;
            break;
        case CodecCache_SkipReason_refParam:
            ++cache->stats.refParamSkips;
            break;
        case CodecCache_SkipReason_dict:
            ++cache->stats.dictSkips;
            break;
        case CodecCache_SkipReason_mparam:
            ++cache->stats.mparamSkips;
            break;
        case CodecCache_SkipReason_string:
            ++cache->stats.stringSkips;
            break;
        case CodecCache_SkipReason_nonSingleInput:
            ++cache->stats.nonSingleInputSkips;
            break;
    }
}

ZL_CodecOutputCache* ZL_CodecOutputCache_create(void)
{
    return CodecCache_create(CodecCache_getDefaultMaxBytes());
}

ZL_CodecOutputCache* ZL_CodecOutputCache_createWithBudget(size_t maxBytes)
{
    return CodecCache_create(maxBytes);
}

void ZL_CodecOutputCache_free(ZL_CodecOutputCache* cache)
{
    CodecCache_free(cache);
}

void ZL_CodecOutputCache_reset(ZL_CodecOutputCache* cache)
{
    CodecCache_reset(cache);
}

CodecCache_Lookup* CodecCache_lookup(
        ZL_CodecOutputCache* cache,
        ZL_Encoder* encoder,
        ZL_NodeID node,
        const ZL_Data* input)
{
    ZL_ASSERT_NN(cache);
    ZL_ASSERT_NN(encoder);
    CodecCache_Lookup* const lookup =
            ALLOC_Arena_malloc(encoder->wkspArena, sizeof(*lookup));
    if (lookup == NULL
        || !CodecCache_buildKey(&lookup->key, cache, encoder, node, input)) {
        return NULL;
    }
    lookup->cache = cache;
    const CodecCache_Map_Entry* const found =
            CodecCache_Map_find(&cache->map, &lookup->key);
    if (found == NULL) {
        if (cache->statsEnabled) {
            ++cache->stats.misses;
        }
        if (!cache->insertionsEnabled) {
            return NULL;
        }
        lookup->result = NULL;
    } else {
        if (cache->statsEnabled) {
            ++cache->stats.hits;
        }
        lookup->result = found->val;
    }
    return lookup;
}

const CodecCache_Result* CodecCache_Lookup_getResult(
        const CodecCache_Lookup* lookup)
{
    ZL_ASSERT_NN(lookup);
    return lookup->result;
}

static bool CodecCache_storedSize(
        const CodecCache_Key* key,
        const CodecCache_Result* result,
        bool inputContentShared,
        size_t* storedSize)
{
    size_t size                 = 0;
    const size_t inputBytesCost = inputContentShared ? 0 : key->input.contentSize;
    if (!CodecCache_addSize(&size, key->localParamsSize)
        || !CodecCache_addSize(&size, sizeof(key->input))
        || !CodecCache_addArraySize(
                &size, result->nbOutputs, sizeof(result->outputs[0]))
        || !CodecCache_addSize(&size, result->headerSize)
        || !CodecCache_addSize(&size, inputBytesCost)
        || !CodecCache_addArraySize(
                &size, key->input.nbIntMetadata, sizeof(Stream_IntMetadata))) {
        return false;
    }
    for (size_t i = 0; i < result->nbOutputs; ++i) {
        if (!CodecCache_addSize(&size, result->outputs[i].contentSize)
            || !CodecCache_addArraySize(
                    &size,
                    result->outputs[i].nbIntMetadata,
                    sizeof(Stream_IntMetadata))) {
            return false;
        }
    }
    *storedSize = size;
    return true;
}

static void*
CodecCache_copyBytes(Arena* cacheArena, const void* src, size_t size)
{
    if (size == 0) {
        return NULL;
    }
    ZL_ASSERT_NN(src);
    void* const dst = ALLOC_Arena_malloc(cacheArena, size);
    if (dst != NULL) {
        memcpy(dst, src, size);
    }
    return dst;
}

static void CodecCache_freeCopy(Arena* cacheArena, const void* copy)
{
    /* Stored descriptors expose their cache-owned buffers as read-only. */
    void* allocation;
    memcpy(&allocation, &copy, sizeof(allocation));
    ALLOC_Arena_free(cacheArena, allocation);
}

static void CodecCache_freeInput(
        ZL_CodecOutputCache* cache,
        Arena* structArena,
        const CodecCache_Input* input,
        bool ownsContent)
{
    if (ownsContent) {
        CodecCache_freeCopy(
                CodecCache_payloadArena(cache, input->contentSize),
                input->content);
    }
    CodecCache_freeCopy(structArena, input->intMetadata);
}

static bool CodecCache_copyInput(
        ZL_CodecOutputCache* cache,
        Arena* structArena,
        const CodecCache_Input* src,
        const void* sharedContent,
        CodecCache_Input* dst,
        bool* ownsContentOut)
{
    *dst             = *src;
    dst->content     = NULL;
    dst->intMetadata = NULL;
    *ownsContentOut  = false;
    if (sharedContent != NULL) {
        dst->content = sharedContent;
    } else if (src->contentSize != 0) {
        dst->content = CodecCache_copyBytes(
                CodecCache_payloadArena(cache, src->contentSize),
                src->content,
                src->contentSize);
        if (dst->content == NULL) {
            return false;
        }
        *ownsContentOut = true;
    }
    size_t metadataSize;
    if (ZL_overflowMulST(
                src->nbIntMetadata,
                sizeof(Stream_IntMetadata),
                &metadataSize)) {
        return false;
    }
    dst->intMetadata =
            CodecCache_copyBytes(structArena, src->intMetadata, metadataSize);
    if (metadataSize != 0 && dst->intMetadata == NULL) {
        return false;
    }
    return true;
}

static void CodecCache_freeOutputs(
        ZL_CodecOutputCache* cache,
        const CodecCache_Output* outputs,
        size_t nbOutputs)
{
    if (outputs == NULL) {
        return;
    }
    for (size_t i = 0; i < nbOutputs; ++i) {
        CodecCache_freeCopy(
                CodecCache_payloadArena(cache, outputs[i].contentSize),
                outputs[i].content);
        CodecCache_freeCopy(cache->metaArena, outputs[i].intMetadata);
    }
    CodecCache_freeCopy(cache->metaArena, outputs);
}

static bool CodecCache_copyOutputs(
        ZL_CodecOutputCache* cache,
        const CodecCache_Result* src,
        CodecCache_Result* dst)
{
    if (src->nbOutputs == 0) {
        dst->outputs = NULL;
        return true;
    }
    size_t outputsSize;
    if (ZL_overflowMulST(
                src->nbOutputs, sizeof(src->outputs[0]), &outputsSize)) {
        return false;
    }
    CodecCache_Output* const outputs =
            ALLOC_Arena_malloc(cache->metaArena, outputsSize);
    if (outputs == NULL) {
        return false;
    }
    for (size_t i = 0; i < src->nbOutputs; ++i) {
        outputs[i]               = src->outputs[i];
        outputs[i].content       = NULL;
        outputs[i].intMetadata   = NULL;
        outputs[i].keyHash64     = 0;
        outputs[i].hasKeyHash64  = false;
        outputs[i].content       = CodecCache_copyBytes(
                CodecCache_payloadArena(cache, src->outputs[i].contentSize),
                src->outputs[i].content,
                src->outputs[i].contentSize);
        if (src->outputs[i].contentSize != 0 && outputs[i].content == NULL) {
            CodecCache_freeOutputs(cache, outputs, i + 1);
            return false;
        }
        size_t metadataSize;
        if (ZL_overflowMulST(
                    src->outputs[i].nbIntMetadata,
                    sizeof(Stream_IntMetadata),
                    &metadataSize)) {
            CodecCache_freeOutputs(cache, outputs, i + 1);
            return false;
        }
        outputs[i].intMetadata = CodecCache_copyBytes(
                cache->metaArena, src->outputs[i].intMetadata, metadataSize);
        if (metadataSize != 0 && outputs[i].intMetadata == NULL) {
            CodecCache_freeOutputs(cache, outputs, i + 1);
            return false;
        }
        if (outputs[i].contentSize >= CODEC_CACHE_INTERN_MIN_OUTPUT_BYTES) {
            const CodecCache_Input outputAsInput = {
                .type          = outputs[i].type,
                .eltWidth      = outputs[i].eltWidth,
                .numElts       = outputs[i].numElts,
                .contentSize   = outputs[i].contentSize,
                .content       = outputs[i].content,
                .nbIntMetadata = outputs[i].nbIntMetadata,
                .intMetadata   = outputs[i].intMetadata,
            };
            outputs[i].keyHash64    = CodecCache_hashInput(&outputAsInput);
            outputs[i].hasKeyHash64 = true;
        }
    }
    dst->outputs = outputs;
    return true;
}

static void CodecCache_freeStoredEntry(
        ZL_CodecOutputCache* cache,
        CodecCache_Key* key,
        bool ownsInputContent,
        CodecCache_Result* result)
{
    CodecCache_freeCopy(cache->metaArena, key->localParams);
    CodecCache_freeInput(cache, cache->metaArena, &key->input, ownsInputContent);
    if (result != NULL) {
        CodecCache_freeOutputs(cache, result->outputs, result->nbOutputs);
        CodecCache_freeCopy(cache->metaArena, result->header);
        ALLOC_Arena_free(cache->metaArena, result);
    }
}

static void CodecCache_internBlob(
        ZL_CodecOutputCache* cache,
        uint64_t hash64,
        const void* data,
        size_t size)
{
    if (size == 0 || data == NULL) {
        return;
    }
    const CodecCache_BlobMap_Entry entry = {
        .key = { .hash64 = hash64, .size = size, .data = data },
        .val = 1,
    };
    (void)CodecCache_BlobMap_insert(&cache->blobMap, &entry);
}

CodecCache_InsertResult CodecCache_storeAndGetResult(
        const CodecCache_Lookup* lookup,
        const CodecCache_Result* result,
        const CodecCache_Result** storedResultOut)
{
    ZL_ASSERT_NN(lookup);
    ZL_ASSERT_NN(result);
    if (storedResultOut != NULL) {
        *storedResultOut = NULL;
    }
    ZL_CodecOutputCache* const cache = lookup->cache;
    const CodecCache_Key* const key  = &lookup->key;

    if (!cache->insertionsEnabled) {
        return CodecCache_InsertResult_notCacheable;
    }

    for (size_t i = 0; i < result->nbOutputs; ++i) {
        if (result->outputs[i].type == ZL_Type_string) {
            CodecCache_recordSkip(cache, CodecCache_SkipReason_string);
            return CodecCache_InsertResult_notCacheable;
        }
    }

    const CodecCache_Map_Entry* const existingEntry =
            CodecCache_Map_find(&cache->map, key);
    if (existingEntry != NULL) {
        if (storedResultOut != NULL) {
            *storedResultOut = existingEntry->val;
        }
        if (cache->statsEnabled) {
            ++cache->stats.duplicateInserts;
        }
        return CodecCache_InsertResult_duplicate;
    }

    const void* sharedInputContent = NULL;
    if (key->input.contentSize != 0) {
        const CodecCache_BlobKey probe = {
            .hash64 = key->contentHash64,
            .size   = key->input.contentSize,
            .data   = key->input.content,
        };
        const CodecCache_BlobMap_Entry* const foundBlob =
                CodecCache_BlobMap_find(&cache->blobMap, &probe);
        if (foundBlob != NULL) {
            sharedInputContent = foundBlob->key.data;
        }
    }

    size_t storedSize;
    size_t newBytesStored;
    if (!CodecCache_storedSize(
                key, result, sharedInputContent != NULL, &storedSize)
        || ZL_overflowAddST(cache->bytesStored, storedSize, &newBytesStored)
        || newBytesStored > cache->maxBytes) {
        if (cache->statsEnabled) {
            ++cache->stats.budgetSkips;
        }
        return CodecCache_InsertResult_budgetExceeded;
    }

    if (cache->tryGraphBytesStored != 0
        && newBytesStored + cache->tryGraphBytesStored > cache->maxBytes) {
        CodecCache_resetTryGraph(cache);
    }

    CodecCache_Key storedKey        = *key;
    storedKey.localParams           = NULL;
    storedKey.input.content         = NULL;
    storedKey.input.intMetadata     = NULL;
    bool ownsInputContent           = false;
    CodecCache_Result* storedResult = NULL;

    storedKey.localParams = CodecCache_copyBytes(
            cache->metaArena, key->localParams, key->localParamsSize);
    if ((key->localParamsSize != 0 && storedKey.localParams == NULL)
        || !CodecCache_copyInput(
                cache,
                cache->metaArena,
                &key->input,
                sharedInputContent,
                &storedKey.input,
                &ownsInputContent)) {
        goto allocationFailure;
    }

    storedResult = ALLOC_Arena_malloc(cache->metaArena, sizeof(*storedResult));
    if (storedResult == NULL) {
        goto allocationFailure;
    }
    *storedResult         = *result;
    storedResult->outputs = NULL;
    storedResult->header  = NULL;
    if (!CodecCache_copyOutputs(cache, result, storedResult)) {
        goto allocationFailure;
    }
    storedResult->header = CodecCache_copyBytes(
            cache->metaArena, result->header, result->headerSize);
    if (result->headerSize != 0 && storedResult->header == NULL) {
        goto allocationFailure;
    }

    const CodecCache_Map_Entry mapEntry = {
        .key = storedKey,
        .val = storedResult,
    };
    const CodecCache_Map_Insert inserted =
            CodecCache_Map_insert(&cache->map, &mapEntry);
    if (inserted.badAlloc) {
        goto allocationFailure;
    }
    if (!inserted.inserted) {
        CodecCache_freeStoredEntry(
                cache, &storedKey, ownsInputContent, storedResult);
        if (storedResultOut != NULL) {
            *storedResultOut = inserted.ptr->val;
        }
        if (cache->statsEnabled) {
            ++cache->stats.duplicateInserts;
        }
        return CodecCache_InsertResult_duplicate;
    }

    if (ownsInputContent && storedKey.input.contentSize != 0) {
        CodecCache_internBlob(
                cache,
                storedKey.contentHash64,
                storedKey.input.content,
                storedKey.input.contentSize);
    }
    for (size_t i = 0; i < storedResult->nbOutputs; ++i) {
        if (storedResult->outputs[i].hasKeyHash64) {
            CodecCache_internBlob(
                    cache,
                    storedResult->outputs[i].keyHash64,
                    storedResult->outputs[i].content,
                    storedResult->outputs[i].contentSize);
        }
    }

    cache->bytesStored = newBytesStored;
    if (storedResultOut != NULL) {
        *storedResultOut = storedResult;
    }
    if (cache->statsEnabled) {
        ++cache->stats.inserts;
    }
    return CodecCache_InsertResult_inserted;

allocationFailure:
    CodecCache_freeStoredEntry(
            cache, &storedKey, ownsInputContent, storedResult);
    if (cache->statsEnabled) {
        ++cache->stats.allocationFailures;
    }
    return CodecCache_InsertResult_allocationFailure;
}

CodecCache_InsertResult CodecCache_store(
        const CodecCache_Lookup* lookup,
        const CodecCache_Result* result)
{
    return CodecCache_storeAndGetResult(lookup, result, NULL);
}

/* =========================================================================
 * Per-chunk Pure-Graph TryGraph Memoization
 * ========================================================================= */

static bool CodecCache_hasBFNarrowIntParam(const ZL_LocalParams* lparams)
{
    if (lparams == NULL) {
        return false;
    }
    for (size_t i = 0; i < lparams->intParams.nbIntParams; ++i) {
        const int pid = lparams->intParams.intParams[i].paramId;
        if (pid >= ZL_BRUTE_FORCE_NARROW_PARAM_BASE
            && pid < ZL_BRUTE_FORCE_NARROW_PARAM_BASE
                            + ZL_BRUTE_FORCE_NARROW_MAX_STAGES) {
            return true;
        }
    }
    return false;
}

static bool CodecCache_isNodePure(
        const ZL_Compressor* cgraph,
        ZL_NodeID nodeid)
{
    if (!ZL_NodeID_isValid(nodeid)) {
        return false;
    }
    ZL_IDType standardNodeID;
    if (!CodecCache_resolveStandardNodeID(&standardNodeID, cgraph, nodeid)) {
        return false;
    }
    ZL_NodeID cur = nodeid;
    while (!NM_isStandardNode(cur)) {
        const CNode* const cnode = CGRAPH_getCNode(cgraph, cur);
        if (cnode == NULL || cnode->nodetype != node_internalTransform) {
            return false;
        }
        if (CNODE_getDictIndex(cnode) != ZL_DICT_INDEX_NONE
            || CNODE_getMParamObj(cnode) != NULL) {
            return false;
        }
        const ZL_LocalParams* const lp = CNODE_getLocalParams(cnode);
        if (lp != NULL && lp->refParams.nbRefParams != 0) {
            return false;
        }
        if (CodecCache_hasPointerCopyParam(standardNodeID, lp)) {
            return false;
        }
        cur = CNODE_getBaseNodeID(cnode);
    }
    return true;
}

static bool CodecCache_isGraphPure(
        ZL_CodecOutputCache* cache,
        const ZL_Compressor* cgraph,
        ZL_GraphID graphid)
{
    if (!CGRAPH_checkGraphIDExists(cgraph, graphid)) {
        return false;
    }
    const GraphType_e runtimeType = CGRAPH_graphType(cgraph, graphid);
    if (runtimeType == gt_illegal || runtimeType == gt_segmenter) {
        return false;
    }
    if (runtimeType == gt_store) {
        return true;
    }

    const CodecCache_PurityMap_Entry* const cached =
            CodecCache_PurityMap_find(&cache->graphPurityMap, &graphid.gid);
    if (cached != NULL) {
        return cached->val == CodecCache_GraphPurity_pure;
    }

    const CodecCache_PurityMap_Entry visitingEntry = {
        .key = graphid.gid,
        .val = CodecCache_GraphPurity_visiting,
    };
    const CodecCache_PurityMap_Insert ins =
            CodecCache_PurityMap_insert(&cache->graphPurityMap, &visitingEntry);
    if (ins.badAlloc) {
        return false;
    }

    bool isPure = false;
    if (CGRAPH_getGraphMParamObj(cgraph, graphid) != NULL) {
        goto finish;
    }
    {
        const ZL_LocalParams lp =
                ZL_Compressor_Graph_getLocalParams(cgraph, graphid);
        if (lp.refParams.nbRefParams != 0
            || CodecCache_hasBFNarrowIntParam(&lp)) {
            goto finish;
        }
    }

    if (GR_isStandardGraph(graphid)) {
        isPure = true;
        goto finish;
    }

    switch (ZL_Compressor_getGraphType(cgraph, graphid)) {
        case ZL_GraphType_standard:
            isPure = GR_isStandardGraph(graphid) != 0;
            break;
        case ZL_GraphType_static: {
            const ZL_NodeID head =
                    ZL_Compressor_Graph_getHeadNode(cgraph, graphid);
            if (!CodecCache_isNodePure(cgraph, head)) {
                break;
            }
            const ZL_GraphIDList succs =
                    ZL_Compressor_Graph_getSuccessors(cgraph, graphid);
            isPure = true;
            for (size_t i = 0; i < succs.nbGraphIDs; ++i) {
                if (!CodecCache_isGraphPure(
                            cache, cgraph, succs.graphids[i])) {
                    isPure = false;
                    break;
                }
            }
            break;
        }
        case ZL_GraphType_parameterized: {
            const ZL_GraphID base =
                    ZL_Compressor_Graph_getBaseGraphID(cgraph, graphid);
            if (!CodecCache_isGraphPure(cache, cgraph, base)) {
                break;
            }
            const ZL_NodeIDList cnodes =
                    ZL_Compressor_Graph_getCustomNodes(cgraph, graphid);
            for (size_t i = 0; i < cnodes.nbNodeIDs; ++i) {
                if (!CodecCache_isNodePure(cgraph, cnodes.nodeids[i])) {
                    goto finish;
                }
            }
            const ZL_GraphIDList cgraphs =
                    ZL_Compressor_Graph_getCustomGraphs(cgraph, graphid);
            for (size_t i = 0; i < cgraphs.nbGraphIDs; ++i) {
                if (!CodecCache_isGraphPure(
                            cache, cgraph, cgraphs.graphids[i])) {
                    goto finish;
                }
            }
            isPure = true;
            break;
        }
        case ZL_GraphType_selector:
        case ZL_GraphType_function:
        case ZL_GraphType_multiInput:
        case ZL_GraphType_segmenter:
        default:
            isPure = false;
            break;
    }

finish:;
    CodecCache_PurityMap_Entry* const mutEntry =
            CodecCache_PurityMap_findMut(&cache->graphPurityMap, &graphid.gid);
    if (mutEntry != NULL) {
        mutEntry->val = isPure ? CodecCache_GraphPurity_pure
                               : CodecCache_GraphPurity_impure;
    }
    return isPure;
}

static bool CodecCache_serializeRGP(
        ZL_CodecOutputCache* cache,
        const ZL_Compressor* cgraph,
        const ZL_RuntimeGraphParameters* rgp,
        Arena* scratchArena,
        void** outData,
        size_t* outSize)
{
    *outData = NULL;
    *outSize = 0;
    if (rgp == NULL) {
        return true;
    }
    if (rgp->mparam.size != 0 || rgp->mparam.content != NULL) {
        return false;
    }
    if (rgp->localParams != NULL) {
        if (rgp->localParams->refParams.nbRefParams != 0
            || CodecCache_hasBFNarrowIntParam(rgp->localParams)) {
            return false;
        }
    }
    for (size_t i = 0; i < rgp->nbCustomNodes; ++i) {
        if (!CodecCache_isNodePure(cgraph, rgp->customNodes[i])) {
            return false;
        }
    }
    for (size_t i = 0; i < rgp->nbCustomGraphs; ++i) {
        if (!CodecCache_isGraphPure(cache, cgraph, rgp->customGraphs[i])) {
            return false;
        }
    }

    void* lpData  = NULL;
    size_t lpSize = 0;
    if (!CodecCache_serializeLocalParamsWithExtra(
                &lpData,
                &lpSize,
                scratchArena,
                rgp->localParams,
                0,
                NULL,
                0)) {
        return false;
    }
    if (rgp->nbCustomGraphs == 0 && rgp->nbCustomNodes == 0 && lpSize == 0) {
        return true;
    }
    size_t total = sizeof(rgp->nbCustomGraphs) + sizeof(rgp->nbCustomNodes)
            + sizeof(lpSize);
    if (!CodecCache_addArraySize(
                &total, rgp->nbCustomGraphs, sizeof(ZL_GraphID))
        || !CodecCache_addArraySize(
                &total, rgp->nbCustomNodes, sizeof(ZL_NodeID))
        || !CodecCache_addSize(&total, lpSize)) {
        return false;
    }
    uint8_t* const buf = ALLOC_Arena_malloc(scratchArena, total);
    if (buf == NULL) {
        return false;
    }
    uint8_t* p = buf;
    memcpy(p, &rgp->nbCustomGraphs, sizeof(rgp->nbCustomGraphs));
    p += sizeof(rgp->nbCustomGraphs);
    memcpy(p, &rgp->nbCustomNodes, sizeof(rgp->nbCustomNodes));
    p += sizeof(rgp->nbCustomNodes);
    memcpy(p, &lpSize, sizeof(lpSize));
    p += sizeof(lpSize);
    if (rgp->nbCustomGraphs != 0) {
        const size_t cgBytes = rgp->nbCustomGraphs * sizeof(ZL_GraphID);
        memcpy(p, rgp->customGraphs, cgBytes);
        p += cgBytes;
    }
    if (rgp->nbCustomNodes != 0) {
        const size_t cnBytes = rgp->nbCustomNodes * sizeof(ZL_NodeID);
        memcpy(p, rgp->customNodes, cnBytes);
        p += cnBytes;
    }
    if (lpSize != 0) {
        memcpy(p, lpData, lpSize);
        p += lpSize;
    }
    ZL_ASSERT_EQ((size_t)(p - buf), total);
    *outData = buf;
    *outSize = total;
    return true;
}

typedef struct {
    uintptr_t cgraphPtr;
    uint64_t graphID;
    uint32_t formatVersion;
    int32_t compressionLevel;
    int32_t decompressionLevel;
    int32_t permissiveCompression;
    int32_t compressedChecksum;
    int32_t contentChecksum;
    int32_t storeOnExpansion;
    uint32_t minStreamSize;
    uint64_t rgpSize;
    uint64_t nbInputs;
} CodecCache_TryGraphHashHeader;

CodecCache_TryGraphLookup* CodecCache_lookupTryGraph(
        ZL_CodecOutputCache* cache,
        const ZL_Compressor* cgraph,
        ZL_GraphID graph,
        const GCParams* effectiveGCParams,
        const ZL_RuntimeGraphParameters* rgp,
        const ZL_Data* const inputs[],
        size_t nbInputs,
        Arena* scratchArena,
        size_t* cachedSizeOut,
        bool* hitOut)
{
    *hitOut = false;
    if (cache == NULL || cgraph == NULL || effectiveGCParams == NULL
        || nbInputs == 0) {
        return NULL;
    }
    for (size_t i = 0; i < nbInputs; ++i) {
        if (ZL_Data_type(inputs[i]) == ZL_Type_string) {
            return NULL;
        }
    }
    if (!CodecCache_ensureTryGraphInitialized(cache, cgraph)) {
        return NULL;
    }
    if (!CodecCache_isGraphPure(cache, cgraph, graph)) {
        return NULL;
    }

    void* rgpData  = NULL;
    size_t rgpSize = 0;
    if (!CodecCache_serializeRGP(
                cache, cgraph, rgp, scratchArena, &rgpData, &rgpSize)) {
        return NULL;
    }

    size_t inputsArrSize;
    size_t hashesArrSize;
    if (ZL_overflowMulST(nbInputs, sizeof(CodecCache_Input), &inputsArrSize)
        || ZL_overflowMulST(nbInputs, sizeof(uint64_t), &hashesArrSize)) {
        return NULL;
    }
    CodecCache_Input* const keyInputs =
            ALLOC_Arena_malloc(scratchArena, inputsArrSize);
    uint64_t* const inputHashes =
            ALLOC_Arena_malloc(scratchArena, hashesArrSize);
    CodecCache_TryGraphLookup* const lookup =
            ALLOC_Arena_malloc(scratchArena, sizeof(*lookup));
    if (keyInputs == NULL || inputHashes == NULL || lookup == NULL) {
        return NULL;
    }

    for (size_t i = 0; i < nbInputs; ++i) {
        if (!CodecCache_buildInput(
                    &keyInputs[i],
                    &inputHashes[i],
                    cache,
                    scratchArena,
                    inputs[i])) {
            return NULL;
        }
    }

    const CodecCache_TryGraphHashHeader hdr = {
        .cgraphPtr             = (uintptr_t)cgraph,
        .graphID               = (uint64_t)graph.gid,
        .formatVersion         = effectiveGCParams->formatVersion,
        .compressionLevel      = effectiveGCParams->compressionLevel,
        .decompressionLevel    = effectiveGCParams->decompressionLevel,
        .permissiveCompression = (int32_t)effectiveGCParams->permissiveCompression,
        .compressedChecksum    = (int32_t)effectiveGCParams->compressedChecksum,
        .contentChecksum       = (int32_t)effectiveGCParams->contentChecksum,
        .storeOnExpansion      = (int32_t)effectiveGCParams->storeOnExpansion,
        .minStreamSize         = (uint32_t)effectiveGCParams->minStreamSize,
        .rgpSize               = (uint64_t)rgpSize,
        .nbInputs              = (uint64_t)nbInputs,
    };
    XXH3_state_t state;
    XXH3_INITSTATE(&state);
    XXH3_64bits_reset(&state);
    XXH3_64bits_update(&state, &hdr, sizeof(hdr));
    if (rgpSize != 0) {
        XXH3_64bits_update(&state, rgpData, rgpSize);
    }
    XXH3_64bits_update(&state, inputHashes, hashesArrSize);
    const uint64_t combinedHash64 = XXH3_64bits_digest(&state);

    lookup->cache       = cache;
    lookup->inputHashes = inputHashes;
    lookup->key         = (CodecCache_TryGraphKey){
        .combinedHash64        = combinedHash64,
        .cgraph                = cgraph,
        .graphID               = graph.gid,
        .formatVersion         = hdr.formatVersion,
        .compressionLevel      = hdr.compressionLevel,
        .decompressionLevel    = hdr.decompressionLevel,
        .permissiveCompression = hdr.permissiveCompression,
        .compressedChecksum    = hdr.compressedChecksum,
        .contentChecksum       = hdr.contentChecksum,
        .storeOnExpansion      = hdr.storeOnExpansion,
        .minStreamSize         = hdr.minStreamSize,
        .rgpSize               = rgpSize,
        .rgpData               = rgpData,
        .nbInputs              = nbInputs,
        .inputs                = keyInputs,
    };

    const CodecCache_TryGraphMap_Entry* const found =
            CodecCache_TryGraphMap_find(&cache->tryGraphMap, &lookup->key);
    if (found != NULL) {
        *cachedSizeOut = found->val;
        *hitOut        = true;
        return NULL;
    }
    if (!cache->insertionsEnabled) {
        return NULL;
    }
    return lookup;
}

void CodecCache_storeTryGraph(
        const CodecCache_TryGraphLookup* lookup,
        size_t compressedSize)
{
    if (lookup == NULL) {
        return;
    }
    ZL_CodecOutputCache* const cache = lookup->cache;
    if (!cache->insertionsEnabled || !cache->tryGraphInitialized) {
        return;
    }
    const CodecCache_TryGraphKey* const key = &lookup->key;
    if (CodecCache_TryGraphMap_find(&cache->tryGraphMap, key) != NULL) {
        return;
    }

    size_t entryBytes = sizeof(CodecCache_TryGraphKey);
    size_t newBlobBytes = 0;
    if (!CodecCache_addSize(&entryBytes, key->rgpSize)
        || !CodecCache_addArraySize(
                &entryBytes, key->nbInputs, sizeof(CodecCache_Input))) {
        return;
    }

    for (size_t i = 0; i < key->nbInputs; ++i) {
        bool shared = false;
        if (key->inputs[i].contentSize != 0) {
            const CodecCache_BlobKey probe = {
                .hash64 = lookup->inputHashes[i],
                .size   = key->inputs[i].contentSize,
                .data   = key->inputs[i].content,
            };
            shared = CodecCache_BlobMap_find(&cache->blobMap, &probe) != NULL;
        }
        if (!shared
            && !CodecCache_addSize(&newBlobBytes, key->inputs[i].contentSize)) {
            return;
        }
        if (!CodecCache_addArraySize(
                    &entryBytes,
                    key->inputs[i].nbIntMetadata,
                    sizeof(Stream_IntMetadata))) {
            return;
        }
    }

    size_t newTryGraphBytes;
    size_t newBytesStored;
    size_t totalCombinedBytes;
    if (ZL_overflowAddST(
                cache->tryGraphBytesStored, entryBytes, &newTryGraphBytes)
        || ZL_overflowAddST(cache->bytesStored, newBlobBytes, &newBytesStored)
        || ZL_overflowAddST(
                newBytesStored, newTryGraphBytes, &totalCombinedBytes)
        || totalCombinedBytes > cache->maxBytes) {
        return;
    }

    Arena* const tgArena          = cache->tryGraphArena;
    CodecCache_TryGraphKey stored = *key;
    stored.rgpData = CodecCache_copyBytes(tgArena, key->rgpData, key->rgpSize);
    if (key->rgpSize != 0 && stored.rgpData == NULL) {
        return;
    }
    CodecCache_Input* const storedInputs = ALLOC_Arena_malloc(
            tgArena, key->nbInputs * sizeof(CodecCache_Input));
    if (storedInputs == NULL) {
        return;
    }
    for (size_t i = 0; i < key->nbInputs; ++i) {
        const void* sharedContent = NULL;
        if (key->inputs[i].contentSize != 0) {
            const CodecCache_BlobKey probe = {
                .hash64 = lookup->inputHashes[i],
                .size   = key->inputs[i].contentSize,
                .data   = key->inputs[i].content,
            };
            const CodecCache_BlobMap_Entry* const foundBlob =
                    CodecCache_BlobMap_find(&cache->blobMap, &probe);
            if (foundBlob != NULL) {
                sharedContent = foundBlob->key.data;
            }
        }
        bool ownsBlob = false;
        if (!CodecCache_copyInput(
                    cache,
                    tgArena,
                    &key->inputs[i],
                    sharedContent,
                    &storedInputs[i],
                    &ownsBlob)) {
            return;
        }
        if (ownsBlob && storedInputs[i].contentSize != 0) {
            CodecCache_internBlob(
                    cache,
                    lookup->inputHashes[i],
                    storedInputs[i].content,
                    storedInputs[i].contentSize);
            cache->bytesStored += storedInputs[i].contentSize;
        }
    }
    stored.inputs = storedInputs;

    const CodecCache_TryGraphMap_Entry entry = {
        .key = stored,
        .val = compressedSize,
    };
    const CodecCache_TryGraphMap_Insert ins =
            CodecCache_TryGraphMap_insert(&cache->tryGraphMap, &entry);
    if (!ins.badAlloc && ins.inserted) {
        cache->tryGraphBytesStored = newTryGraphBytes;
    }
}
