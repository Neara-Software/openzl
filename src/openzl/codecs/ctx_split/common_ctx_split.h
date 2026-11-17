// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_CTX_SPLIT_COMMON_H
#define OPENZL_CODECS_CTX_SPLIT_COMMON_H

#include <stdint.h>

#include "openzl/shared/bits.h"
#include "openzl/shared/portability.h"

enum {
    CTX_SPLIT_MODE_LINEAR      = 0,
    CTX_SPLIT_MODE_LOG2        = 1,
    CTX_SPLIT_MODE_PASSTHROUGH = 2,

    CTX_SPLIT_MIN_NCTX_LOG     = 1,
    CTX_SPLIT_MAX_NCTX_LOG     = 3,
    CTX_SPLIT_DEFAULT_NCTX_LOG = 3,
    CTX_SPLIT_MAX_K            = 8,

    CTX_SPLIT_MIN_ELTS         = 256,
    CTX_SPLIT_STREAM_OVH_BYTES = 32,
    CTX_SPLIT_MIN_GAIN_BYTES   = 128,

    CTX_SPLIT_FULL_SCAN_LIMIT  = 65536,
    CTX_SPLIT_SAMPLE_BLOCKS    = 8,
    CTX_SPLIT_SAMPLE_BLOCK_LEN = 4096,
};

ZL_FORCE_INLINE unsigned ctx_bitlen64(uint64_t x)
{
    return x == 0 ? 0u : (unsigned)(64 - ZL_clz64(x));
}

ZL_FORCE_INLINE uint32_t ctx_cls_mode0(
        uint64_t prev,
        uint64_t minVal,
        unsigned shift,
        uint32_t maxCls)
{
    if (prev <= minVal) {
        return 0u;
    }
    uint64_t const d = (prev - minVal) >> shift;
    return d >= (uint64_t)maxCls ? maxCls : (uint32_t)d;
}

ZL_FORCE_INLINE uint32_t ctx_cls_mode1(uint64_t prev, uint32_t maxCls)
{
    unsigned const bl = ctx_bitlen64(prev);
    return bl >= (unsigned)maxCls ? maxCls : (uint32_t)bl;
}

#endif // OPENZL_CODECS_CTX_SPLIT_COMMON_H
