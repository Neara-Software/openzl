// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_CTX_SPLIT_H
#define OPENZL_CODECS_CTX_SPLIT_H

#include "openzl/zl_nodes.h"

#if defined(__cplusplus)
extern "C" {
#endif

// ctx_split
// Order-1 predecessor context-splitting codec.
// Partitions elements into up to K = (1 << nCtxLog) context sub-streams
// selected solely from the immediately preceding element (prev = 0 at index 0),
// or passes through as a single stream when splitting does not save entropy.
//
// ZL_NODE_CTX_SPLIT     : Serial input (1-byte elements) -> 1..K Serial outputs
// ZL_NODE_CTX_SPLIT_NUM : Numeric input (1, 2, 4, 8 B)   -> 1..K Numeric outputs
#define ZL_NODE_CTX_SPLIT ZL_MAKE_NODE_ID(ZL_StandardNodeID_ctx_split)
#define ZL_NODE_CTX_SPLIT_NUM ZL_MAKE_NODE_ID(ZL_StandardNodeID_ctx_split_num)

// Optional int parameter: log2 of number of context classes (1..3 -> 2/4/8, default 3).
#define ZL_CTX_SPLIT_NCTX_LOG_PID 0

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // OPENZL_CODECS_CTX_SPLIT_H
