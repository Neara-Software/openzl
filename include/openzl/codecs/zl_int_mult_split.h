// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_INT_MULT_SPLIT_H
#define OPENZL_CODECS_INT_MULT_SPLIT_H

#include "openzl/zl_nodes.h"

#if defined(__cplusplus)
extern "C" {
#endif

// int_mult_split
// Splits a numeric stream into (quotients, adjustments) using an adaptive
// per-chunk multiplier base >= 1 via triple-GCD sampling (or forced base).
// Output 0 (numeric, same width): quotients x / base (or x when base == 1).
// Output 1 (numeric, same width): adjustments x % base in [0, base - 1]
// (0 elements when base == 1).
#define ZL_NODE_INT_MULT_SPLIT ZL_MAKE_NODE_ID(ZL_StandardNodeID_int_mult_split)

// Optional int parameter: 0 = unsigned (default), 1 = signed (Euclidean floor_div)
#define ZL_INT_MULT_SPLIT_MODE_PID 0
// Optional int parameter: 0 = auto-detect per chunk (default), >= 1 = force base
#define ZL_INT_MULT_SPLIT_BASE_PID 1

typedef enum {
    ZL_IntMultSplit_unsigned = 0,
    ZL_IntMultSplit_signed   = 1,
} ZL_IntMultSplitMode;

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // OPENZL_CODECS_INT_MULT_SPLIT_H
