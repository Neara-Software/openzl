// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_FLOAT_MULT_SPLIT_H
#define OPENZL_CODECS_FLOAT_MULT_SPLIT_H

#include "openzl/zl_nodes.h"

#if defined(__cplusplus)
extern "C" {
#endif

// float_mult_split
// Pcodec-style per-chunk adaptive floating-point multiplier split (f32 / f64).
// Splits a 4-byte or 8-byte numeric stream into (multipliers, adjustments)
// using an adaptive per-chunk multiplier base (via trailing-zero and
// Euclidean GCD sampling).
// Output 0 (numeric, same width): signed integer multipliers (int32_t / int64_t
// stored in uint32_t / uint64_t) around x / base, or raw float bits when
// inactive.
// Output 1 (numeric, same width): ULP adjustments (uint32_t / uint64_t,
// two's-complement difference ordered(x) - ordered(mult * base));
// 0 elements when inactive, n elements when active.
#define ZL_NODE_FLOAT_MULT_SPLIT \
    ZL_MAKE_NODE_ID(ZL_StandardNodeID_float_mult_split)

#if defined(__cplusplus)
} // extern "C"
#endif

#endif // OPENZL_CODECS_FLOAT_MULT_SPLIT_H
