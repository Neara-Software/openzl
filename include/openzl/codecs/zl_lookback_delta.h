// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_LOOKBACK_DELTA_H
#define OPENZL_CODECS_LOOKBACK_DELTA_H

#include "openzl/zl_nodes.h"

#if defined(__cplusplus)
extern "C" {
#endif

// lookback_delta
// Input  0: numeric stream of element width 1, 2, 4, or 8 bytes.
// Output 0: numeric stream of wrapping residuals (same width as input).
// Output 1: numeric u16 stream of (lookback - 1) distances (empty when raw or
//           consecutive delta-1 is selected).
#define ZL_NODE_LOOKBACK_DELTA ZL_MAKE_NODE_ID(ZL_StandardNodeID_lookback_delta)

// Optional int parameter: log2 of lookback window (default 15 -> 32768, range 4..16).
#define ZL_LOOKBACK_DELTA_WINDOW_LOG_PID 0
// Optional int parameter: if non-zero, always emit lookbacks for N > 16.
#define ZL_LOOKBACK_DELTA_FORCE_PID 1

#if defined(__cplusplus)
}
#endif

#endif
