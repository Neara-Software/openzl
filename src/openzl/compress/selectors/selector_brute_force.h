// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef ZSTRONG_COMPRESS_SELECTORS_SELECTOR_BRUTE_FORCE_H
#define ZSTRONG_COMPRESS_SELECTORS_SELECTOR_BRUTE_FORCE_H

#include "openzl/shared/portability.h"
#include "openzl/zl_selector.h"

ZL_BEGIN_C_DECLS

/// Optional int local params of the graph running the brute force selector.
/// SLACK (units of 1e-4): successors are listed fastest-decoding first, so take
/// the first one whose size is within (1 + slack) of the smallest.
/// LAMBDA (bytes per ms of decode time) with RATE_PARAM_BASE + i (successor i's
/// decode cost in ns per KiB of input): take argmin size + lambda * time.
/// LAMBDA takes precedence over SLACK.
#define ZL_BRUTE_FORCE_SLACK_PARAM 7311
#define ZL_BRUTE_FORCE_LAMBDA_PARAM 7312
/// PROBE (bytes): first try every successor on a contiguous slice of about
/// this many bytes from the middle of the input, then fully try only the
/// PROBE_TOPK (default 1) best of them. Inputs smaller than 4 * PROBE, and
/// string inputs, are tried in full.
#define ZL_BRUTE_FORCE_PROBE_PARAM 7313
#define ZL_BRUTE_FORCE_PROBE_TOPK_PARAM 7314
#define ZL_BRUTE_FORCE_RATE_PARAM_BASE 7320
/// NARROW_PARAM_BASE + s (s < NARROW_MAX_STAGES): narrow the successors tried
/// across the inputs this selector sees within one CCtx session (e.g. the
/// frames of one file). After the (s+1)-th distinct input of at least
/// NARROW_MIN_INPUT bytes, keep only the given number of successors with the
/// smallest total size over the inputs seen so far; once the schedule ends the
/// kept set is final. Smaller inputs try every successor and are not counted.
/// Reset with ZL_CCtx_resetBruteForceNarrowing(). Encoder-only.
#define ZL_BRUTE_FORCE_NARROW_PARAM_BASE 7300
#define ZL_BRUTE_FORCE_NARROW_MAX_STAGES 10
#define ZL_BRUTE_FORCE_NARROW_MIN_INPUT 1024

ZL_GraphID SI_selector_brute_force(
        const ZL_Selector* selCtx,
        const ZL_Input* inputStream,
        const ZL_GraphID* customGraphs,
        size_t nbCustomGraphs);

ZL_END_C_DECLS

#endif
