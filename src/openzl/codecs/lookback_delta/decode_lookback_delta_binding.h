// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_LOOKBACK_DELTA_DECODE_BINDING_H
#define OPENZL_CODECS_LOOKBACK_DELTA_DECODE_BINDING_H

#include "openzl/codecs/lookback_delta/graph_lookback_delta.h"
#include "openzl/zl_dtransform.h"

#if defined(__cplusplus)
extern "C" {
#endif

ZL_Report DI_lookback_delta(ZL_Decoder* dictx, const ZL_Input* ins[]);

#define DI_LOOKBACK_DELTA(id) \
    { .transform_f = DI_lookback_delta, .name = "!zl.lookback_delta" }

#ifdef __cplusplus
}
#endif

#endif
