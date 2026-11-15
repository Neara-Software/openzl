// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_LOOKBACK_DELTA_ENCODE_BINDING_H
#define OPENZL_CODECS_LOOKBACK_DELTA_ENCODE_BINDING_H

#include "openzl/codecs/lookback_delta/graph_lookback_delta.h"
#include "openzl/zl_ctransform.h"

#if defined(__cplusplus)
extern "C" {
#endif

ZL_Report EI_lookback_delta(
        ZL_Encoder* eictx,
        const ZL_Input* ins[],
        size_t nbIns);

#define EI_LOOKBACK_DELTA(id)                           {                                                       .gd          = LOOKBACK_DELTA_GRAPH(id),            .transform_f = EI_lookback_delta,                   .name        = "!zl.lookback_delta",            }

#if defined(__cplusplus)
}
#endif

#endif
