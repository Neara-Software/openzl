// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_FLOAT_MULT_SPLIT_ENCODE_BINDING_H
#define OPENZL_CODECS_FLOAT_MULT_SPLIT_ENCODE_BINDING_H

#include "openzl/codecs/float_mult_split/graph_float_mult_split.h"
#include "openzl/zl_ctransform.h"

#if defined(__cplusplus)
extern "C" {
#endif

ZL_Report EI_float_mult_split(
        ZL_Encoder* eictx,
        const ZL_Input* ins[],
        size_t nbIns);

#define EI_FLOAT_MULT_SPLIT(id)                      \
    {                                                \
        .gd          = FLOAT_MULT_SPLIT_GRAPH(id),   \
        .transform_f = EI_float_mult_split,          \
        .name        = "!zl.float_mult_split",       \
    }

#if defined(__cplusplus)
}
#endif

#endif
