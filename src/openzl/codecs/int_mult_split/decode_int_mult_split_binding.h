// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_INT_MULT_SPLIT_DECODE_BINDING_H
#define OPENZL_CODECS_INT_MULT_SPLIT_DECODE_BINDING_H

#include "openzl/codecs/int_mult_split/graph_int_mult_split.h"
#include "openzl/zl_dtransform.h"

#if defined(__cplusplus)
extern "C" {
#endif

ZL_Report DI_int_mult_split(ZL_Decoder* dictx, const ZL_Input* ins[]);

#define DI_INT_MULT_SPLIT(id) \
    { .transform_f = DI_int_mult_split, .name = "!zl.int_mult_split" }

#if defined(__cplusplus)
}
#endif

#endif
