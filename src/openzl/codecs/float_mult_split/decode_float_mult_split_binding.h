// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_FLOAT_MULT_SPLIT_DECODE_BINDING_H
#define OPENZL_CODECS_FLOAT_MULT_SPLIT_DECODE_BINDING_H

#include "openzl/codecs/float_mult_split/graph_float_mult_split.h"
#include "openzl/zl_dtransform.h"

#if defined(__cplusplus)
extern "C" {
#endif

ZL_Report DI_float_mult_split(ZL_Decoder* dictx, const ZL_Input* ins[]);

#define DI_FLOAT_MULT_SPLIT(id)     { .transform_f = DI_float_mult_split, .name = "!zl.float_mult_split" }

#if defined(__cplusplus)
}
#endif

#endif
