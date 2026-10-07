// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_MORTON3_DECODE_MORTON3_BINDING_H
#define OPENZL_CODECS_MORTON3_DECODE_MORTON3_BINDING_H

#include "openzl/codecs/morton3/graph_morton3.h"
#include "openzl/shared/portability.h"
#include "openzl/zl_dtransform.h"

ZL_BEGIN_C_DECLS

ZL_Report DI_morton3_u16(ZL_Decoder* dictx, const ZL_Input* ins[]);

#define DI_MORTON3_U16(id) \
    { .transform_f = DI_morton3_u16, .name = "morton3_u16" }

ZL_Report DI_morton3_i32(ZL_Decoder* dictx, const ZL_Input* ins[]);

#define DI_MORTON3_I32(id) \
    { .transform_f = DI_morton3_i32, .name = "morton3_i32" }

ZL_END_C_DECLS

#endif
