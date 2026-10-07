// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_MORTON3_ENCODE_MORTON3_BINDING_H
#define OPENZL_CODECS_MORTON3_ENCODE_MORTON3_BINDING_H

#include "openzl/codecs/morton3/graph_morton3.h"
#include "openzl/shared/portability.h"
#include "openzl/zl_ctransform.h"

ZL_BEGIN_C_DECLS

ZL_Report
EI_morton3_u16(ZL_Encoder* eictx, const ZL_Input* ins[], size_t nbIns);

#define EI_MORTON3_U16(id)                  \
    { .gd          = MORTON3_U16_GRAPH(id), \
      .transform_f = EI_morton3_u16,        \
      .name        = "!zl.morton3_u16" }

ZL_Report
EI_morton3_i32(ZL_Encoder* eictx, const ZL_Input* ins[], size_t nbIns);

#define EI_MORTON3_I32(id)                  \
    { .gd          = MORTON3_I32_GRAPH(id), \
      .transform_f = EI_morton3_i32,        \
      .name        = "!zl.morton3_i32" }

ZL_END_C_DECLS

#endif
