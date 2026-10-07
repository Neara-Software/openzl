// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_HILBERT3_ENCODE_HILBERT3_BINDING_H
#define OPENZL_CODECS_HILBERT3_ENCODE_HILBERT3_BINDING_H

#include "openzl/codecs/hilbert3/graph_hilbert3.h"
#include "openzl/shared/portability.h"
#include "openzl/zl_ctransform.h"

ZL_BEGIN_C_DECLS

ZL_Report
EI_hilbert3_u16(ZL_Encoder* eictx, const ZL_Input* ins[], size_t nbIns);

#define EI_HILBERT3_U16(id)                  \
    { .gd          = HILBERT3_U16_GRAPH(id), \
      .transform_f = EI_hilbert3_u16,        \
      .name        = "!zl.hilbert3_u16" }

ZL_END_C_DECLS

#endif
