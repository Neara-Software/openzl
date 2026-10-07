// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_HILBERT3_DECODE_HILBERT3_BINDING_H
#define OPENZL_CODECS_HILBERT3_DECODE_HILBERT3_BINDING_H

#include "openzl/codecs/hilbert3/graph_hilbert3.h"
#include "openzl/shared/portability.h"
#include "openzl/zl_dtransform.h"

ZL_BEGIN_C_DECLS

ZL_Report DI_hilbert3_u16(ZL_Decoder* dictx, const ZL_Input* ins[]);

#define DI_HILBERT3_U16(id) \
    { .transform_f = DI_hilbert3_u16, .name = "hilbert3_u16" }

ZL_END_C_DECLS

#endif
