// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_CTX_SPLIT_DECODE_BINDING_H
#define OPENZL_CODECS_CTX_SPLIT_DECODE_BINDING_H

#include "openzl/codecs/ctx_split/graph_ctx_split.h"
#include "openzl/shared/portability.h"
#include "openzl/zl_dtransform.h"

ZL_BEGIN_C_DECLS

ZL_Report DI_ctx_split(
        ZL_Decoder* dictx,
        const ZL_Input* compulsorySrcs[],
        size_t nbCompulsorySrcs,
        const ZL_Input* variableSrcs[],
        size_t nbVariableSrcs) ZL_NOEXCEPT_FUNC_PTR;

#define DI_CTX_SPLIT(id) \
    { .transform_f = DI_ctx_split, .name = "!zl.ctx_split" }

ZL_END_C_DECLS

#endif // OPENZL_CODECS_CTX_SPLIT_DECODE_BINDING_H
