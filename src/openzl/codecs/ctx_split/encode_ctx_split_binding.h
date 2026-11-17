// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_CTX_SPLIT_ENCODE_BINDING_H
#define OPENZL_CODECS_CTX_SPLIT_ENCODE_BINDING_H

#include "openzl/codecs/common/graph_vo.h"
#include "openzl/codecs/ctx_split/graph_ctx_split.h"
#include "openzl/shared/portability.h"
#include "openzl/zl_ctransform.h"

ZL_BEGIN_C_DECLS

ZL_Report
EI_ctx_split(ZL_Encoder* eictx, const ZL_Input* ins[], size_t nbIns);

#define EI_CTX_SPLIT(id)                  \
    { .gd          = GRAPH_VO_SERIAL(id), \
      .transform_f = EI_ctx_split,        \
      .name        = "!zl.ctx_split" }

#define EI_CTX_SPLIT_NUM(id)              \
    { .gd          = GRAPH_VO_NUM(id),    \
      .transform_f = EI_ctx_split,        \
      .name        = "!zl.ctx_split_num" }

ZL_END_C_DECLS

#endif // OPENZL_CODECS_CTX_SPLIT_ENCODE_BINDING_H
