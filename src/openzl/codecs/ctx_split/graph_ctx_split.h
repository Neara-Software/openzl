// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_CTX_SPLIT_GRAPH_H
#define OPENZL_CODECS_CTX_SPLIT_GRAPH_H

#include "openzl/codecs/common/graph_vo.h"
#include "openzl/zl_data.h"

#define GRAPH_CTX_SPLIT_DEC(id)                                                \
    {                                                                          \
        .CTid       = id,                                                      \
        .inputTypes = ZL_STREAMTYPELIST(ZL_Type_serial),                       \
        .soTypes    = NULL,                                                    \
        0,                                                                     \
        ZL_STREAMTYPELIST((ZL_Type)(ZL_Type_serial | ZL_Type_numeric)),        \
    }

#endif // OPENZL_CODECS_CTX_SPLIT_GRAPH_H
