// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_PCO_BIN_SPLIT_GRAPH_PCO_BIN_SPLIT_H
#define OPENZL_CODECS_PCO_BIN_SPLIT_GRAPH_PCO_BIN_SPLIT_H

#include "openzl/zl_data.h"

#define PCO_BIN_SPLIT_GRAPH(id)                                              {                                                                            .CTid       = id,                                                        .inputTypes = ZL_STREAMTYPELIST(ZL_Type_numeric),                        .soTypes    = ZL_STREAMTYPELIST(ZL_Type_numeric, ZL_Type_serial),     }

#endif
