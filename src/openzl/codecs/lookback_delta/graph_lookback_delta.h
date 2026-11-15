// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_LOOKBACK_DELTA_GRAPH_LOOKBACK_DELTA_H
#define OPENZL_CODECS_LOOKBACK_DELTA_GRAPH_LOOKBACK_DELTA_H

#include "openzl/zl_data.h"

#define LOOKBACK_DELTA_GRAPH(id)                                               {                                                                              .CTid       = id,                                                          .inputTypes = ZL_STREAMTYPELIST(ZL_Type_numeric),                          .soTypes    = ZL_STREAMTYPELIST(ZL_Type_numeric, ZL_Type_numeric),     }

#endif
