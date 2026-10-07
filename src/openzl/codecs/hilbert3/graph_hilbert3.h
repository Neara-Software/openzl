// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_HILBERT3_GRAPH_HILBERT3_H
#define OPENZL_CODECS_HILBERT3_GRAPH_HILBERT3_H

#include "openzl/zl_data.h"

/* Input 0: struct stream, 6-byte records (x, y, z as UInt16LE)
 * Output 0: numeric u64 stream of 3D Hilbert key gaps */
#define HILBERT3_U16_GRAPH(id)                            \
    {                                                     \
        .CTid       = id,                                 \
        .inputTypes = ZL_STREAMTYPELIST(ZL_Type_struct),  \
        .soTypes    = ZL_STREAMTYPELIST(ZL_Type_numeric), \
    }

#endif
