// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_MORTON3_GRAPH_MORTON3_H
#define OPENZL_CODECS_MORTON3_GRAPH_MORTON3_H

#include "openzl/zl_data.h"

/* Input 0: struct stream, 6-byte records (x, y, z as UInt16LE)
 * Output 0: numeric u64 stream of 3D Morton key gaps */
#define MORTON3_U16_GRAPH(id)                             \
    {                                                     \
        .CTid       = id,                                 \
        .inputTypes = ZL_STREAMTYPELIST(ZL_Type_struct),  \
        .soTypes    = ZL_STREAMTYPELIST(ZL_Type_numeric), \
    }

/* Input 0: struct stream, 12-byte records (x, y, z as Int32LE)
 * Output 0: numeric u64 stream of 96-bit 3D Morton key gaps with escapes */
#define MORTON3_I32_GRAPH(id) MORTON3_U16_GRAPH(id)

#endif
