// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_MORTON3_H
#define OPENZL_CODECS_MORTON3_H

#include "openzl/zl_nodes.h"

#if defined(__cplusplus)
extern "C" {
#endif

// morton3_u16
// Input : 1 struct stream of 6-byte records: x, y, z as UInt16LE.
// Output : 1 numeric u64 stream: g[0] = k[0], g[i] = k[i] - k[i-1]
//          (wrapping), where k is the 48-bit 3D Morton key of (x, y, z)
//          (bit i of x, y, z at key bits 3i, 3i + 1, 3i + 2).
//          Input sorted by key gives small non-negative gaps.
#define ZL_NODE_MORTON3_U16 ZL_MAKE_NODE_ID(ZL_StandardNodeID_morton3_u16)

// morton3_i32
// Input : 1 struct stream of 12-byte records: x, y, z as Int32LE.
// Output : 1 numeric u64 stream of gaps k[i] - k[i-1] of the 96-bit
//          3D Morton key of (x ^ 2^31, y ^ 2^31, z ^ 2^31).
//          The first key, and any key that goes down or jumps by
//          >= 2^64 - 1, is written as UINT64_MAX, hi32, lo64 of the key.
#define ZL_NODE_MORTON3_I32 ZL_MAKE_NODE_ID(ZL_StandardNodeID_morton3_i32)

#if defined(__cplusplus)
}
#endif

#endif
