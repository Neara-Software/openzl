// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_HILBERT3_H
#define OPENZL_CODECS_HILBERT3_H

#include "openzl/zl_nodes.h"

#if defined(__cplusplus)
extern "C" {
#endif

// hilbert3_u16
// Input : 1 struct stream of 6-byte records: x, y, z as UInt16LE.
// Output : 1 numeric u64 stream: g[0] = k[0], g[i] = k[i] - k[i-1]
//          (wrapping), where k is the 16-level 3D Hilbert key of (x, y, z).
//          Input sorted by key gives small non-negative gaps.
#define ZL_NODE_HILBERT3_U16 ZL_MAKE_NODE_ID(ZL_StandardNodeID_hilbert3_u16)

// hilbert3_i32
// Input : 1 struct stream of 12-byte records: x, y, z as Int32LE.
// Output : 1 numeric u64 stream of gaps k[i] - k[i-1] of the 96-bit
//          32-level 3D Hilbert key of (x ^ 2^31, y ^ 2^31, z ^ 2^31).
//          The first key, and any key that goes down or jumps by
//          >= 2^64 - 1, is written as UINT64_MAX, hi64, lo64 of the key.
#define ZL_NODE_HILBERT3_I32 ZL_MAKE_NODE_ID(ZL_StandardNodeID_hilbert3_i32)

#if defined(__cplusplus)
}
#endif

#endif
