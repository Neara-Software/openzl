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

#if defined(__cplusplus)
}
#endif

#endif
