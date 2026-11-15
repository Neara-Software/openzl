// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_PCO_BIN_SPLIT_H
#define OPENZL_CODECS_PCO_BIN_SPLIT_H

#include "openzl/zl_nodes.h"

#if defined(__cplusplus)
extern "C" {
#endif

// pco_bin_split
// Input  0: numeric stream of element width 1, 2, 4, or 8 bytes.
// Output 0: numeric u8 stream of bin IDs (empty when 1 bin or n == 0).
// Output 1: serial stream of tightly bit-packed per-bin offsets.
#define ZL_NODE_PCO_BIN_SPLIT ZL_MAKE_NODE_ID(ZL_StandardNodeID_pco_bin_split)

// Optional int parameter controlling total-order latent mapping.
#define ZL_PCO_BIN_SPLIT_LATENT_MODE_PID 0

typedef enum {
    ZL_PcoBinSplit_unsigned = 0,
    ZL_PcoBinSplit_signed   = 1,
    ZL_PcoBinSplit_float    = 2,
} ZL_PcoBinSplitLatentMode;

#if defined(__cplusplus)
}
#endif

#endif
