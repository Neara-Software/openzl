// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_PCO_BIN_SPLIT_ENCODE_BINDING_H
#define OPENZL_CODECS_PCO_BIN_SPLIT_ENCODE_BINDING_H

#include "openzl/codecs/pco_bin_split/graph_pco_bin_split.h"
#include "openzl/zl_ctransform.h"

#if defined(__cplusplus)
extern "C" {
#endif

ZL_Report EI_pco_bin_split(
        ZL_Encoder* eictx,
        const ZL_Input* ins[],
        size_t nbIns);

#define EI_PCO_BIN_SPLIT(id)                           {                                                      .gd          = PCO_BIN_SPLIT_GRAPH(id),            .transform_f = EI_pco_bin_split,                   .name        = "!zl.pco_bin_split",            }

#if defined(__cplusplus)
}
#endif

#endif
