// Copyright (c) Meta Platforms, Inc. and affiliates.

#ifndef OPENZL_CODECS_PCO_BIN_SPLIT_DECODE_BINDING_H
#define OPENZL_CODECS_PCO_BIN_SPLIT_DECODE_BINDING_H

#include "openzl/codecs/pco_bin_split/graph_pco_bin_split.h"
#include "openzl/zl_dtransform.h"

#if defined(__cplusplus)
extern "C" {
#endif

ZL_Report DI_pco_bin_split(ZL_Decoder* dictx, const ZL_Input* ins[]);

#define DI_PCO_BIN_SPLIT(id) \
    { .transform_f = DI_pco_bin_split, .name = "!zl.pco_bin_split" }

#ifdef __cplusplus
}
#endif

#endif
