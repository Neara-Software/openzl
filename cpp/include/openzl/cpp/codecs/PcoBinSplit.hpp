// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_entropy.h"
#include "openzl/codecs/zl_pco_bin_split.h"
#include "openzl/codecs/zl_store.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"
#include "openzl/cpp/codecs/Node.hpp"

namespace openzl {
namespace nodes {
class PcoBinSplit : public Node {
   public:
    enum class LatentMode : int {
        Unsigned = ZL_PcoBinSplit_unsigned,
        Signed   = ZL_PcoBinSplit_signed,
        Float    = ZL_PcoBinSplit_float,
    };

    static constexpr NodeID node = ZL_NODE_PCO_BIN_SPLIT;

    static constexpr NodeMetadata<1, 2> metadata = {
        .inputs           = { InputMetadata{ .type = Type::Numeric } },
        .singletonOutputs = { OutputMetadata{ .type = Type::Numeric,
                                              .name = "bin_ids" },
                              OutputMetadata{ .type = Type::Serial,
                                              .name = "offsets" } },
        .description =
                "Pcodec-style equi-depth histogram + Shannon-entropy DP bin splitting with bit-packed offsets",
    };

    explicit PcoBinSplit(LatentMode mode = LatentMode::Unsigned) : mode_(mode)
    {
    }

    NodeID baseNode() const override
    {
        return node;
    }

    poly::optional<NodeParameters> parameters() const override
    {
        if (mode_ == LatentMode::Unsigned) {
            return poly::nullopt;
        }
        NodeParameters params;
        params.localParams = LocalParams{};
        params.localParams->addIntParam(
                ZL_PCO_BIN_SPLIT_LATENT_MODE_PID, static_cast<int>(mode_));
        return params;
    }

    GraphID operator()(
            Compressor& compressor,
            GraphID binIds  = ZL_GRAPH_ENTROPY,
            GraphID offsets = ZL_GRAPH_STORE) const
    {
        return buildGraph(
                compressor, std::initializer_list<GraphID>{ binIds, offsets });
    }

    ~PcoBinSplit() override = default;

   private:
    LatentMode mode_;
};
} // namespace nodes
} // namespace openzl
