// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/zl_graphs.h"
#include "openzl/codecs/zl_float_mult_split.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"
#include "openzl/cpp/codecs/Node.hpp"

namespace openzl {
namespace nodes {
class FloatMultSplit : public Node {
   public:
    static constexpr NodeID node = ZL_NODE_FLOAT_MULT_SPLIT;

    static constexpr NodeMetadata<1, 2> metadata = {
        .inputs           = { InputMetadata{ .type = Type::Numeric } },
        .singletonOutputs = { OutputMetadata{ .type = Type::Numeric,
                                              .name = "multipliers" },
                              OutputMetadata{ .type = Type::Numeric,
                                              .name = "adjustments" } },
        .description =
                "Pcodec-style per-chunk adaptive floating-point multiplier split (multipliers + ULP adjustments)",
    };

    FloatMultSplit() = default;

    NodeID baseNode() const override
    {
        return node;
    }

    GraphID operator()(
            Compressor& compressor,
            GraphID multipliers = ZL_GRAPH_COMPRESS_GENERIC,
            GraphID adjustments = ZL_GRAPH_COMPRESS_GENERIC) const
    {
        return buildGraph(
                compressor,
                std::initializer_list<GraphID>{ multipliers, adjustments });
    }

    ~FloatMultSplit() override = default;
};
} // namespace nodes
} // namespace openzl
