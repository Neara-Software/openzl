// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/zl_graphs.h"
#include "openzl/codecs/zl_int_mult_split.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"
#include "openzl/cpp/codecs/Node.hpp"

namespace openzl {
namespace nodes {
class IntMultSplit : public Node {
   public:
    enum class Mode : int {
        Unsigned = ZL_IntMultSplit_unsigned,
        Signed   = ZL_IntMultSplit_signed,
    };

    static constexpr NodeID node = ZL_NODE_INT_MULT_SPLIT;

    static constexpr NodeMetadata<1, 2> metadata = {
        .inputs           = { InputMetadata{ .type = Type::Numeric } },
        .singletonOutputs = { OutputMetadata{ .type = Type::Numeric,
                                              .name = "quotients" },
                              OutputMetadata{ .type = Type::Numeric,
                                              .name = "adjustments" } },
        .description =
                "Pcodec-style per-chunk adaptive integer multiplier split (quotients + adjustments)",
    };

    explicit IntMultSplit(Mode mode = Mode::Unsigned, int base = 0)
            : mode_(mode), base_(base)
    {
    }

    NodeID baseNode() const override
    {
        return node;
    }

    poly::optional<NodeParameters> parameters() const override
    {
        if (mode_ == Mode::Unsigned && base_ == 0) {
            return poly::nullopt;
        }
        NodeParameters params;
        params.localParams = LocalParams{};
        if (mode_ != Mode::Unsigned) {
            params.localParams->addIntParam(
                    ZL_INT_MULT_SPLIT_MODE_PID, static_cast<int>(mode_));
        }
        if (base_ != 0) {
            params.localParams->addIntParam(ZL_INT_MULT_SPLIT_BASE_PID, base_);
        }
        return params;
    }

    GraphID operator()(
            Compressor& compressor,
            GraphID quotients   = ZL_GRAPH_COMPRESS_GENERIC,
            GraphID adjustments = ZL_GRAPH_COMPRESS_GENERIC) const
    {
        return buildGraph(
                compressor,
                std::initializer_list<GraphID>{ quotients, adjustments });
    }

    ~IntMultSplit() override = default;

   private:
    Mode mode_;
    int base_;
};
} // namespace nodes
} // namespace openzl
