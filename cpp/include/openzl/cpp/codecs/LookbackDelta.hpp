// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_lookback_delta.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"
#include "openzl/cpp/codecs/Node.hpp"

namespace openzl {
namespace nodes {
class LookbackDelta : public Node {
   public:
    static constexpr NodeID node = ZL_NODE_LOOKBACK_DELTA;

    static constexpr NodeMetadata<1, 2> metadata = {
        .inputs           = { InputMetadata{ .type = Type::Numeric } },
        .singletonOutputs = { OutputMetadata{ .type = Type::Numeric,
                                              .name = "residuals" },
                              OutputMetadata{ .type = Type::Numeric,
                                              .name = "lookbacks" } },
        .description =
                "Adaptive lookback-delta coding (Pcodec-style hash/repeating/brute lookback with consecutive-1 fallback)",
    };

    explicit LookbackDelta(int windowLog = 15, bool forceLookback = false)
            : windowLog_(windowLog), forceLookback_(forceLookback)
    {
    }

    NodeID baseNode() const override
    {
        return node;
    }

    poly::optional<NodeParameters> parameters() const override
    {
        if (windowLog_ == 15 && !forceLookback_) {
            return poly::nullopt;
        }
        NodeParameters params;
        params.localParams = LocalParams{};
        if (windowLog_ != 15) {
            params.localParams->addIntParam(
                    ZL_LOOKBACK_DELTA_WINDOW_LOG_PID, windowLog_);
        }
        if (forceLookback_) {
            params.localParams->addIntParam(ZL_LOOKBACK_DELTA_FORCE_PID, 1);
        }
        return params;
    }

    GraphID operator()(
            Compressor& compressor,
            GraphID residuals,
            GraphID lookbacks) const
    {
        return buildGraph(
                compressor,
                std::initializer_list<GraphID>{ residuals, lookbacks });
    }

    ~LookbackDelta() override = default;

   private:
    int windowLog_;
    bool forceLookback_;
};
} // namespace nodes
} // namespace openzl
