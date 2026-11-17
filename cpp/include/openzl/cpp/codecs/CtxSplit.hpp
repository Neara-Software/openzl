// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_ctx_split.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"
#include "openzl/cpp/codecs/Node.hpp"

namespace openzl {
namespace nodes {

class CtxSplit : public SimplePipeNode<CtxSplit> {
   public:
    static constexpr NodeID node = ZL_NODE_CTX_SPLIT;

    static constexpr NodeMetadata<1, 0, 1> metadata = {
        .inputs          = { InputMetadata{ .type = Type::Serial } },
        .variableOutputs = { OutputMetadata{ .type = Type::Serial,
                                             .name = "ctx_streams" } },
        .description =
                "Split serial byte stream into up to 2^nCtxLog order-1 predecessor context streams",
    };

    explicit CtxSplit(int nCtxLog = 3)
            : nCtxLog_(nCtxLog < 1 ? 1 : (nCtxLog > 3 ? 3 : nCtxLog))
    {
    }

    poly::optional<NodeParameters> parameters() const override
    {
        if (nCtxLog_ == 3) {
            return poly::nullopt;
        }
        NodeParameters params;
        params.localParams = LocalParams{};
        params.localParams->addIntParam(ZL_CTX_SPLIT_NCTX_LOG_PID, nCtxLog_);
        return params;
    }

    ~CtxSplit() override = default;

   private:
    int nCtxLog_;
};

class CtxSplitNum : public SimplePipeNode<CtxSplitNum> {
   public:
    static constexpr NodeID node = ZL_NODE_CTX_SPLIT_NUM;

    static constexpr NodeMetadata<1, 0, 1> metadata = {
        .inputs          = { InputMetadata{ .type = Type::Numeric } },
        .variableOutputs = { OutputMetadata{ .type = Type::Numeric,
                                             .name = "ctx_streams" } },
        .description =
                "Split numeric stream into up to 2^nCtxLog order-1 predecessor context streams",
    };

    explicit CtxSplitNum(int nCtxLog = 3)
            : nCtxLog_(nCtxLog < 1 ? 1 : (nCtxLog > 3 ? 3 : nCtxLog))
    {
    }

    poly::optional<NodeParameters> parameters() const override
    {
        if (nCtxLog_ == 3) {
            return poly::nullopt;
        }
        NodeParameters params;
        params.localParams = LocalParams{};
        params.localParams->addIntParam(ZL_CTX_SPLIT_NCTX_LOG_PID, nCtxLog_);
        return params;
    }

    ~CtxSplitNum() override = default;

   private:
    int nCtxLog_;
};

} // namespace nodes
} // namespace openzl
