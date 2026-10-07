// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_morton3.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"
#include "openzl/cpp/codecs/Node.hpp"

namespace openzl {
namespace nodes {
struct Morton3U16 : public SimplePipeNode<Morton3U16> {
   public:
    static constexpr NodeID node = ZL_NODE_MORTON3_U16;

    static constexpr NodeMetadata<1, 1> metadata = {
        .inputs           = { InputMetadata{ .type = Type::Struct } },
        .singletonOutputs = { OutputMetadata{ .type = Type::Numeric } },
        .description =
                "6-byte xyz u16 records to u64 gaps of their 3D Morton keys",
    };
};

struct Morton3I32 : public SimplePipeNode<Morton3I32> {
   public:
    static constexpr NodeID node = ZL_NODE_MORTON3_I32;

    static constexpr NodeMetadata<1, 1> metadata = {
        .inputs           = { InputMetadata{ .type = Type::Struct } },
        .singletonOutputs = { OutputMetadata{ .type = Type::Numeric } },
        .description =
                "12-byte xyz i32 records to u64 gaps of their 3D Morton keys",
    };
};
} // namespace nodes
} // namespace openzl
