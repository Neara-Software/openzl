// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "openzl/codecs/zl_hilbert3.h"
#include "openzl/cpp/Compressor.hpp"
#include "openzl/cpp/codecs/Metadata.hpp"
#include "openzl/cpp/codecs/Node.hpp"

namespace openzl {
namespace nodes {
struct Hilbert3U16 : public SimplePipeNode<Hilbert3U16> {
   public:
    static constexpr NodeID node = ZL_NODE_HILBERT3_U16;

    static constexpr NodeMetadata<1, 1> metadata = {
        .inputs           = { InputMetadata{ .type = Type::Struct } },
        .singletonOutputs = { OutputMetadata{ .type = Type::Numeric } },
        .description =
                "6-byte xyz u16 records to u64 gaps of their 3D Hilbert keys",
    };
};

struct Hilbert3I32 : public SimplePipeNode<Hilbert3I32> {
   public:
    static constexpr NodeID node = ZL_NODE_HILBERT3_I32;

    static constexpr NodeMetadata<1, 1> metadata = {
        .inputs           = { InputMetadata{ .type = Type::Struct } },
        .singletonOutputs = { OutputMetadata{ .type = Type::Numeric } },
        .description =
                "12-byte xyz i32 records to u64 gaps of their 3D Hilbert keys",
    };
};
} // namespace nodes
} // namespace openzl
