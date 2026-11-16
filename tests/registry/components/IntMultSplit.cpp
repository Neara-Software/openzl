// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdint>
#include <limits>
#include <vector>

#include "openzl/cpp/codecs/IntMultSplit.hpp"
#include "tests/registry/OpenZLComponents.h"
#include "tests/registry/OpenZLInput.h"

namespace openzl::tests::components {
namespace {

template <typename T>
std::vector<T> makeScaledWithOutliers(size_t count, T mult, T offset)
{
    std::vector<T> out(count);
    for (size_t i = 0; i < count; ++i) {
        T base = static_cast<T>((i * 13ULL) * static_cast<uint64_t>(mult) + static_cast<uint64_t>(offset));
        if (i % 19 == 0) {
            base = static_cast<T>(base + 1);
        }
        out[i] = base;
    }
    return out;
}

class IntMultSplitComponent : public OpenZLComponent {
   public:
    std::string name() const override
    {
        return "IntMultSplit";
    }

    int minFormatVersion() const override
    {
        return ZL_MAX_FORMAT_VERSION;
    }

    size_t compressBound(poly::span<const Input> inputs) const override
    {
        return this->OpenZLComponent::compressBound(inputs) * 4 + 4096;
    }

    std::vector<NodeID> predefinedNodes(Compressor& compressor) const override
    {
        return {
            nodes::IntMultSplit{ nodes::IntMultSplit::Mode::Unsigned }
                    .parameterize(compressor),
            nodes::IntMultSplit{ nodes::IntMultSplit::Mode::Signed }
                    .parameterize(compressor),
            nodes::IntMultSplit{ nodes::IntMultSplit::Mode::Unsigned, 16 }
                    .parameterize(compressor),
            nodes::IntMultSplit{ nodes::IntMultSplit::Mode::Signed, 16 }
                    .parameterize(compressor),
            nodes::IntMultSplit{ nodes::IntMultSplit::Mode::Unsigned, 1 }
                    .parameterize(compressor),
        };
    }

    std::vector<NodeID> generateNodes(
            Compressor& compressor,
            datagen::DataGen& gen,
            size_t num) const override
    {
        std::vector<NodeID> nodes;
        nodes.reserve(num);
        for (size_t i = 0; i < num; ++i) {
            auto mode = gen.boolean("signed")
                    ? nodes::IntMultSplit::Mode::Signed
                    : nodes::IntMultSplit::Mode::Unsigned;
            int base = 0;
            size_t bc = gen.usize_range("base_choice", 0, 2);
            if (bc == 1) {
                base = 1;
            } else if (bc == 2) {
                base = 16;
            }
            nodes.push_back(nodes::IntMultSplit{ mode, base }.parameterize(
                    compressor));
        }
        return nodes;
    }

    std::vector<std::unique_ptr<OpenZLInput>> predefinedInputs() const override
    {
        std::vector<std::unique_ptr<OpenZLInput>> inputs;
        // u8
        inputs.push_back(U8OpenZLInput::make(std::vector<uint8_t>{}));
        inputs.push_back(U8OpenZLInput::make(std::vector<uint8_t>{ 42 }));
        inputs.push_back(
                U8OpenZLInput::make(std::vector<uint8_t>(200, uint8_t(7))));
        inputs.push_back(U8OpenZLInput::make(
                makeScaledWithOutliers<uint8_t>(512, 16, 3)));
        // u16
        inputs.push_back(U16OpenZLInput::make(std::vector<uint16_t>{}));
        inputs.push_back(U16OpenZLInput::make(std::vector<uint16_t>{ 9 }));
        inputs.push_back(U16OpenZLInput::make(
                std::vector<uint16_t>{ 0, UINT16_MAX, 1, 0x8000 }));
        inputs.push_back(U16OpenZLInput::make(
                makeScaledWithOutliers<uint16_t>(1024, 16, 5)));
        // u32
        inputs.push_back(U32OpenZLInput::make(std::vector<uint32_t>{}));
        inputs.push_back(U32OpenZLInput::make(std::vector<uint32_t>{ 0 }));
        inputs.push_back(
                U32OpenZLInput::make(
                        std::vector<uint32_t>{
                                0,
                                UINT32_MAX,
                                static_cast<uint32_t>(INT32_MIN),
                                static_cast<uint32_t>(INT32_MAX) }));
        inputs.push_back(U32OpenZLInput::make(
                makeScaledWithOutliers<uint32_t>(1024, 1000, 7)));
        // u64
        inputs.push_back(U64OpenZLInput::make(std::vector<uint64_t>{}));
        inputs.push_back(U64OpenZLInput::make(std::vector<uint64_t>{ 123 }));
        inputs.push_back(
                U64OpenZLInput::make(
                        std::vector<uint64_t>{
                                0,
                                UINT64_MAX,
                                static_cast<uint64_t>(INT64_MIN),
                                static_cast<uint64_t>(INT64_MAX) }));
        inputs.push_back(U64OpenZLInput::make(
                makeScaledWithOutliers<uint64_t>(1024, 1000000ULL, 11)));
        return inputs;
    }

    std::vector<std::unique_ptr<OpenZLInput>> generateInputs(
            datagen::DataGen& gen,
            size_t num,
            size_t maxInputSize,
            const Compressor&,
            GraphID) const override
    {
        std::vector<std::unique_ptr<OpenZLInput>> inputs;
        inputs.reserve(num);
        for (size_t i = 0; i < num; ++i) {
            auto widthChoice = gen.usize_range("width", 0, 3);
            switch (widthChoice) {
                case 0:
                    inputs.push_back(U8OpenZLInput::make(gen.randVector<uint8_t>(
                            "values", 0, UINT8_MAX, maxInputSize / 1)));
                    break;
                case 1:
                    inputs.push_back(
                            U16OpenZLInput::make(gen.randVector<uint16_t>(
                                    "values", 0, UINT16_MAX, maxInputSize / 2)));
                    break;
                case 2:
                    inputs.push_back(
                            U32OpenZLInput::make(gen.randVector<uint32_t>(
                                    "values", 0, UINT32_MAX, maxInputSize / 4)));
                    break;
                case 3:
                    inputs.push_back(
                            U64OpenZLInput::make(gen.randVector<uint64_t>(
                                    "values", 0, UINT64_MAX, maxInputSize / 8)));
                    break;
            }
        }
        return inputs;
    }
};

} // namespace

std::unique_ptr<OpenZLComponent> makeIntMultSplitComponent()
{
    return std::make_unique<IntMultSplitComponent>();
}

} // namespace openzl::tests::components
