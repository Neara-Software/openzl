// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdint>
#include <limits>
#include <vector>

#include "openzl/cpp/codecs/PcoBinSplit.hpp"
#include "tests/registry/OpenZLComponents.h"
#include "tests/registry/OpenZLInput.h"

namespace openzl::tests::components {
namespace {

template <typename T>
std::vector<T> makeMultiModal(size_t count)
{
    std::vector<T> out(count);
    for (size_t i = 0; i < count; ++i) {
        if ((i & 3) == 0) {
            out[i] = static_cast<T>(i);
        } else {
            out[i] = static_cast<T>(1000000ULL + (i * 13ULL));
        }
    }
    return out;
}

class PcoBinSplitComponent : public OpenZLComponent {
   public:
    std::string name() const override
    {
        return "PcoBinSplit";
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
            nodes::PcoBinSplit{ nodes::PcoBinSplit::LatentMode::Unsigned }
                    .parameterize(compressor),
            nodes::PcoBinSplit{ nodes::PcoBinSplit::LatentMode::Signed }
                    .parameterize(compressor),
        };
    }

    std::vector<std::unique_ptr<OpenZLInput>> predefinedInputs() const override
    {
        std::vector<std::unique_ptr<OpenZLInput>> inputs;
        // u8
        inputs.push_back(U8OpenZLInput::make(std::vector<uint8_t>{}));
        inputs.push_back(U8OpenZLInput::make(std::vector<uint8_t>{ 42 }));
        inputs.push_back(
                U8OpenZLInput::make(std::vector<uint8_t>(200, uint8_t(7))));
        inputs.push_back(
                U8OpenZLInput::make(std::vector<uint8_t>{ 0, 255, 1, 254 }));
        // u16
        inputs.push_back(U16OpenZLInput::make(std::vector<uint16_t>{}));
        inputs.push_back(U16OpenZLInput::make(std::vector<uint16_t>{ 9 }));
        inputs.push_back(
                U16OpenZLInput::make(
                        std::vector<uint16_t>{ 0, UINT16_MAX, 1, 0x8000 }));
        inputs.push_back(
                U16OpenZLInput::make(makeMultiModal<uint16_t>(1024)));
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
        inputs.push_back(
                U32OpenZLInput::make(makeMultiModal<uint32_t>(1024)));
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
        inputs.push_back(
                U64OpenZLInput::make(makeMultiModal<uint64_t>(1024)));
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

class PcoBinSplitFloatComponent : public OpenZLComponent {
   public:
    std::string name() const override
    {
        return "PcoBinSplitFloat";
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
        return { nodes::PcoBinSplit{ nodes::PcoBinSplit::LatentMode::Float }
                         .parameterize(compressor) };
    }

    std::vector<std::unique_ptr<OpenZLInput>> predefinedInputs() const override
    {
        std::vector<std::unique_ptr<OpenZLInput>> inputs;
        // fp32
        inputs.push_back(F32OpenZLInput::make(std::vector<float>{}));
        inputs.push_back(F32OpenZLInput::make(std::vector<float>{ 0.0f, -0.0f }));
        inputs.push_back(
                F32OpenZLInput::make(
                        std::vector<float>{
                                std::numeric_limits<float>::quiet_NaN(),
                                std::numeric_limits<float>::infinity(),
                                -std::numeric_limits<float>::infinity(),
                                std::numeric_limits<float>::lowest(),
                                std::numeric_limits<float>::max(),
                                std::numeric_limits<float>::denorm_min() }));
        // fp64
        inputs.push_back(F64OpenZLInput::make(std::vector<double>{}));
        inputs.push_back(F64OpenZLInput::make(std::vector<double>{ 0.0, -0.0 }));
        inputs.push_back(
                F64OpenZLInput::make(
                        std::vector<double>{
                                std::numeric_limits<double>::quiet_NaN(),
                                std::numeric_limits<double>::infinity(),
                                -std::numeric_limits<double>::infinity(),
                                std::numeric_limits<double>::lowest(),
                                std::numeric_limits<double>::max(),
                                std::numeric_limits<double>::denorm_min() }));
        // fp16 bit patterns
        inputs.push_back(U16OpenZLInput::make(std::vector<uint16_t>{}));
        inputs.push_back(
                U16OpenZLInput::make(
                        std::vector<uint16_t>{
                                0x0000, 0x8000, 0x7E00, 0x7C00, 0xFC00, 0x0400, 0x7BFF, 0x0001 }));
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
            auto widthChoice = gen.usize_range("width", 0, 2);
            switch (widthChoice) {
                case 0:
                    inputs.push_back(
                            U16OpenZLInput::make(gen.randVector<uint16_t>(
                                    "values", 0, UINT16_MAX, maxInputSize / 2)));
                    break;
                case 1:
                    inputs.push_back(
                            U32OpenZLInput::make(gen.randVector<uint32_t>(
                                    "values", 0, UINT32_MAX, maxInputSize / 4)));
                    break;
                case 2:
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

std::unique_ptr<OpenZLComponent> makePcoBinSplitComponent()
{
    return std::make_unique<PcoBinSplitComponent>();
}

std::unique_ptr<OpenZLComponent> makePcoBinSplitFloatComponent()
{
    return std::make_unique<PcoBinSplitFloatComponent>();
}
} // namespace openzl::tests::components
