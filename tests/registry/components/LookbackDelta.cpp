// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdint>
#include <vector>

#include "openzl/cpp/codecs/LookbackDelta.hpp"
#include "tests/registry/OpenZLComponents.h"
#include "tests/registry/OpenZLInput.h"

namespace openzl::tests::components {
namespace {

template <typename T>
std::vector<T> makePeriodicPattern(size_t count, size_t period)
{
    std::vector<T> out(count);
    for (size_t i = 0; i < count; ++i) {
        out[i] = static_cast<T>((i % period) * 17 + (i / period));
    }
    return out;
}

class LookbackDeltaComponent : public OpenZLComponent {
   public:
    std::string name() const override
    {
        return "LookbackDelta";
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
            nodes::LookbackDelta{}.parameterize(compressor),
            nodes::LookbackDelta{ 4, false }.parameterize(compressor),
            nodes::LookbackDelta{ 10, true }.parameterize(compressor),
            nodes::LookbackDelta{ 16, true }.parameterize(compressor),
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
            int const wl   = static_cast<int>(gen.usize_range("wlog", 4, 16));
            bool const frc = gen.boolean("force");
            nodes.push_back(
                    nodes::LookbackDelta{ wl, frc }.parameterize(compressor));
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
                U8OpenZLInput::make(std::vector<uint8_t>{ 0, 255, 0, 255 }));
        inputs.push_back(
                U8OpenZLInput::make(makePeriodicPattern<uint8_t>(512, 13)));
        // u16
        inputs.push_back(U16OpenZLInput::make(std::vector<uint16_t>{}));
        inputs.push_back(U16OpenZLInput::make(std::vector<uint16_t>{ 7 }));
        inputs.push_back(
                U16OpenZLInput::make(
                        std::vector<uint16_t>{ 0, UINT16_MAX, 0, UINT16_MAX }));
        inputs.push_back(
                U16OpenZLInput::make(makePeriodicPattern<uint16_t>(512, 17)));
        // u32
        inputs.push_back(U32OpenZLInput::make(std::vector<uint32_t>{}));
        inputs.push_back(U32OpenZLInput::make(std::vector<uint32_t>{ 0 }));
        inputs.push_back(
                U32OpenZLInput::make(std::vector<uint32_t>{ 1, 2, 3, 4, 5 }));
        inputs.push_back(
                U32OpenZLInput::make(
                        std::vector<uint32_t>{ 0, UINT32_MAX, 0, UINT32_MAX }));
        inputs.push_back(
                U32OpenZLInput::make(makePeriodicPattern<uint32_t>(512, 19)));
        // u64
        inputs.push_back(U64OpenZLInput::make(std::vector<uint64_t>{}));
        inputs.push_back(U64OpenZLInput::make(std::vector<uint64_t>{ 123 }));
        inputs.push_back(
                U64OpenZLInput::make(
                        std::vector<uint64_t>{ 0, UINT64_MAX, 0, UINT64_MAX }));
        inputs.push_back(
                U64OpenZLInput::make(makePeriodicPattern<uint64_t>(512, 23)));
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

std::unique_ptr<OpenZLComponent> makeLookbackDeltaComponent()
{
    return std::make_unique<LookbackDeltaComponent>();
}
} // namespace openzl::tests::components
