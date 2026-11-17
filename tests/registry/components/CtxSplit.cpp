// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "openzl/cpp/codecs/CtxSplit.hpp"
#include "tests/registry/OpenZLComponents.h"
#include "tests/registry/OpenZLInput.h"

namespace openzl::tests::components {
namespace {

std::string makeMarkovSerial(size_t count)
{
    std::string out(count, '\0');
    uint64_t rng  = 0x9E3779B97F4A7C15ULL;
    uint8_t state = 0;
    for (size_t i = 0; i < count; ++i) {
        rng          = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        uint8_t base = static_cast<uint8_t>((state & 7u) * 32u);
        uint8_t off  = static_cast<uint8_t>((rng >> 56) & 3u);
        uint8_t val  = static_cast<uint8_t>(base + off);
        out[i]       = static_cast<char>(val);
        state        = static_cast<uint8_t>(((rng >> 48) & 7u) == 0u
                                                    ? ((state + 3u) & 7u)
                                                    : (val >> 5));
    }
    return out;
}

template <typename T>
std::vector<T> makeMarkovNumericLinear(size_t count, T minOffset, T step)
{
    std::vector<T> out(count);
    uint64_t rng = 0x123456789ABCDEF0ULL;
    uint32_t cls = 0;
    for (size_t i = 0; i < count; ++i) {
        rng   = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        T val = static_cast<T>(
                static_cast<uint64_t>(minOffset)
                + static_cast<uint64_t>(cls) * static_cast<uint64_t>(step)
                + ((rng >> 60) & 3ULL));
        out[i] = val;
        cls    = ((rng >> 52) & 3ULL) == 0ULL ? ((cls + 5u) & 7u) : ((cls + 1u) & 7u);
    }
    return out;
}

template <typename T>
std::vector<T> makeMarkovNumericLog2(size_t count)
{
    std::vector<T> out(count);
    uint64_t rng = 0xCAFEBABE12345678ULL;
    uint32_t mag = 0;
    for (size_t i = 0; i < count; ++i) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        if (i == count - 1) {
            out[i] = std::numeric_limits<T>::max();
            break;
        }
        uint64_t v = (mag == 0) ? 0ULL : ((1ULL << (mag - 1u)) | ((rng >> 62) & ((1ULL << (mag - 1u)) - 1ULL)));
        out[i]     = static_cast<T>(v);
        mag        = ((rng >> 50) & 3ULL) == 0ULL ? static_cast<uint32_t>((rng >> 40) % 7ULL) : mag;
    }
    return out;
}

class CtxSplitComponent : public OpenZLComponent {
   public:
    std::string name() const override
    {
        return "CtxSplit";
    }

    int minFormatVersion() const override
    {
        return 27;
    }

    size_t compressBound(poly::span<const Input> inputs) const override
    {
        return this->OpenZLComponent::compressBound(inputs) * 2 + 4096;
    }

    std::vector<NodeID> predefinedNodes(Compressor& compressor) const override
    {
        return {
            nodes::CtxSplit{}.parameterize(compressor),
            nodes::CtxSplit{ 1 }.parameterize(compressor),
            nodes::CtxSplit{ 2 }.parameterize(compressor),
            nodes::CtxSplit{ 3 }.parameterize(compressor),
        };
    }

    std::vector<NodeID> generateNodes(
            Compressor& compressor,
            datagen::DataGen& gen,
            size_t num) const override
    {
        std::vector<NodeID> out;
        out.reserve(num);
        for (size_t i = 0; i < num; ++i) {
            int nCtxLog = static_cast<int>(gen.usize_range("nCtxLog", 1, 3));
            out.push_back(nodes::CtxSplit{ nCtxLog }.parameterize(compressor));
        }
        return out;
    }

    std::vector<std::unique_ptr<OpenZLInput>> predefinedInputs() const override
    {
        std::vector<std::unique_ptr<OpenZLInput>> inputs;
        inputs.push_back(SerialOpenZLInput::make(std::string{}));
        inputs.push_back(SerialOpenZLInput::make(std::string(1, 'x')));
        inputs.push_back(SerialOpenZLInput::make(std::string(255, 'a')));
        inputs.push_back(SerialOpenZLInput::make(std::string(512, 'z')));
        inputs.push_back(SerialOpenZLInput::make(makeMarkovSerial(1024)));
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
            inputs.push_back(SerialOpenZLInput::make(
                    gen.randString("serial", maxInputSize)));
        }
        return inputs;
    }
};

class CtxSplitNumComponent : public OpenZLComponent {
   public:
    std::string name() const override
    {
        return "CtxSplitNum";
    }

    int minFormatVersion() const override
    {
        return 27;
    }

    size_t compressBound(poly::span<const Input> inputs) const override
    {
        return this->OpenZLComponent::compressBound(inputs) * 2 + 4096;
    }

    std::vector<NodeID> predefinedNodes(Compressor& compressor) const override
    {
        return {
            nodes::CtxSplitNum{}.parameterize(compressor),
            nodes::CtxSplitNum{ 1 }.parameterize(compressor),
            nodes::CtxSplitNum{ 2 }.parameterize(compressor),
            nodes::CtxSplitNum{ 3 }.parameterize(compressor),
        };
    }

    std::vector<NodeID> generateNodes(
            Compressor& compressor,
            datagen::DataGen& gen,
            size_t num) const override
    {
        std::vector<NodeID> out;
        out.reserve(num);
        for (size_t i = 0; i < num; ++i) {
            int nCtxLog = static_cast<int>(gen.usize_range("nCtxLog", 1, 3));
            out.push_back(
                    nodes::CtxSplitNum{ nCtxLog }.parameterize(compressor));
        }
        return out;
    }

    std::vector<std::unique_ptr<OpenZLInput>> predefinedInputs() const override
    {
        std::vector<std::unique_ptr<OpenZLInput>> inputs;
        // u8
        inputs.push_back(U8OpenZLInput::make(std::vector<uint8_t>{}));
        inputs.push_back(U8OpenZLInput::make(std::vector<uint8_t>{ 7 }));
        inputs.push_back(U8OpenZLInput::make(std::vector<uint8_t>(300, 42)));
        inputs.push_back(U8OpenZLInput::make(
                makeMarkovNumericLinear<uint8_t>(1024, 16, 28)));
        // u16
        inputs.push_back(U16OpenZLInput::make(std::vector<uint16_t>{}));
        inputs.push_back(U16OpenZLInput::make(
                makeMarkovNumericLinear<uint16_t>(1024, 500, 4000)));
        inputs.push_back(
                U16OpenZLInput::make(makeMarkovNumericLog2<uint16_t>(1024)));
        // u32
        inputs.push_back(U32OpenZLInput::make(std::vector<uint32_t>{}));
        inputs.push_back(U32OpenZLInput::make(
                makeMarkovNumericLinear<uint32_t>(1024, 10000, 1000000)));
        inputs.push_back(
                U32OpenZLInput::make(makeMarkovNumericLog2<uint32_t>(1024)));
        // u64
        inputs.push_back(U64OpenZLInput::make(std::vector<uint64_t>{}));
        inputs.push_back(U64OpenZLInput::make(
                makeMarkovNumericLinear<uint64_t>(
                        1024, 1000000ULL, 100000000000ULL)));
        inputs.push_back(
                U64OpenZLInput::make(makeMarkovNumericLog2<uint64_t>(1024)));
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

std::unique_ptr<OpenZLComponent> makeCtxSplitComponent()
{
    return std::make_unique<CtxSplitComponent>();
}

std::unique_ptr<OpenZLComponent> makeCtxSplitNumComponent()
{
    return std::make_unique<CtxSplitNumComponent>();
}

} // namespace openzl::tests::components
