// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

#include "openzl/cpp/codecs/FloatMultSplit.hpp"
#include "tests/registry/OpenZLComponents.h"
#include "tests/registry/OpenZLInput.h"

namespace openzl::tests::components {
namespace {

template <typename FloatT, typename UIntT>
std::vector<UIntT> makeFloatMultiples(
        size_t count,
        double base,
        bool withOutliers)
{
    std::vector<UIntT> out(count);
    for (size_t i = 0; i < count; ++i) {
        double v = static_cast<double>(static_cast<int64_t>(i % 257) - 128)
                * base;
        FloatT fv = static_cast<FloatT>(v);
        UIntT bits;
        std::memcpy(&bits, &fv, sizeof(bits));
        if (withOutliers && i % 23 == 0) {
            bits ^= UIntT(1);
        }
        out[i] = bits;
    }
    return out;
}

class FloatMultSplitComponent : public OpenZLComponent {
   public:
    std::string name() const override
    {
        return "FloatMultSplit";
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
        return { nodes::FloatMultSplit{}.parameterize(compressor) };
    }

    std::vector<std::unique_ptr<OpenZLInput>> predefinedInputs() const override
    {
        std::vector<std::unique_ptr<OpenZLInput>> inputs;
        // f32
        inputs.push_back(U32OpenZLInput::make(std::vector<uint32_t>{}));
        inputs.push_back(U32OpenZLInput::make(std::vector<uint32_t>{ 0x3f800000u }));
        inputs.push_back(
                U32OpenZLInput::make(
                        std::vector<uint32_t>{
                                0x00000000u, // +0.0f
                                0x80000000u, // -0.0f
                                0x7f800000u, // +inf
                                0xff800000u, // -inf
                                0x7fc01234u, // qNaN payload
                                0xff800001u, // -sNaN payload
                                0x00000001u, // subnormal min
                                0x7f7fffffu, // FLT_MAX
                        }));
        inputs.push_back(U32OpenZLInput::make(
                makeFloatMultiples<float, uint32_t>(512, 0.1, true)));
        inputs.push_back(U32OpenZLInput::make(
                makeFloatMultiples<float, uint32_t>(512, 0.25, false)));

        // f64
        inputs.push_back(U64OpenZLInput::make(std::vector<uint64_t>{}));
        inputs.push_back(
                U64OpenZLInput::make(
                        std::vector<uint64_t>{ 0x3ff0000000000000ULL }));
        inputs.push_back(
                U64OpenZLInput::make(
                        std::vector<uint64_t>{
                                0x0000000000000000ULL, // +0.0
                                0x8000000000000000ULL, // -0.0
                                0x7ff0000000000000ULL, // +inf
                                0xfff0000000000000ULL, // -inf
                                0x7ff8000000001234ULL, // qNaN payload
                                0xfff0000000000001ULL, // -sNaN payload
                                0x0000000000000001ULL, // subnormal min
                                0x7fefffffffffffffULL, // DBL_MAX
                        }));
        inputs.push_back(U64OpenZLInput::make(
                makeFloatMultiples<double, uint64_t>(512, 0.001, true)));
        inputs.push_back(U64OpenZLInput::make(
                makeFloatMultiples<double, uint64_t>(512, 1.0, false)));
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
            if (gen.boolean("is_f64")) {
                inputs.push_back(
                        U64OpenZLInput::make(gen.randVector<uint64_t>(
                                "values", 0, UINT64_MAX, maxInputSize / 8)));
            } else {
                inputs.push_back(
                        U32OpenZLInput::make(gen.randVector<uint32_t>(
                                "values", 0, UINT32_MAX, maxInputSize / 4)));
            }
        }
        return inputs;
    }
};

} // namespace

std::unique_ptr<OpenZLComponent> makeFloatMultSplitComponent()
{
    return std::make_unique<FloatMultSplitComponent>();
}

} // namespace openzl::tests::components
