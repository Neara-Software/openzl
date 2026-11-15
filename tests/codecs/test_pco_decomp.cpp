// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "openzl/cpp/codecs/LookbackDelta.hpp"
#include "openzl/cpp/codecs/PcoBinSplit.hpp"
#include "openzl/openzl.hpp"
#include "tests/codecs/test_codec.h"

namespace openzl {
namespace tests {
namespace {

class CaptureLookbackCountGraph : public FunctionGraph {
   public:
    explicit CaptureLookbackCountGraph(size_t* outCount) : outCount_(outCount) {}

    FunctionGraphDescription functionGraphDescription() const override
    {
        return FunctionGraphDescription{
            .name           = "CaptureLookbackCount",
            .inputTypeMasks = { TypeMask::Numeric },
        };
    }

    void graph(GraphState& state) const override
    {
        *outCount_ = state.edges()[0].getInput().numElts();
        state.edges()[0].setDestination(graphs::Store{}());
    }

   private:
    size_t* outCount_;
};

class PcoDecompTest : public CodecTest {
   protected:
    template <typename T>
    void checkLookbackBehavior(
            const std::vector<T>& data,
            bool forceLookback,
            bool expectLookbacks)
    {
        compressor_.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
        size_t lbCount = 999999;
        auto capGraph  = compressor_.registerFunctionGraph(
                std::make_shared<CaptureLookbackCountGraph>(&lbCount));
        auto node  = nodes::LookbackDelta{ 15, forceLookback }.parameterize(
                compressor_);
        auto graph = compressor_.buildStaticGraph(
                node, { graphs::Store{}(), capGraph });
        compressor_.selectStartingGraph(graph);

        Input in = Input::refNumeric(poly::span<const T>{ data });
        testRoundTrip(in);
        if (expectLookbacks) {
            EXPECT_GT(lbCount, 0u);
        } else {
            EXPECT_EQ(lbCount, 0u);
        }
    }

    template <typename T>
    void checkPcoBinSplitCompresses(
            nodes::PcoBinSplit::LatentMode mode,
            const std::vector<T>& data,
            double maxRatioOfRaw)
    {
        compressor_.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
        auto graph = nodes::PcoBinSplit{ mode }(
                compressor_, graphs::Entropy{}(), graphs::Store{}());
        compressor_.selectStartingGraph(graph);

        Input in = Input::refNumeric(poly::span<const T>{ data });
        auto comp = testRoundTrip(in);
        EXPECT_LT(
                static_cast<double>(comp.size()),
                maxRatioOfRaw * static_cast<double>(data.size() * sizeof(T)));
    }
};

TEST_F(PcoDecompTest, LookbackDeltaModeSelection)
{
    // Linear ramp should fall back to consecutive-1 delta (0 lookbacks)
    std::vector<uint32_t> ramp(1024);
    for (size_t i = 0; i < ramp.size(); ++i) {
        ramp[i] = static_cast<uint32_t>(i * 7 + 100);
    }
    checkLookbackBehavior(ramp, /*forceLookback=*/false, /*expectLookbacks=*/false);

    // Interleaved periodic streams should engage lookback mode
    std::vector<uint64_t> periodic(1024);
    for (size_t i = 0; i < periodic.size(); ++i) {
        periodic[i] = (i % 8) * 1000000ULL + (i / 8);
    }
    checkLookbackBehavior(periodic, /*forceLookback=*/false, /*expectLookbacks=*/true);

    // forceLookback=true forces lookback stream even when consecutive is better
    checkLookbackBehavior(ramp, /*forceLookback=*/true, /*expectLookbacks=*/true);
}

TEST_F(PcoDecompTest, PcoBinSplitCompressesBimodalAndPreservesFloatBits)
{
    std::vector<uint64_t> bimodal(2048);
    for (size_t i = 0; i < bimodal.size(); ++i) {
        bimodal[i] = (i % 5 == 0) ? (1000000000ULL + (i & 31)) : (100ULL + (i & 15));
    }
    checkPcoBinSplitCompresses(
            nodes::PcoBinSplit::LatentMode::Unsigned, bimodal, 0.25);

    std::vector<int32_t> signedBimodal(2048);
    for (size_t i = 0; i < signedBimodal.size(); ++i) {
        signedBimodal[i] = (i & 1) ? static_cast<int32_t>(-500000 + (i & 7))
                                   : static_cast<int32_t>(500000 + (i & 7));
    }
    checkPcoBinSplitCompresses(
            nodes::PcoBinSplit::LatentMode::Signed, signedBimodal, 0.30);

    // Distinct NaN payloads + -0.0 must survive bit-exactly
    std::vector<double> edgeFloats(512);
    for (size_t i = 0; i < edgeFloats.size(); ++i) {
        if (i == 0) {
            edgeFloats[i] = -0.0;
        } else if (i == 1) {
            uint64_t nanBits = 0x7FF8000000001234ULL;
            std::memcpy(&edgeFloats[i], &nanBits, sizeof(double));
        } else {
            edgeFloats[i] = (i % 3 == 0) ? (-1000.0 - static_cast<double>(i & 7))
                                         : (1000.0 + static_cast<double>(i & 7));
        }
    }
    checkPcoBinSplitCompresses(
            nodes::PcoBinSplit::LatentMode::Float, edgeFloats, 0.35);
}

} // namespace
} // namespace tests
} // namespace openzl
