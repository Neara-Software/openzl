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

TEST_F(PcoDecompTest, PcoBinSplitWidthsModesAndBatchBoundaries)
{
    const std::vector<size_t> lengths = {
        0, 1, 7, 8, 9, 255, 256, 257, 511, 512, 513, 1000, 5000
    };
    auto testType = [&](auto tag, nodes::PcoBinSplit::LatentMode mode) {
        using T = decltype(tag);
        for (size_t len : lengths) {
            compressor_.setParameter(
                    CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
            auto graph = nodes::PcoBinSplit{ mode }(
                    compressor_, graphs::Entropy{}(), graphs::Store{}());
            compressor_.selectStartingGraph(graph);

            // Multi-bin with ob==0 and ob>0 bins
            std::vector<T> multi(len);
            for (size_t i = 0; i < len; ++i) {
                if (i % 7 == 0) {
                    multi[i] = static_cast<T>(42);
                } else if (i % 3 == 0) {
                    multi[i] = static_cast<T>(100 + (i & 31));
                } else {
                    multi[i] = static_cast<T>((i * 13) & 0x7f);
                }
            }
            testRoundTrip(Input::refNumeric(poly::span<const T>{ multi }));

            // Constant (singleBin, ob==0)
            std::vector<T> constant(len, static_cast<T>(77));
            testRoundTrip(Input::refNumeric(poly::span<const T>{ constant }));

            // All-zero offsets across multiple exact bins
            std::vector<T> allZero(len);
            for (size_t i = 0; i < len; ++i) {
                allZero[i] = static_cast<T>((i % 4) * 33 + 11);
            }
            testRoundTrip(Input::refNumeric(poly::span<const T>{ allZero }));
        }
    };

    testType(uint8_t{}, nodes::PcoBinSplit::LatentMode::Unsigned);
    testType(int8_t{}, nodes::PcoBinSplit::LatentMode::Signed);
    testType(uint16_t{}, nodes::PcoBinSplit::LatentMode::Unsigned);
    testType(int16_t{}, nodes::PcoBinSplit::LatentMode::Signed);
    testType(uint32_t{}, nodes::PcoBinSplit::LatentMode::Unsigned);
    testType(int32_t{}, nodes::PcoBinSplit::LatentMode::Signed);
    testType(uint64_t{}, nodes::PcoBinSplit::LatentMode::Unsigned);
    testType(int64_t{}, nodes::PcoBinSplit::LatentMode::Signed);

    // Float widths (32-bit and 64-bit)
    for (size_t len : lengths) {
        compressor_.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
        auto graph = nodes::PcoBinSplit{ nodes::PcoBinSplit::LatentMode::Float }(
                compressor_, graphs::Entropy{}(), graphs::Store{}());
        compressor_.selectStartingGraph(graph);
        std::vector<float> f32(len);
        std::vector<double> f64(len);
        for (size_t i = 0; i < len; ++i) {
            f32[i] = (i % 5 == 0) ? 1.5f : static_cast<float>(i % 19) * 0.25f - 2.0f;
            f64[i] = (i % 5 == 0) ? -3.5 : static_cast<double>(i % 23) * 0.125 - 1.0;
        }
        testRoundTrip(Input::refNumeric(poly::span<const float>{ f32 }));
        testRoundTrip(Input::refNumeric(poly::span<const double>{ f64 }));
    }
}

TEST_F(PcoDecompTest, PcoBinSplitWideOffsetsAbove57Bits)
{
    for (size_t len : { size_t(1), size_t(17), size_t(255), size_t(256), size_t(520) }) {
        compressor_.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
        auto graph = nodes::PcoBinSplit{ nodes::PcoBinSplit::LatentMode::Unsigned }(
                compressor_, graphs::Entropy{}(), graphs::Store{}());
        compressor_.selectStartingGraph(graph);

        // Single-bin wide offsets (ob == 64)
        std::vector<uint64_t> wideSingle(len);
        for (size_t i = 0; i < len; ++i) {
            wideSingle[i] = (i * 0x9E3779B97F4A7C15ULL) ^ 0xA5A5A5A5A5A5A5A5ULL;
        }
        testRoundTrip(Input::refNumeric(poly::span<const uint64_t>{ wideSingle }));

        // Mixed ob==0, small ob, and ob >= 59 bins in the same stream
        std::vector<uint64_t> wideMixed(len);
        for (size_t i = 0; i < len; ++i) {
            if (i % 4 == 0) {
                wideMixed[i] = 12345ULL; // ob == 0 bin
            } else if (i % 4 == 1) {
                wideMixed[i] = 100000ULL + (i & 31); // small ob bin
            } else {
                // High-entropy 62-bit span in high range
                wideMixed[i] = 0x8000000000000000ULL
                        | ((i * 0x123456789ABCDEFULL) & 0x3FFFFFFFFFFFFFFFULL);
            }
        }
        testRoundTrip(Input::refNumeric(poly::span<const uint64_t>{ wideMixed }));
    }
}

} // namespace
} // namespace tests
} // namespace openzl
