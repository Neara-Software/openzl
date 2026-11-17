// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include "openzl/cpp/codecs/LookbackDelta.hpp"
#include "openzl/cpp/codecs/PcoBinSplit.hpp"
#include "openzl/cpp/codecs/Entropy.hpp"
#include "openzl/openzl.hpp"
#include "tests/codecs/test_codec.h"

namespace openzl {
namespace tests {
namespace {

class CaptureResidualsGraph : public FunctionGraph {
   public:
    CaptureResidualsGraph(std::vector<uint8_t>* rawBytes, size_t* count)
            : rawBytes_(rawBytes), count_(count)
    {
    }

    FunctionGraphDescription functionGraphDescription() const override
    {
        return FunctionGraphDescription{
            .name           = "CaptureLbdStream",
            .inputTypeMasks = { TypeMask::Numeric },
        };
    }

    void graph(GraphState& state) const override
    {
        const auto& input = state.edges()[0].getInput();
        if (count_ != nullptr) {
            *count_ = input.numElts();
        }
        if (rawBytes_ != nullptr) {
            const uint8_t* ptr =
                    reinterpret_cast<const uint8_t*>(input.ptr());
            rawBytes_->assign(ptr, ptr + input.contentSize());
        }
        state.edges()[0].setDestination(graphs::Store{}());
    }

   private:
    std::vector<uint8_t>* rawBytes_;
    size_t* count_;
};

class LookbackDeltaTest : public CodecTest {
   protected:
    template <typename T>
    void testForcedLookbackRoundTrip(const std::vector<T>& data, int windowLog)
    {
        compressor_.setParameter(
                CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
        auto node = nodes::LookbackDelta{ windowLog, true }.parameterize(compressor_);
        auto graph = compressor_.buildStaticGraph(
                node, { graphs::Store{}(), graphs::Store{}() });
        compressor_.selectStartingGraph(graph);
        cctx_.refCompressor(compressor_);

        auto input = Input::refNumeric(poly::span<const T>{ data });
        size_t const cap = ZL_COMPRESSBOUND_UNGUARDED(
                data.size() * (sizeof(T) + sizeof(uint16_t)) + 4096u);
        std::string compressed(cap, '\0');
        compressed.resize(cctx_.compress(compressed, { &input, 1 }));
        auto roundTripped = dctx_.decompress(compressed);
        ASSERT_EQ(roundTripped.size(), 1u);
        EXPECT_EQ(roundTripped[0], input);
    }

    template <typename T>
    void testPeriodicAndMixed(size_t period, size_t count)
    {
        std::vector<T> data(count);
        for (size_t i = 0; i < count; ++i) {
            // Periodic pattern with slowly varying drift so lookback d == period dominates
            size_t const cycle = i / period;
            size_t const phase = i % period;
            data[i] = static_cast<T>(
                    phase * 131u + cycle * 3u + ((i / 97u) & 7u));
        }
        compressor_.setParameter(
                CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
        auto node  = nodes::LookbackDelta{ 16, true }.parameterize(compressor_);
        auto graph = compressor_.buildStaticGraph(
                node, { graphs::Store{}(), graphs::Store{}() });
        compressor_.selectStartingGraph(graph);
        testRoundTrip(Input::refNumeric(poly::span<const T>{ data }));
    }

    template <typename T>
    void testPolynomialRoundTrip(int degree, size_t count, bool largeWrap)
    {
        std::vector<T> data(count);
        for (size_t i = 0; i < count; ++i) {
            uint64_t const x = static_cast<uint64_t>(i);
            uint64_t val     = 0;
            if (largeWrap) {
                // Large multipliers force frequent unsigned wrapping mod 2^W
                uint64_t const c0 = 0xdeadbeefcafebabeULL;
                uint64_t const c1 = 0x123456789abcdef1ULL;
                uint64_t const c2 = 0x0fedcba987654321ULL;
                uint64_t const c3 = 0x0011223344556677ULL;
                val               = c0 + c1 * x;
                if (degree >= 2) {
                    val += c2 * x * x;
                }
                if (degree >= 3) {
                    val += c3 * x * x * x;
                }
            } else {
                val = 100ULL + 7ULL * x;
                if (degree >= 2) {
                    val += 5ULL * x * x;
                }
                if (degree >= 3) {
                    val += 3ULL * x * x * x;
                }
            }
            data[i] = static_cast<T>(val);
        }
        compressor_.setParameter(
                CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
        auto node = nodes::LookbackDelta{ 15, false }.parameterize(compressor_);
        auto graph = compressor_.buildStaticGraph(
                node, { graphs::Store{}(), graphs::Store{}() });
        compressor_.selectStartingGraph(graph);
        testRoundTrip(Input::refNumeric(poly::span<const T>{ data }));
    }
};

TEST_F(LookbackDeltaTest, AllWidthsAndLookbackDistances)
{
    const std::vector<size_t> periods = { 1, 2, 3, 4, 8, 15, 16, 17, 31, 32, 33, 64 };
    for (size_t p : periods) {
        testPeriodicAndMixed<uint8_t>(p, 512);
        testPeriodicAndMixed<uint16_t>(p, 512);
        testPeriodicAndMixed<uint32_t>(p, 512);
        testPeriodicAndMixed<uint64_t>(p, 512);
    }

    // Edge lengths including 0, 1, 16, 17, and > 65535 validation boundary
    for (size_t n : { size_t(0), size_t(1), size_t(2), size_t(16), size_t(17), size_t(65540) }) {
        testPeriodicAndMixed<uint32_t>(7, n);
        testPeriodicAndMixed<uint16_t>(16, n);
    }
}

TEST_F(LookbackDeltaTest, ConsecutiveAndRawModes)
{
    for (size_t n : { size_t(0), size_t(1), size_t(7), size_t(16), size_t(17), size_t(1025) }) {
        std::vector<uint64_t> data(n);
        for (size_t i = 0; i < n; ++i) {
            data[i] = 1000000ULL + i * 17ULL + (i % 5);
        }
        compressor_.setParameter(
                CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
        auto node = nodes::LookbackDelta{ 15, false }.parameterize(compressor_);
        auto graph = compressor_.buildStaticGraph(
                node, { graphs::Store{}(), graphs::Store{}() });
        compressor_.selectStartingGraph(graph);
        testRoundTrip(Input::refNumeric(poly::span<const uint64_t>{ data }));
    }
}

TEST_F(LookbackDeltaTest, HigherOrderConsecutiveAllWidthsAndWrapping)
{
    const std::vector<size_t> lengths = {
        0, 1, 2, 3, 4, 5, 15, 16, 17, 31, 32, 33, 64, 513, 3072, 3073, 4097
    };
    for (int degree : { 1, 2, 3 }) {
        for (size_t n : lengths) {
            testPolynomialRoundTrip<uint8_t>(degree, n, false);
            testPolynomialRoundTrip<uint8_t>(degree, n, true);
            testPolynomialRoundTrip<uint16_t>(degree, n, false);
            testPolynomialRoundTrip<uint16_t>(degree, n, true);
            testPolynomialRoundTrip<uint32_t>(degree, n, false);
            testPolynomialRoundTrip<uint32_t>(degree, n, true);
            testPolynomialRoundTrip<uint64_t>(degree, n, false);
            testPolynomialRoundTrip<uint64_t>(degree, n, true);
        }
    }
}

TEST_F(LookbackDeltaTest, HigherOrderModeSelectionAndResiduals)
{
    // Quadratic: x[i] = 1000 + 50*i + 3*i^2 -> order 2 residuals[2..] == 6
    {
        constexpr size_t kN = 1024;
        std::vector<uint64_t> quad(kN);
        for (size_t i = 0; i < kN; ++i) {
            uint64_t const x = static_cast<uint64_t>(i);
            quad[i]          = 1000ULL + 50ULL * x + 3ULL * x * x;
        }
        std::vector<uint8_t> resBytes;
        size_t lbCount = 999999;
        compressor_.setParameter(
                CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
        auto resCap = compressor_.registerFunctionGraph(
                std::make_shared<CaptureResidualsGraph>(&resBytes, nullptr));
        auto lbCap = compressor_.registerFunctionGraph(
                std::make_shared<CaptureResidualsGraph>(nullptr, &lbCount));
        auto node = nodes::LookbackDelta{ 15, false }.parameterize(compressor_);
        auto graph = compressor_.buildStaticGraph(node, { resCap, lbCap });
        compressor_.selectStartingGraph(graph);
        testRoundTrip(Input::refNumeric(poly::span<const uint64_t>{ quad }));
        EXPECT_EQ(lbCount, 0u);
        ASSERT_EQ(resBytes.size(), kN * sizeof(uint64_t));
        const uint64_t* res =
                reinterpret_cast<const uint64_t*>(resBytes.data());
        EXPECT_EQ(res[0], quad[0]);
        EXPECT_EQ(res[1], quad[1] - quad[0]);
        for (size_t i = 2; i < kN; ++i) {
            EXPECT_EQ(res[i], 6ULL);
        }
    }

    // Cubic with uint32_t wrapping: x[i] = 5 + 11*i + 7*i^2 + 1000000007*i^3
    {
        constexpr size_t kN = 1024;
        std::vector<uint32_t> cubic(kN);
        constexpr uint32_t kC3 = 1000000007U;
        for (size_t i = 0; i < kN; ++i) {
            uint32_t const x = static_cast<uint32_t>(i);
            cubic[i] = 5U + 11U * x + 7U * x * x + kC3 * x * x * x;
        }
        std::vector<uint8_t> resBytes;
        size_t lbCount = 999999;
        compressor_.setParameter(
                CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
        auto resCap = compressor_.registerFunctionGraph(
                std::make_shared<CaptureResidualsGraph>(&resBytes, nullptr));
        auto lbCap = compressor_.registerFunctionGraph(
                std::make_shared<CaptureResidualsGraph>(nullptr, &lbCount));
        auto node = nodes::LookbackDelta{ 15, false }.parameterize(compressor_);
        auto graph = compressor_.buildStaticGraph(node, { resCap, lbCap });
        compressor_.selectStartingGraph(graph);
        testRoundTrip(Input::refNumeric(poly::span<const uint32_t>{ cubic }));
        EXPECT_EQ(lbCount, 0u);
        ASSERT_EQ(resBytes.size(), kN * sizeof(uint32_t));
        const uint32_t* res =
                reinterpret_cast<const uint32_t*>(resBytes.data());
        EXPECT_EQ(res[0], cubic[0]);
        EXPECT_EQ(res[1], static_cast<uint32_t>(cubic[1] - cubic[0]));
        EXPECT_EQ(
                res[2],
                static_cast<uint32_t>(
                        (cubic[2] - cubic[1]) - (cubic[1] - cubic[0])));
        uint32_t const expectedC3 = static_cast<uint32_t>(6U * kC3);
        for (size_t i = 3; i < kN; ++i) {
            EXPECT_EQ(res[i], expectedC3);
        }
    }
}

TEST_F(LookbackDeltaTest, ClusteredConsecutiveAndUncorrelatedRawAvoidLookback)
{
    // Clustered consecutive-1 deltas (like GPS timestamps with regular step jumps + small jitter)
    {
        constexpr size_t kN = 4096;
        std::vector<uint64_t> ts(kN);
        uint64_t cur = 342466189383205ULL;
        for (size_t i = 0; i < kN; ++i) {
            int64_t step = ((i % 5) == 0) ? 4260 : (((i % 3) == 0) ? -4260 : 1);
            cur += static_cast<uint64_t>(step + static_cast<int64_t>(i & 3));
            ts[i] = cur;
        }
        size_t lbCount = 999999;
        compressor_.setParameter(
                CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
        auto lbCap = compressor_.registerFunctionGraph(
                std::make_shared<CaptureResidualsGraph>(nullptr, &lbCount));
        auto node = nodes::LookbackDelta{ 15, false }.parameterize(compressor_);
        auto graph = compressor_.buildStaticGraph(
                node, { graphs::Store{}(), lbCap });
        compressor_.selectStartingGraph(graph);
        testRoundTrip(Input::refNumeric(poly::span<const uint64_t>{ ts }));
        EXPECT_EQ(lbCount, 0u);
    }

    // Constant-0 stream should pick RAW and avoid lookback
    {
        std::vector<int16_t> zeros(8192, 0);
        std::vector<uint8_t> resBytes;
        size_t lbCount = 999999;
        compressor_.setParameter(
                CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
        auto resCap = compressor_.registerFunctionGraph(
                std::make_shared<CaptureResidualsGraph>(&resBytes, nullptr));
        auto lbCap = compressor_.registerFunctionGraph(
                std::make_shared<CaptureResidualsGraph>(nullptr, &lbCount));
        auto node = nodes::LookbackDelta{ 15, false }.parameterize(compressor_);
        auto graph = compressor_.buildStaticGraph(node, { resCap, lbCap });
        compressor_.selectStartingGraph(graph);
        testRoundTrip(Input::refNumeric(poly::span<const int16_t>{ zeros }));
        EXPECT_EQ(lbCount, 0u);
    }
}


TEST_F(LookbackDeltaTest, SimdBoundaryLengthsAndCapAcrossWidths)
{
    auto runMixedLookback = [this](auto tag, size_t n, int windowLog) {
        using T = decltype(tag);
        std::vector<T> data(n);
        for (size_t i = 0; i < n; ++i) {
            // Repeats values from varying distances (1..64 and 65536) plus drift
            if (i >= 65536 && (i % 4 == 0)) {
                data[i] = data[i - 65536];
            } else if (i >= 37 && (i % 7 == 0)) {
                data[i] = static_cast<T>(data[i - 37] + 1u);
            } else if (i >= 16 && ((i / 32) % 3 == 0)) {
                data[i] = static_cast<T>(data[i - 16] + 3u);
            } else if (i >= 1 && ((i / 32) % 3 == 1)) {
                data[i] = static_cast<T>(data[i - 1] + 5u);
            } else {
                size_t const back = (i > 0) ? ((i * 5u) % std::min<size_t>(i, 15u)) + 1u : 0u;
                data[i] = back ? static_cast<T>(data[i - back] + static_cast<T>(i * 13u + 7u))
                               : static_cast<T>(0x9e3779b97f4a7c15ULL);
            }
        }
        testForcedLookbackRoundTrip(data, windowLog);
    };

    const std::vector<size_t> boundaryLengths = {
        0,  1,  2,  3,  4,  7,  8,  9,  15, 16, 17, 31,
        32, 33, 47, 48, 49, 63, 64, 65, 127, 128, 129, 135,
    };
    for (size_t n : boundaryLengths) {
        runMixedLookback(uint8_t{}, n, 8);
        runMixedLookback(uint16_t{}, n, 8);
        runMixedLookback(uint32_t{}, n, 8);
        runMixedLookback(uint64_t{}, n, 8);
    }

    for (size_t n : { size_t(65534), size_t(65535), size_t(65536), size_t(65537), size_t(65545) }) {
        runMixedLookback(uint8_t{}, n, 16);
        runMixedLookback(uint16_t{}, n, 16);
        runMixedLookback(uint32_t{}, n, 16);
        runMixedLookback(uint64_t{}, n, 16);
    }
}

TEST_F(LookbackDeltaTest, Mode2RegimeTransitionsAllWidths)
{
    auto testTransitions = [this](auto tag) {
        using T = decltype(tag);
        constexpr size_t kN = 2048;
        std::vector<T> data(kN);
        uint32_t rng = 0x13579bdfU;
        auto nextRng = [&rng]() {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            return rng;
        };
        for (size_t i = 0; i < kN; ++i) {
            size_t const phase = (i / 43u) % 5u;
            if (i < 64) {
                data[i] = static_cast<T>(nextRng());
            } else if (phase == 0) {
                // Uniform d = 1
                data[i] = data[i - 1];
            } else if (phase == 1) {
                // Uniform d = 8 (128-bit half-vector stride on u8)
                data[i] = data[i - 8];
            } else if (phase == 2) {
                // Uniform d = 16
                data[i] = data[i - 16];
            } else if (phase == 3) {
                // Long non-uniform lookbacks (d in 20..63), exercising scalar fallback
                size_t const d = 20u + (nextRng() % 44u);
                data[i]        = data[i - d];
            } else {
                // Short causal DAG lookbacks (d in 1..8), exercising prev reload after scalar
                size_t const d = 1u + (nextRng() % 8u);
                data[i]        = static_cast<T>(data[i - d] + static_cast<T>(nextRng() & 3u));
            }
        }
        testForcedLookbackRoundTrip(data, 8);
    };

    testTransitions(uint8_t{});
    testTransitions(uint16_t{});
    testTransitions(uint32_t{});
    testTransitions(uint64_t{});
}

TEST_F(LookbackDeltaTest, RejectsOutOfBoundsLookbacksAtSimdBoundaries)
{
    constexpr size_t kN = 65545;
    std::vector<uint16_t> data(kN);
    for (size_t i = 0; i < kN; ++i) {
        data[i] = static_cast<uint16_t>((i * 257u) ^ (i >> 3));
    }

    std::vector<uint8_t> lbBytes;
    compressor_.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    compressor_.setParameter(
            CParam::CompressedChecksum, ZL_TernaryParam_disable);
    compressor_.setParameter(CParam::ContentChecksum, ZL_TernaryParam_disable);
    auto lbCap = compressor_.registerFunctionGraph(
            std::make_shared<CaptureResidualsGraph>(&lbBytes, nullptr));
    auto node = nodes::LookbackDelta{ 8, true }.parameterize(compressor_);
    auto graph = compressor_.buildStaticGraph(
            node, { graphs::Store{}(), lbCap });
    compressor_.selectStartingGraph(graph);

    std::string const frame =
            testRoundTrip(Input::refNumeric(poly::span<const uint16_t>{ data }));
    ASSERT_EQ(lbBytes.size(), (kN - 1u) * sizeof(uint16_t));

    std::string const needle(
            reinterpret_cast<const char*>(lbBytes.data()), lbBytes.size());
    size_t const lbOffset = frame.find(needle);
    ASSERT_NE(lbOffset, std::string::npos);

    // Mutate lb[k] = k + 1 (distance k + 2 > i = k + 1) across SIMD vector boundaries
    // and at the k = 65534 max-validation boundary.
    for (size_t k : { size_t(0),
                      size_t(1),
                      size_t(7),
                      size_t(8),
                      size_t(15),
                      size_t(16),
                      size_t(31),
                      size_t(32),
                      size_t(63),
                      size_t(64),
                      size_t(65533),
                      size_t(65534) }) {
        std::string corrupted   = frame;
        uint16_t const invalidLb = static_cast<uint16_t>(k + 1u);
        std::memcpy(&corrupted[lbOffset + k * sizeof(uint16_t)], &invalidLb, sizeof(uint16_t));
        EXPECT_THROW(dctx_.decompress(corrupted), Exception) << "k=" << k;
    }
}

TEST_F(LookbackDeltaTest, MultiRegionSamplingFindsLookbackInMixedStream)
{
    // Build a 32768-element stream whose start, exact midpoint, and end are
    // single-track ramps, while the interior octiles are 4 interleaved tracks
    // separated by 10^12 (requiring LOOKBACK with distance 4).
    constexpr size_t kN = 32768;
    std::vector<uint64_t> data(kN);
    constexpr uint64_t kBases[4] = {
        1000000000000ULL,
        2500000000000ULL,
        4000000000000ULL,
        5500000000000ULL,
    };
    for (size_t i = 0; i < kN; ++i) {
        bool const inRamp =
                (i < 1024u) || (i >= 15800u && i < 16900u) || (i >= kN - 1024u);
        if (inRamp) {
            data[i] = 1000000ULL + i * 100ULL;
        } else {
            data[i] = kBases[i & 3u] + (i >> 2u) * 10ULL;
        }
    }

    size_t lbCount = 0;
    compressor_.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    auto lbProbe = compressor_.registerFunctionGraph(
            std::make_shared<CaptureResidualsGraph>(nullptr, &lbCount));
    auto resGraph = nodes::PcoBinSplit{ nodes::PcoBinSplit::LatentMode::Signed }(
            compressor_, graphs::Entropy{}(), graphs::Store{}());
    auto node  = nodes::LookbackDelta{ 15, true }.parameterize(compressor_);
    auto graph = compressor_.buildStaticGraph(node, { resGraph, lbProbe });
    compressor_.selectStartingGraph(graph);

    auto comp = testRoundTrip(
            Input::refNumeric(poly::span<const uint64_t>{ data }));
    EXPECT_EQ(lbCount, kN - 1u);
    EXPECT_LT(comp.size(), kN * sizeof(uint64_t) / 2u);
}

} // namespace
} // namespace tests
} // namespace openzl
