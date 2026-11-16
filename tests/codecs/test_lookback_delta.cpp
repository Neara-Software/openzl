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

} // namespace
} // namespace tests
} // namespace openzl
