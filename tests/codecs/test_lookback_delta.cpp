// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "openzl/cpp/codecs/LookbackDelta.hpp"
#include "openzl/openzl.hpp"
#include "tests/codecs/test_codec.h"

namespace openzl {
namespace tests {
namespace {

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

} // namespace
} // namespace tests
} // namespace openzl
