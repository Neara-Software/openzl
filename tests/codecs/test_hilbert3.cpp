// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include "openzl/codecs/zl_hilbert3.h"
#include "openzl/openzl.hpp"
#include "tests/codecs/test_codec.h"

namespace openzl {
namespace tests {
namespace {

uint64_t split3Ref(uint64_t a)
{
    a = (a | (a << 16)) & 0x0000FF0000FFULL;
    a = (a | (a << 8)) & 0x00F00F00F00FULL;
    a = (a | (a << 4)) & 0x0C30C30C30C3ULL;
    a = (a | (a << 2)) & 0x249249249249ULL;
    return a;
}

uint64_t mortonToHilbert3Ref(uint64_t h, uint32_t bits)
{
    uint32_t block = bits * 3 - 3;
    uint32_t hcode = static_cast<uint32_t>((h >> block) & 7);
    uint32_t shift = 0;
    uint32_t signs = 0;
    while (block > 0) {
        block -= 3;
        hcode <<= 2;
        uint32_t mcode = (0x20212021u >> hcode) & 3;
        shift          = (0x48u >> (7 - shift - mcode)) & 3;
        signs          = (signs | (signs << 3)) >> mcode;
        signs          = (signs ^ (0x53560300u >> hcode)) & 7;
        mcode          = static_cast<uint32_t>((h >> block) & 7);
        hcode          = mcode;
        hcode          = ((hcode | (hcode << 3)) >> shift) & 7;
        hcode ^= signs;
        h ^= (static_cast<uint64_t>(mcode ^ hcode)) << block;
    }
    uint64_t const mask = 0x492492492492ULL;
    h ^= (h >> 1) & mask;
    h ^= (h & mask) >> 1;
    return h;
}

void hilbert3Key32Ref(
        uint32_t x,
        uint32_t y,
        uint32_t z,
        uint32_t* hi,
        uint64_t* lo)
{
    uint32_t d[32];
    for (int i = 0; i < 32; i++) {
        d[i] = ((x >> i) & 1) | (((y >> i) & 1) << 1) | (((z >> i) & 1) << 2);
    }
    uint32_t hcode = d[31];
    uint32_t shift = 0;
    uint32_t signs = 0;
    for (int b = 30; b >= 0; b--) {
        hcode <<= 2;
        uint32_t const mcode = (0x20212021u >> hcode) & 3;
        shift                = (0x48u >> (7 - shift - mcode)) & 3;
        signs                = (signs | (signs << 3)) >> mcode;
        signs                = (signs ^ (0x53560300u >> hcode)) & 7;
        hcode                = (((d[b] | (d[b] << 3)) >> shift) & 7) ^ signs;
        d[b]                 = hcode;
    }
    uint64_t l = 0;
    uint32_t h = 0;
    for (int i = 0; i < 32; i++) {
        uint32_t v = d[i];
        v ^= (v >> 1) & 2;
        v ^= (v & 2) >> 1;
        if (i < 21) {
            l |= static_cast<uint64_t>(v) << (3 * i);
        } else if (i == 21) {
            l |= static_cast<uint64_t>(v & 1) << 63;
            h |= v >> 1;
        } else {
            h |= v << (3 * i - 64);
        }
    }
    *hi = h;
    *lo = l;
}

class CaptureU64StreamGraph : public FunctionGraph {
   public:
    explicit CaptureU64StreamGraph(std::vector<uint64_t>* out) : out_(out) {}

    FunctionGraphDescription functionGraphDescription() const override
    {
        return FunctionGraphDescription{
            .name           = "CaptureU64Stream",
            .inputTypeMasks = { TypeMask::Numeric },
        };
    }

    void graph(GraphState& state) const override
    {
        const auto& in = state.edges()[0].getInput();
        const size_t n = in.numElts();
        out_->resize(n);
        if (n > 0) {
            std::memcpy(out_->data(), in.ptr(), n * sizeof(uint64_t));
        }
        state.edges()[0].setDestination(graphs::Store{}());
    }

   private:
    std::vector<uint64_t>* out_;
};

class Hilbert3Test : public CodecTest {
   protected:
    void checkU16ParityAndRoundTrip(const std::vector<uint16_t>& xyz)
    {
        ASSERT_EQ(xyz.size() % 3, 0u);
        const size_t n = xyz.size() / 3;

        std::vector<uint64_t> expectedGaps(n);
        uint64_t prev = 0;
        for (size_t i = 0; i < n; i++) {
            uint64_t const m = split3Ref(xyz[3 * i])
                    | (split3Ref(xyz[3 * i + 1]) << 1)
                    | (split3Ref(xyz[3 * i + 2]) << 2);
            uint64_t const k = mortonToHilbert3Ref(m, 16);
            expectedGaps[i]  = k - prev;
            prev             = k;
        }

        std::vector<uint64_t> actualGaps;
        compressor_.setParameter(
                CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
        auto capture = compressor_.registerFunctionGraph(
                std::make_shared<CaptureU64StreamGraph>(&actualGaps));
        auto graph = compressor_.buildStaticGraph(
                NodeID(ZL_NODE_HILBERT3_U16), { capture });
        compressor_.selectStartingGraph(graph);

        testRoundTrip(Input::refStruct(xyz.data(), 6, n));
        EXPECT_EQ(actualGaps, expectedGaps) << "n=" << n;
    }

    void checkI32ParityAndRoundTrip(const std::vector<int32_t>& xyz)
    {
        ASSERT_EQ(xyz.size() % 3, 0u);
        const size_t n = xyz.size() / 3;

        std::vector<uint64_t> expectedGaps;
        uint32_t phi = 0;
        uint64_t plo = 0;
        for (size_t i = 0; i < n; i++) {
            uint32_t hi = 0;
            uint64_t lo = 0;
            hilbert3Key32Ref(
                    static_cast<uint32_t>(xyz[3 * i]) ^ 0x80000000u,
                    static_cast<uint32_t>(xyz[3 * i + 1]) ^ 0x80000000u,
                    static_cast<uint32_t>(xyz[3 * i + 2]) ^ 0x80000000u,
                    &hi,
                    &lo);
            uint64_t const dlo = lo - plo;
            uint32_t const dhi = hi - phi - (lo < plo);
            bool const down    = hi < phi || (hi == phi && lo < plo);
            if (i == 0 || down || dhi != 0 || dlo == UINT64_MAX) {
                expectedGaps.push_back(UINT64_MAX);
                expectedGaps.push_back(hi);
                expectedGaps.push_back(lo);
            } else {
                expectedGaps.push_back(dlo);
            }
            phi = hi;
            plo = lo;
        }

        std::vector<uint64_t> actualGaps;
        compressor_.setParameter(
                CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
        auto capture = compressor_.registerFunctionGraph(
                std::make_shared<CaptureU64StreamGraph>(&actualGaps));
        auto graph = compressor_.buildStaticGraph(
                NodeID(ZL_NODE_HILBERT3_I32), { capture });
        compressor_.selectStartingGraph(graph);

        testRoundTrip(Input::refStruct(xyz.data(), 12, n));
        EXPECT_EQ(actualGaps, expectedGaps) << "n=" << n;
    }
};

TEST_F(Hilbert3Test, Hilbert3U16ParityAndRoundTrip)
{
    std::mt19937 rng(42);
    const std::vector<size_t> lengths = {
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 1000
    };
    for (size_t n : lengths) {
        std::vector<uint16_t> randomPts(3 * n);
        for (auto& v : randomPts) {
            v = static_cast<uint16_t>(rng() & 0xffffu);
        }
        checkU16ParityAndRoundTrip(randomPts);

        // Clustered / spatially coherent points (exercises top-prefix cache)
        std::vector<uint16_t> coherentPts(3 * n);
        for (size_t i = 0; i < n; i++) {
            coherentPts[3 * i + 0] = static_cast<uint16_t>(0x8000u + (i & 63u));
            coherentPts[3 * i + 1] =
                    static_cast<uint16_t>(0x4000u + ((i * 3) & 127u));
            coherentPts[3 * i + 2] =
                    static_cast<uint16_t>(0xC000u + ((i * 5) & 63u));
        }
        checkU16ParityAndRoundTrip(coherentPts);
    }

    // Extremes
    checkU16ParityAndRoundTrip({
            0, 0, 0,
            65535, 65535, 65535,
            65535, 0, 32768,
            0, 65535, 32767,
            32768, 32768, 32768,
    });
}

TEST_F(Hilbert3Test, Hilbert3I32ParityAndRoundTrip)
{
    std::mt19937 rng(123);
    const std::vector<size_t> lengths = {
        0, 1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 1000
    };
    for (size_t n : lengths) {
        std::vector<int32_t> randomPts(3 * n);
        for (auto& v : randomPts) {
            v = static_cast<int32_t>(rng());
        }
        checkI32ParityAndRoundTrip(randomPts);

        // Coherent points sharing top 15 bits across runs
        std::vector<int32_t> coherentPts(3 * n);
        for (size_t i = 0; i < n; i++) {
            int32_t const base = static_cast<int32_t>((i / 17) * 100000 - 500000);
            coherentPts[3 * i + 0] = base + static_cast<int32_t>(i & 31);
            coherentPts[3 * i + 1] = base + static_cast<int32_t>((i * 7) & 63);
            coherentPts[3 * i + 2] = -base + static_cast<int32_t>((i * 3) & 31);
        }
        checkI32ParityAndRoundTrip(coherentPts);
    }

    // Extremes (INT32_MIN, INT32_MAX, 0)
    checkI32ParityAndRoundTrip({
            INT32_MIN, INT32_MIN, INT32_MIN,
            INT32_MAX, INT32_MAX, INT32_MAX,
            0, 0, 0,
            INT32_MIN, INT32_MAX, 0,
            -1, -1, -1,
    });
}

} // namespace
} // namespace tests
} // namespace openzl
