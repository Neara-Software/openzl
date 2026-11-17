// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <gtest/gtest.h>

#include "openzl/cpp/codecs/LookbackDelta.hpp"
#include "openzl/cpp/codecs/Delta.hpp"
#include "openzl/cpp/codecs/FloatMultSplit.hpp"
#include "openzl/cpp/codecs/IntMultSplit.hpp"
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

TEST_F(PcoDecompTest, PcoBinSplitSimdLutAndOffsetTiers)
{
    struct TierCase {
        size_t numClusters;
        uint64_t spreadMask; // 0 => allZeroOffsets; >0 => maxOb tier
    };
    const TierCase tiers[] = {
        { 6, 0ULL },                  // <=8/16 LUT, allZeroOffsets
        { 12, 0xFFULL },              // <=16 byte-plane LUT, ob<=8
        { 40, 0x7FFFULL },            // 17..64 multi-register LUT, ob<=15
        { 48, 0x1FFFFFULL },          // 17..64 LUT, ob==21 (32-bit extractor)
        { 140, 0x3FFFFFFFFFULL },     // >64 scalar LUT, ob==38 (64-bit extractor)
    };
    const size_t lengths[] = { 15, 16, 17, 31, 32, 33, 255, 256, 257, 600 };

    auto checkTiers = [&](auto tag, nodes::PcoBinSplit::LatentMode mode) {
        using T            = decltype(tag);
        constexpr size_t W = sizeof(T) * 8u;
        const uint64_t typeMask =
                (W == 64u) ? ~uint64_t(0) : ((uint64_t(1) << W) - 1ULL);
        compressor_.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
        compressor_.selectStartingGraph(nodes::PcoBinSplit{ mode }(
                compressor_, graphs::Entropy{}(), graphs::Store{}()));

        for (const auto& tc : tiers) {
            uint64_t const mask = tc.spreadMask & (typeMask >> 2);
            uint64_t const step =
                    std::max<uint64_t>(mask + 17ULL, (typeMask / 200ULL) | 1ULL);
            for (size_t len : lengths) {
                std::vector<T> data(len);
                for (size_t i = 0; i < len; ++i) {
                    size_t const c   = (i * 7u + (i >> 3)) % tc.numClusters;
                    uint64_t const v = (uint64_t(c) * step
                                        + ((i * 0x9E3779B97F4A7C15ULL) & mask))
                            & typeMask;
                    data[i] = static_cast<T>(v);
                }
                testRoundTrip(Input::refNumeric(poly::span<const T>{ data }));
            }
        }
    };

    checkTiers(uint8_t{}, nodes::PcoBinSplit::LatentMode::Unsigned);
    checkTiers(int8_t{}, nodes::PcoBinSplit::LatentMode::Signed);
    checkTiers(uint16_t{}, nodes::PcoBinSplit::LatentMode::Unsigned);
    checkTiers(int16_t{}, nodes::PcoBinSplit::LatentMode::Signed);
    checkTiers(uint32_t{}, nodes::PcoBinSplit::LatentMode::Unsigned);
    checkTiers(int32_t{}, nodes::PcoBinSplit::LatentMode::Signed);
    checkTiers(uint64_t{}, nodes::PcoBinSplit::LatentMode::Unsigned);
    checkTiers(int64_t{}, nodes::PcoBinSplit::LatentMode::Signed);
}

TEST_F(PcoDecompTest, PcoBinSplitFloat16AndCorruptBinRejection)
{
    compressor_.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    compressor_.selectStartingGraph(
            nodes::PcoBinSplit{ nodes::PcoBinSplit::LatentMode::Float }(
                    compressor_, graphs::Entropy{}(), graphs::Store{}()));

    // 16-bit IEEE-754 half-precision bit patterns (+0, -0, normals, negatives, NaNs)
    for (size_t len : { size_t(15), size_t(16), size_t(17), size_t(33), size_t(257) }) {
        std::vector<uint16_t> f16Bits(len);
        for (size_t i = 0; i < len; ++i) {
            if (i == 0) {
                f16Bits[i] = 0x8000u; // -0.0
            } else if (i == 1) {
                f16Bits[i] = 0x7E21u; // quiet NaN payload
            } else {
                uint16_t const mag = static_cast<uint16_t>(0x3C00u + ((i * 19u) & 0x03FFu));
                f16Bits[i] = (i & 1u) ? static_cast<uint16_t>(mag | 0x8000u) : mag;
            }
        }
        testRoundTrip(Input::refNumeric(poly::span<const uint16_t>{ f16Bits }));
    }

    // Out-of-bounds binId (>= numBins) must be rejected by decoder bin validation
    compressor_.setParameter(CParam::CompressedChecksum, ZL_TernaryParam_disable);
    compressor_.setParameter(CParam::ContentChecksum, ZL_TernaryParam_disable);
    compressor_.selectStartingGraph(
            nodes::PcoBinSplit{ nodes::PcoBinSplit::LatentMode::Unsigned }(
                    compressor_, graphs::Store{}(), graphs::Store{}()));
    std::vector<uint32_t> twoCluster(64);
    for (size_t i = 0; i < twoCluster.size(); ++i) {
        twoCluster[i] = (i & 1u) ? (100000u + uint32_t(i & 7u)) : (10u + uint32_t(i & 7u));
    }
    cctx_.refCompressor(compressor_);
    std::string frame = cctx_.compressOne(
            Input::refNumeric(poly::span<const uint32_t>{ twoCluster }));

    std::string pattern(64, '\0');
    for (size_t i = 0; i < pattern.size(); ++i) {
        pattern[i] = static_cast<char>(i & 1u);
    }
    size_t const pos = frame.find(pattern);
    ASSERT_NE(pos, std::string::npos);

    frame[pos + 17] = 0; // Valid binId < numBins succeeds (positive control)
    EXPECT_NO_THROW((void)dctx_.decompressOne(frame));

    for (size_t lane : { size_t(17), size_t(61) }) {
        for (uint8_t badBin : { uint8_t(2), uint8_t(0xFE) }) {
            std::string bad = frame;
            bad[pos + lane] = static_cast<char>(badBin);
            EXPECT_ANY_THROW((void)dctx_.decompressOne(bad));
        }
    }
}

TEST_F(PcoDecompTest, IntMultSplitDetectsMultiplierAndFallsBack)
{
    compressor_.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);

    // 1. Multiples of 1,000,000 + offset + 5% noisy outliers -> auto-detects base
    size_t adjCount = 0;
    auto capAdj = compressor_.registerFunctionGraph(
            std::make_shared<CaptureLookbackCountGraph>(&adjCount));
    auto multNode = nodes::IntMultSplit{ nodes::IntMultSplit::Mode::Unsigned }
                            .parameterize(compressor_);
    auto gCap = compressor_.buildStaticGraph(
            multNode, { graphs::Store{}(), capAdj });
    compressor_.selectStartingGraph(gCap);

    std::vector<uint64_t> scaled(2048);
    for (size_t i = 0; i < scaled.size(); ++i) {
        scaled[i] = (i * 7ULL + 123ULL) * 1000000ULL + 42ULL;
        if (i % 20 == 0) {
            scaled[i] += 3ULL;
        }
    }
    testRoundTrip(Input::refNumeric(poly::span<const uint64_t>{ scaled }));
    EXPECT_EQ(adjCount, scaled.size());

    // End-to-end compression ratio with Delta + PcoBinSplit on quotients
    auto qGraph = nodes::DeltaInt{}(
            compressor_,
            nodes::PcoBinSplit{ nodes::PcoBinSplit::LatentMode::Signed }(
                    compressor_, graphs::Entropy{}(), graphs::Store{}()));
    auto gComp = nodes::IntMultSplit{ nodes::IntMultSplit::Mode::Unsigned }(
            compressor_, qGraph, graphs::Compress{}());
    compressor_.selectStartingGraph(gComp);
    auto comp = testRoundTrip(
            Input::refNumeric(poly::span<const uint64_t>{ scaled }));
    EXPECT_LT(
            static_cast<double>(comp.size()),
            0.15 * static_cast<double>(scaled.size() * sizeof(uint64_t)));

    // 2. Signed mode on negative/positive multiples of 16
    std::vector<int32_t> signedScaled(1024);
    for (size_t i = 0; i < signedScaled.size(); ++i) {
        int32_t k = static_cast<int32_t>((i & 31) - 16);
        signedScaled[i] = k * 16 + ((i % 25 == 0) ? 1 : 0);
    }
    compressor_.selectStartingGraph(
            nodes::IntMultSplit{ nodes::IntMultSplit::Mode::Signed }(
                    compressor_,
                    nodes::PcoBinSplit{ nodes::PcoBinSplit::LatentMode::Signed }(
                            compressor_, graphs::Entropy{}(), graphs::Store{}()),
                    graphs::Compress{}()));
    auto compS = testRoundTrip(
            Input::refNumeric(poly::span<const int32_t>{ signedScaled }));
    EXPECT_LT(
            static_cast<double>(compS.size()),
            0.20 * static_cast<double>(signedScaled.size() * sizeof(int32_t)));

    // 3. No multiplier (pseudorandom) -> falls back to base=1 (0 adjustments)
    adjCount = 999999;
    compressor_.selectStartingGraph(gCap);
    std::vector<uint32_t> dense(1024);
    uint64_t rng = 0x123456789abcdef0ULL;
    for (size_t i = 0; i < dense.size(); ++i) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        dense[i] = static_cast<uint32_t>(rng >> 32);
    }
    testRoundTrip(Input::refNumeric(poly::span<const uint32_t>{ dense }));
    EXPECT_EQ(adjCount, 0u);
}

TEST_F(PcoDecompTest, FloatMultSplitDetectsMultipliersAndPreservesAllBits)
{
    compressor_.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);

    size_t adjCount = 0;
    auto capAdj     = compressor_.registerFunctionGraph(
            std::make_shared<CaptureLookbackCountGraph>(&adjCount));
    auto multNode = nodes::FloatMultSplit{}.parameterize(compressor_);
    auto gCap     = compressor_.buildStaticGraph(
            multNode, { graphs::Store{}(), capAdj });

    auto signedPco = nodes::PcoBinSplit{
        nodes::PcoBinSplit::LatentMode::Signed
    }(compressor_, graphs::Entropy{}(), graphs::Store{}());
    auto unsignedPco = nodes::PcoBinSplit{
        nodes::PcoBinSplit::LatentMode::Unsigned
    }(compressor_, graphs::Entropy{}(), graphs::Store{}());
    auto lbdGraph = compressor_.buildStaticGraph(
            nodes::LookbackDelta{}.parameterize(compressor_),
            { signedPco, unsignedPco });
    auto gComp = nodes::FloatMultSplit{}(compressor_, lbdGraph, signedPco);

    // 1. f64 decimal multiples (0.001) + special bit patterns (-0.0, +/-Inf, NaN payloads)
    std::vector<double> f64Scaled(2048);
    for (size_t i = 0; i < f64Scaled.size(); ++i) {
        f64Scaled[i] = static_cast<double>(static_cast<int64_t>(i) - 512) * 0.001;
        if (i % 37 == 0) {
            uint64_t bits;
            std::memcpy(&bits, &f64Scaled[i], sizeof(bits));
            bits += 1ULL;
            std::memcpy(&f64Scaled[i], &bits, sizeof(bits));
        }
    }
    f64Scaled[0] = -0.0;
    f64Scaled[1] = std::numeric_limits<double>::infinity();
    f64Scaled[2] = -std::numeric_limits<double>::infinity();
    {
        uint64_t qnan = 0x7ff8000000001234ULL;
        uint64_t snan = 0xfff0000000005678ULL;
        uint64_t subn = 0x0000000000000003ULL;
        std::memcpy(&f64Scaled[3], &qnan, sizeof(qnan));
        std::memcpy(&f64Scaled[4], &snan, sizeof(snan));
        std::memcpy(&f64Scaled[5], &subn, sizeof(subn));
    }

    compressor_.selectStartingGraph(gCap);
    testRoundTrip(Input::refNumeric(poly::span<const double>{ f64Scaled }));
    EXPECT_EQ(adjCount, f64Scaled.size());

    compressor_.selectStartingGraph(gComp);
    auto compF64 = testRoundTrip(
            Input::refNumeric(poly::span<const double>{ f64Scaled }));
    EXPECT_LT(
            static_cast<double>(compF64.size()),
            0.15 * static_cast<double>(f64Scaled.size() * sizeof(double)));

    // 2. f64 trailing-zero integer multiples (e.g. gps_time seconds)
    std::vector<double> f64Sec(1024);
    for (size_t i = 0; i < f64Sec.size(); ++i) {
        f64Sec[i] = 381355792.0 + static_cast<double>(i * 3);
    }
    adjCount = 0;
    compressor_.selectStartingGraph(gCap);
    testRoundTrip(Input::refNumeric(poly::span<const double>{ f64Sec }));
    EXPECT_EQ(adjCount, f64Sec.size());

    compressor_.selectStartingGraph(gComp);
    auto compSec = testRoundTrip(
            Input::refNumeric(poly::span<const double>{ f64Sec }));
    EXPECT_LT(
            static_cast<double>(compSec.size()),
            0.05 * static_cast<double>(f64Sec.size() * sizeof(double)));

    // 3. f32 decimal + reciprocal multiples (0.1f and 1/7) with edge floats
    std::vector<float> f32Scaled(2048);
    for (size_t i = 0; i < f32Scaled.size(); ++i) {
        f32Scaled[i] = static_cast<float>(static_cast<int32_t>(i) - 512) * 0.1f;
    }
    f32Scaled[0] = -0.0f;
    {
        uint32_t qnan = 0x7fc01234u;
        uint32_t snan = 0xff800007u;
        uint32_t subn = 0x00000005u;
        std::memcpy(&f32Scaled[1], &qnan, sizeof(qnan));
        std::memcpy(&f32Scaled[2], &snan, sizeof(snan));
        std::memcpy(&f32Scaled[3], &subn, sizeof(subn));
    }
    adjCount = 0;
    compressor_.selectStartingGraph(gCap);
    testRoundTrip(Input::refNumeric(poly::span<const float>{ f32Scaled }));
    EXPECT_EQ(adjCount, f32Scaled.size());

    compressor_.selectStartingGraph(gComp);
    auto compF32 = testRoundTrip(
            Input::refNumeric(poly::span<const float>{ f32Scaled }));
    EXPECT_LT(
            static_cast<double>(compF32.size()),
            0.15 * static_cast<double>(f32Scaled.size() * sizeof(float)));

    // 4. Random non-multiple f64 -> fallback (0 adjustments) + bit-exact roundtrip
    adjCount = 999999;
    compressor_.selectStartingGraph(gCap);
    std::vector<uint64_t> randomBits(1024);
    uint64_t rng = 0x123456789abcdef0ULL;
    for (size_t i = 0; i < randomBits.size(); ++i) {
        rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
        randomBits[i] = rng;
    }
    testRoundTrip(Input::refNumeric(poly::span<const uint64_t>{ randomBits }));
    EXPECT_EQ(adjCount, 0u);
}

TEST_F(PcoDecompTest, PcoBinSplitSampledNearUint64MaxTightGap)
{
    compressor_.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    auto graph = nodes::PcoBinSplit{ nodes::PcoBinSplit::LatentMode::Unsigned }(
            compressor_, graphs::Store{}(), graphs::Store{}());
    compressor_.selectStartingGraph(graph);

    // n > PBS_SAMPLE_MAX (8192) with values near UINT64_MAX so unsaturating
    // (lower + mask64(ob)) wraps and shrinks offsetBits during anyTightGap.
    constexpr size_t kN     = 10000;
    constexpr uint64_t kLo  = UINT64_MAX - 4194239ULL; // kLo + mask64(22) wraps to 63
    std::vector<uint64_t> data(kN);
    for (size_t i = 0; i < kN; ++i) {
        if (i < 9920) {
            data[i] = UINT64_MAX - 100000000ULL + (i / 80u) * 100000ULL;
        } else {
            data[i] = kLo + (i & 7u);
        }
    }
    data[9990] = kLo + 4194200ULL; // sampled in top bin -> forces ob=22
    data[9999] = kLo + 16000ULL;   // unsampled tail -> shrinks ob if maxSafe wraps

    testRoundTrip(Input::refNumeric(poly::span<const uint64_t>{ data }));
}

TEST_F(PcoDecompTest, PcoBinSplitExactU16And256Bins)
{
    compressor_.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    auto graph = nodes::PcoBinSplit{ nodes::PcoBinSplit::LatentMode::Unsigned }(
            compressor_, graphs::Entropy{}(), graphs::Store{}());
    compressor_.selectStartingGraph(graph);

    // 256 sparse spikes spaced by 256 in uint16_t: exact u16 histogram finds
    // all 256 zero-offset bins without sampling or radix sorting.
    std::vector<uint16_t> sparse256(4096);
    for (size_t i = 0; i < sparse256.size(); ++i) {
        sparse256[i] = static_cast<uint16_t>((i & 255u) * 256u);
    }
    auto comp = testRoundTrip(
            Input::refNumeric(poly::span<const uint16_t>{ sparse256 }));
    // With 256 zero-offset bins, packed offset bytes are 0 and bin stream
    // compresses to ~8 bits/elem + ~1 KB header (< 5 KB total vs 8192 raw).
    EXPECT_LT(comp.size(), 5000u);

    // Also verify n >= 512 with narrow tight clusters and rare outliers
    // (non-trivial offsets + >128 bins).
    std::vector<uint16_t> clustered(10000);
    for (size_t i = 0; i < clustered.size(); ++i) {
        clustered[i] = static_cast<uint16_t>((i % 200u) * 300u + (i & 3u));
    }
    clustered[9997] = 65535u;
    testRoundTrip(Input::refNumeric(poly::span<const uint16_t>{ clustered }));
}

TEST_F(PcoDecompTest, PcoBinSplitPowerOfTwoRefinementAndSkewedUnique)
{
    compressor_.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    auto graph = nodes::PcoBinSplit{ nodes::PcoBinSplit::LatentMode::Unsigned }(
            compressor_, graphs::Entropy{}(), graphs::Bitpack{}());
    compressor_.selectStartingGraph(graph);

    // 1. Skewed u16 stream with 200 distinct values (<= 256): adaptive exact
    // runs keep zero-offset bins instead of merging consecutive heavy values.
    std::vector<uint16_t> skewed200(32768);
    for (size_t i = 0; i < skewed200.size(); ++i) {
        uint32_t const r = static_cast<uint32_t>((i * 2654435761u) >> 16);
        if ((r & 3u) == 0u) {
            skewed200[i] = 0u;
        } else {
            uint16_t const v = static_cast<uint16_t>(1u + ((r * r) >> 24) % 199u);
            skewed200[i]     = static_cast<uint16_t>(v * 17u);
        }
    }
    auto compSkewed = testRoundTrip(
            Input::refNumeric(poly::span<const uint16_t>{ skewed200 }));
    EXPECT_LT(compSkewed.size(), 28500u);

    // 2. Sampled u32 stream (n > 8192, > 256 distinct values) with a dense
    // power-of-2 cluster [1000..1015] plus wide tail and unsampled gap values.
    std::vector<uint32_t> powerAligned(32768);
    for (size_t i = 0; i < powerAligned.size(); ++i) {
        uint32_t const h = static_cast<uint32_t>(i * 2654435761u);
        if ((h & 7u) != 0u) {
            powerAligned[i] = 1000u + (h >> 28); // exact 16-value [1000..1015]
        } else {
            powerAligned[i] = 1016u + ((h >> 16) % 3500u);
        }
    }
    powerAligned[32761] = 2047u;
    powerAligned[32763] = 2048u;
    auto compAligned = testRoundTrip(
            Input::refNumeric(poly::span<const uint32_t>{ powerAligned }));
    EXPECT_LT(compAligned.size(), 36000u);
}

} // namespace
} // namespace tests
} // namespace openzl
