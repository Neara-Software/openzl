// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "openzl/codecs/ctx_split/decode_ctx_split_binding.h"
#include "openzl/codecs/zl_ctx_split.h"
#include "openzl/common/wire_format.h"
#include "openzl/cpp/codecs/CtxSplit.hpp"
#include "openzl/cpp/codecs/Delta.hpp"
#include "openzl/cpp/codecs/PcoBinSplit.hpp"
#include "openzl/openzl.hpp"
#include "openzl/zl_ctransform.h"
#include "openzl/zl_dtransform.h"
#include "openzl/zl_reflection.h"
#include "tests/codecs/test_codec.h"

namespace openzl {
namespace tests {
namespace {

class CountVOStreamsGraph : public FunctionGraph {
   public:
    CountVOStreamsGraph(size_t* streamCount, GraphID downstream)
            : streamCount_(streamCount), downstream_(downstream)
    {
    }

    FunctionGraphDescription functionGraphDescription() const override
    {
        return FunctionGraphDescription{
            .name           = "CountVOStreams",
            .inputTypeMasks = { TypeMask::Serial | TypeMask::Numeric },
        };
    }

    void graph(GraphState& state) const override
    {
        (*streamCount_)++;
        state.edges()[0].setDestination(downstream_);
    }

   private:
    size_t* streamCount_;
    GraphID downstream_;
};

struct RawCodecCase {
    std::vector<uint8_t> header;
    std::vector<std::vector<uint32_t>> streams;
};

static ZL_Report rawInjectEncoder(
        ZL_Encoder* eictx,
        const ZL_Input* in) noexcept
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(eictx);
    (void)in;
    const auto* spec =
            static_cast<const RawCodecCase*>(ZL_Encoder_getOpaquePtr(eictx));
    if (!spec->header.empty()) {
        ZL_Encoder_sendCodecHeader(
                eictx, spec->header.data(), spec->header.size());
    }
    for (const auto& s : spec->streams) {
        ZL_Output* out =
                ZL_Encoder_createTypedStream(eictx, 0, s.size(), sizeof(uint32_t));
        ZL_ERR_IF_NULL(out, allocation);
        if (!s.empty()) {
            std::memcpy(
                    ZL_Output_ptr(out),
                    s.data(),
                    s.size() * sizeof(uint32_t));
        }
        ZL_ERR_IF_ERR(ZL_Output_commit(out, s.size()));
    }
    return ZL_returnSuccess();
}

class CtxSplitTest : public CodecTest {
   protected:
    template <typename T>
    std::vector<T> makeBimodalLinearMarkov(
            size_t n,
            uint64_t minOffset,
            uint64_t step)
    {
        std::vector<T> out(n);
        uint64_t rng = 0x9E3779B97F4A7C15ULL;
        uint32_t cls = 0;
        for (size_t i = 0; i < n; ++i) {
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            // Given predecessor class `cls`, current element deterministically
            // jumps to class `(cls * 5 + 3) & 7` with only 2-bit jitter.
            uint32_t nextCls = ((rng >> 58) == 0ULL) ? ((cls + 1u) & 7u)
                                                     : ((cls * 5u + 3u) & 7u);
            uint64_t val     = minOffset + uint64_t(nextCls) * step
                    + ((rng >> 60) & 3ULL);
            out[i] = static_cast<T>(val);
            cls    = nextCls;
        }
        return out;
    }

    template <typename T>
    std::vector<T> makeLog2MagnitudeMarkov(size_t n)
    {
        // One huge outlier collapses Mode 0's linear buckets into class 0 while
        // Mode 1 (bit_length(prev)) cleanly separates small magnitude states.
        std::vector<T> out(n);
        uint64_t rng  = 0xD1B54A32D192ED03ULL;
        uint32_t prev = 0;
        for (size_t i = 0; i < n; ++i) {
            if (i == n / 2) {
                out[i] = std::numeric_limits<T>::max();
                prev   = 7u;
                continue;
            }
            rng = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            // Map prev magnitude class -> distinct tight value cluster
            uint32_t mag = (prev == 0u) ? 4u
                    : (prev <= 2u)      ? 1u
                    : (prev <= 4u)      ? 6u
                                        : 0u;
            if (((rng >> 58) & 15ULL) == 0ULL) {
                mag = static_cast<uint32_t>((rng >> 52) % 7ULL);
            }
            uint64_t v = (mag == 0u)
                    ? 0ULL
                    : ((1ULL << (mag - 1u))
                       | ((rng >> 56) & ((1ULL << (mag - 1u)) - 1ULL)));
            out[i]     = static_cast<T>(v);
            uint64_t x = static_cast<uint64_t>(out[i]);
            uint32_t bl = 0;
            while (x != 0) {
                bl++;
                x >>= 1;
            }
            prev = bl > 7u ? 7u : bl;
        }
        return out;
    }

    void expectRawCaseFails(const RawCodecCase& spec)
    {
        constexpr ZL_IDType kCustomId = 4242;
        ZL_Compressor* cgraph         = ZL_Compressor_create();
        ASSERT_NE(cgraph, nullptr);
        EXPECT_FALSE(ZL_isError(ZL_Compressor_setParameter(
                cgraph, ZL_CParam_formatVersion, ZL_MAX_FORMAT_VERSION)));
        EXPECT_FALSE(ZL_isError(ZL_Compressor_setParameter(
                cgraph, ZL_CParam_minStreamSize, -1)));
        EXPECT_FALSE(ZL_isError(ZL_Compressor_setParameter(
                cgraph, ZL_CParam_storeOnExpansion, ZL_TernaryParam_disable)));

        const ZL_Type voType  = ZL_Type_numeric;
        ZL_VOEncoderDesc encD = {};
        encD.gd.CTid          = kCustomId;
        encD.gd.inStreamType  = ZL_Type_numeric;
        encD.gd.voTypes       = &voType;
        encD.gd.nbVOs         = 1;
        encD.transform_f      = rawInjectEncoder;
        encD.name             = "raw_ctx_split_inject";
        encD.opaque.ptr       = const_cast<RawCodecCase*>(&spec);
        ZL_NodeID nid   = ZL_Compressor_registerVOEncoder(cgraph, &encD);
        ZL_GraphID store = ZL_GRAPH_STORE;
        ZL_GraphID gid   = ZL_Compressor_registerStaticGraph_fromNode(
                cgraph, nid, &store, 1);
        EXPECT_FALSE(
                ZL_isError(ZL_Compressor_selectStartingGraphID(cgraph, gid)));

        ZL_CCtx* cctx = ZL_CCtx_create();
        ASSERT_NE(cctx, nullptr);
        EXPECT_FALSE(ZL_isError(ZL_CCtx_refCompressor(cctx, cgraph)));

        uint32_t dummy[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
        ZL_TypedRef* tref =
                ZL_TypedRef_createNumeric(dummy, sizeof(uint32_t), 8);
        std::string compressed(4096, '\0');
        ZL_Report cRep = ZL_CCtx_compressTypedRef(
                cctx, compressed.data(), compressed.size(), tref);
        ASSERT_FALSE(ZL_isError(cRep));
        compressed.resize(ZL_validResult(cRep));

        ZL_DCtx* dctx = ZL_DCtx_create();
        ASSERT_NE(dctx, nullptr);
        ZL_VODecoderDesc decD = {};
        decD.gd.CTid          = kCustomId;
        decD.gd.inStreamType  = ZL_Type_numeric;
        decD.gd.voTypes       = &voType;
        decD.gd.nbVOs         = 1;
        decD.transform_f      = DI_ctx_split;
        decD.name             = "raw_ctx_split_decode";
        EXPECT_FALSE(ZL_isError(ZL_DCtx_registerVODecoder(dctx, &decD)));

        uint32_t outBuf[64]    = { 0 };
        ZL_TypedBuffer* outTbuf = ZL_TypedBuffer_createWrapNumeric(
                outBuf, sizeof(uint32_t), 64);
        ZL_Report dRep = ZL_DCtx_decompressTBuffer(
                dctx, outTbuf, compressed.data(), compressed.size());
        EXPECT_TRUE(ZL_isError(dRep));

        ZL_TypedBuffer_free(outTbuf);
        ZL_TypedRef_free(tref);
        ZL_DCtx_free(dctx);
        ZL_CCtx_free(cctx);
        ZL_Compressor_free(cgraph);
    }
};

TEST_F(CtxSplitTest, PassthroughOnTinyConstantAndRandomAcrossAllWidths)
{
    compressor_.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);

    // 1. Serial passthrough: empty, tiny (< 256), constant, white noise (4K & 80K sampled)
    for (int nCtxLog : { 1, 2, 3 }) {
        size_t voCount = 0;
        auto countGraph = compressor_.registerFunctionGraph(
                std::make_shared<CountVOStreamsGraph>(
                        &voCount, graphs::Store{}()));
        auto g = compressor_.buildStaticGraph(
                nodes::CtxSplit{ nCtxLog }.parameterize(compressor_),
                { countGraph });
        compressor_.selectStartingGraph(g);

        for (size_t sz : { size_t(0), size_t(1), size_t(17), size_t(255) }) {
            voCount = 0;
            std::string tiny(sz, 'q');
            testRoundTrip(Input::refSerial(tiny));
            EXPECT_EQ(voCount, 1u);
        }

        voCount = 0;
        std::string constant(1024, 'k');
        testRoundTrip(Input::refSerial(constant));
        EXPECT_EQ(voCount, 1u);

        for (size_t sz : { size_t(4096), size_t(80000) }) {
            voCount = 0;
            std::string noise(sz, '\0');
            uint64_t rng = 0xA5A5A5A55A5A5A5AULL ^ sz;
            for (size_t i = 0; i < sz; ++i) {
                rng      = rng * 6364136223846793005ULL + 1442695040888963407ULL;
                noise[i] = static_cast<char>(rng >> 56);
            }
            testRoundTrip(Input::refSerial(noise));
            EXPECT_EQ(voCount, 1u);
        }
    }

    // 2. Numeric passthrough across u8, u16, u32, u64
    auto checkNumPassthrough = [&](auto tag) {
        using T        = decltype(tag);
        size_t voCount = 0;
        auto countGraph = compressor_.registerFunctionGraph(
                std::make_shared<CountVOStreamsGraph>(
                        &voCount, graphs::Store{}()));
        compressor_.selectStartingGraph(compressor_.buildStaticGraph(
                nodes::CtxSplitNum{ 3 }.parameterize(compressor_),
                { countGraph }));

        voCount = 0;
        std::vector<T> empty;
        testRoundTrip(Input::refNumeric(poly::span<const T>{ empty }));
        EXPECT_EQ(voCount, 1u);

        voCount = 0;
        std::vector<T> tiny(255, static_cast<T>(9));
        testRoundTrip(Input::refNumeric(poly::span<const T>{ tiny }));
        EXPECT_EQ(voCount, 1u);

        voCount = 0;
        std::vector<T> constant(1024, static_cast<T>(77));
        testRoundTrip(Input::refNumeric(poly::span<const T>{ constant }));
        EXPECT_EQ(voCount, 1u);

        voCount = 0;
        std::vector<T> noise(4096);
        uint64_t rng = 0x1122334455667788ULL + sizeof(T);
        for (size_t i = 0; i < noise.size(); ++i) {
            rng      = rng * 6364136223846793005ULL + 1442695040888963407ULL;
            noise[i] = static_cast<T>(rng >> (64 - sizeof(T) * 8));
        }
        testRoundTrip(Input::refNumeric(poly::span<const T>{ noise }));
        EXPECT_EQ(voCount, 1u);
    };
    checkNumPassthrough(uint8_t{});
    checkNumPassthrough(uint16_t{});
    checkNumPassthrough(uint32_t{});
    checkNumPassthrough(uint64_t{});
}

TEST_F(CtxSplitTest, SplitsMarkovStreamsAndBeatsSingleStreamFse)
{
    compressor_.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);

    // 1. Serial bimodal/order-1 Markov stream: CtxSplit + Fse beats bare Fse
    std::string serialMarkov(8192, '\0');
    {
        auto v = makeBimodalLinearMarkov<uint8_t>(8192, 16, 28);
        std::memcpy(serialMarkov.data(), v.data(), v.size());
    }
    compressor_.selectStartingGraph(graphs::Fse{}());
    auto bareSerial = testRoundTrip(Input::refSerial(serialMarkov));

    for (int nCtxLog : { 1, 2, 3 }) {
        size_t voCount  = 0;
        auto countGraph = compressor_.registerFunctionGraph(
                std::make_shared<CountVOStreamsGraph>(
                        &voCount, graphs::Fse{}()));
        compressor_.selectStartingGraph(compressor_.buildStaticGraph(
                nodes::CtxSplit{ nCtxLog }.parameterize(compressor_),
                { countGraph }));
        auto splitSerial = testRoundTrip(Input::refSerial(serialMarkov));
        EXPECT_GT(voCount, 1u);
        EXPECT_LE(voCount, size_t(1u << nCtxLog));
        if (nCtxLog == 3) {
            EXPECT_LT(splitSerial.size() + 500, bareSerial.size());
        }
    }

    // 2. Numeric Mode 0 (linear shift, minVal > 0 so prev=0 < minVal at i=0) across u8/u16/u32/u64 + 80K sampled
    auto checkLinearWidth = [&](auto tag, uint64_t minOff, uint64_t step, size_t len) {
        using T = decltype(tag);
        auto data = makeBimodalLinearMarkov<T>(len, minOff, step);

        compressor_.selectStartingGraph(graphs::Fse{}());
        auto bareComp =
                testRoundTrip(Input::refNumeric(poly::span<const T>{ data }));

        size_t voCount  = 0;
        auto countGraph = compressor_.registerFunctionGraph(
                std::make_shared<CountVOStreamsGraph>(
                        &voCount, graphs::Fse{}()));
        compressor_.selectStartingGraph(compressor_.buildStaticGraph(
                nodes::CtxSplitNum{ 3 }.parameterize(compressor_),
                { countGraph }));
        auto splitComp =
                testRoundTrip(Input::refNumeric(poly::span<const T>{ data }));
        EXPECT_GT(voCount, 1u);
        EXPECT_LT(splitComp.size(), bareComp.size());
    };
    checkLinearWidth(uint8_t{}, 12, 28, 8192);
    checkLinearWidth(uint16_t{}, 1000, 7000, 8192);
    checkLinearWidth(uint32_t{}, 50000, 200000000ULL, 8192);
    checkLinearWidth(uint64_t{}, 900000ULL, 1000000000000000ULL, 8192);
    // Sampled path (> 65536 elements)
    checkLinearWidth(uint32_t{}, 1000, 10000000ULL, 80000);

    // 3. Numeric Mode 1 (log2 magnitude with extreme outlier) across u16/u32/u64
    auto checkLog2Width = [&](auto tag) {
        using T   = decltype(tag);
        auto data = makeLog2MagnitudeMarkov<T>(8192);

        size_t voCount  = 0;
        auto countGraph = compressor_.registerFunctionGraph(
                std::make_shared<CountVOStreamsGraph>(
                        &voCount, graphs::Fse{}()));
        compressor_.selectStartingGraph(compressor_.buildStaticGraph(
                nodes::CtxSplitNum{ 3 }.parameterize(compressor_),
                { countGraph }));
        auto comp =
                testRoundTrip(Input::refNumeric(poly::span<const T>{ data }));
        EXPECT_GT(voCount, 1u);
        (void)comp;
    };
    checkLog2Width(uint16_t{});
    checkLog2Width(uint32_t{});
    checkLog2Width(uint64_t{});
}

TEST_F(CtxSplitTest, PcoBinSplitPlusCtxSplitPipelineAndParamClamping)
{
    compressor_.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);

    // 1. Out-of-range C API local param values clamp cleanly to [1, 3]
    for (int rawParam : { -5, 0, 4, 99 }) {
        ZL_IntParam ip              = { ZL_CTX_SPLIT_NCTX_LOG_PID, rawParam };
        ZL_LocalParams lp           = {};
        lp.intParams.intParams      = &ip;
        lp.intParams.nbIntParams    = 1;
        ZL_ParameterizedNodeDesc pd = {
            .node        = ZL_NODE_CTX_SPLIT_NUM,
            .localParams = &lp,
        };
        ZL_NodeID nid = ZL_Compressor_registerParameterizedNode(
                compressor_.get(), &pd);
        ASSERT_NE(nid.nid, ZL_NODE_ILLEGAL.nid);
        GraphID fse = graphs::Fse{}();
        ZL_GraphID gid = ZL_Compressor_registerStaticGraph_fromNode1o(
                compressor_.get(), nid, fse);
        compressor_.selectStartingGraph(gid);

        auto data = makeBimodalLinearMarkov<uint32_t>(2048, 100, 100000);
        testRoundTrip(Input::refNumeric(poly::span<const uint32_t>{ data }));
    }

    // 2. DeltaInt -> PcoBinSplit -> [CtxSplitNum -> Fse, Store] pipeline
    auto ctxFse = nodes::CtxSplitNum{ 3 }(compressor_, graphs::Fse{}());
    auto pco    = nodes::PcoBinSplit{
        nodes::PcoBinSplit::LatentMode::Signed
    }(compressor_, ctxFse, graphs::Store{}());
    auto g = nodes::DeltaInt{}(compressor_, pco);
    compressor_.selectStartingGraph(g);

    std::vector<int64_t> cumMarkov(4096);
    auto deltas  = makeBimodalLinearMarkov<uint16_t>(4096, 10, 2000);
    int64_t acc  = 1000000;
    for (size_t i = 0; i < cumMarkov.size(); ++i) {
        acc += static_cast<int64_t>(deltas[i]) - 7000;
        cumMarkov[i] = acc;
    }
    testRoundTrip(Input::refNumeric(poly::span<const int64_t>{ cumMarkov }));
}

TEST_F(CtxSplitTest, ReflectionAndCorruptHeaderRejection)
{
    // Verify ZL_ReflectionCtx reports ZL_StandardTransformID_ctx_split = 76
    compressor_.setParameter(CParam::FormatVersion, ZL_MAX_FORMAT_VERSION);
    compressor_.setParameter(
            CParam::CompressedChecksum, ZL_TernaryParam_disable);
    compressor_.setParameter(CParam::ContentChecksum, ZL_TernaryParam_disable);
    compressor_.selectStartingGraph(
            nodes::CtxSplitNum{ 3 }(compressor_, graphs::Store{}()));

    auto data = makeBimodalLinearMarkov<uint32_t>(2048, 500, 100000);
    cctx_.refCompressor(compressor_);
    std::string frame = cctx_.compressOne(
            Input::refNumeric(poly::span<const uint32_t>{ data }));

    ZL_ReflectionCtx* rctx = ZL_ReflectionCtx_create();
    ASSERT_NE(rctx, nullptr);
    ASSERT_FALSE(ZL_isError(ZL_ReflectionCtx_setCompressedFrame(
            rctx, frame.data(), frame.size())));
    size_t const nbCodecs = ZL_ReflectionCtx_getNumCodecs_lastChunk(rctx);
    ASSERT_GE(nbCodecs, 1u);
    const ZL_CodecInfo* ci = nullptr;
    for (size_t k = 0; k < nbCodecs; ++k) {
        const ZL_CodecInfo* cand = ZL_ReflectionCtx_getCodec_lastChunk(rctx, k);
        if (cand != nullptr
            && ZL_CodecInfo_getCodecID(cand)
                    == ZL_StandardTransformID_ctx_split) {
            ci = cand;
            break;
        }
    }
    ASSERT_NE(ci, nullptr);
    EXPECT_EQ(ZL_StandardTransformID_ctx_split, 76);

    const void* hdrPtr = ZL_CodecInfo_getHeaderPtr(ci);
    size_t hdrSize     = ZL_CodecInfo_getHeaderSize(ci);
    ASSERT_GE(hdrSize, 4u);
    std::string hdrBytes(static_cast<const char*>(hdrPtr), hdrSize);
    ZL_ReflectionCtx_free(rctx);

    size_t hdrPos = frame.find(hdrBytes);
    ASSERT_NE(hdrPos, std::string::npos);

    // Mutating flagsByte to invalid mode=3, nCtxLog=0, nCtxLog=4 in real frame fails
    for (uint8_t badFlags : { uint8_t(0x33), uint8_t(0x00), uint8_t(0x04) }) {
        std::string bad = frame;
        bad[hdrPos]     = static_cast<char>(badFlags);
        EXPECT_ANY_THROW((void)dctx_.decompressOne(bad));
    }
    // Mutating activeMask to 0 or toggling a bit (popcount != nbStreams) fails
    {
        std::string badZero = frame;
        badZero[hdrPos + 1] = 0;
        EXPECT_ANY_THROW((void)dctx_.decompressOne(badZero));

        std::string badPop  = frame;
        uint8_t origMask    = static_cast<uint8_t>(hdrBytes[1]);
        uint8_t flippedMask = static_cast<uint8_t>(origMask ^ 0x01u);
        badPop[hdrPos + 1]  = static_cast<char>(flippedMask);
        EXPECT_ANY_THROW((void)dctx_.decompressOne(badPop));
    }
    // Mutating shift >= 32 (for u32) fails cleanly
    {
        std::string badShift = frame;
        badShift[hdrPos + 2] = static_cast<char>(32);
        EXPECT_ANY_THROW((void)dctx_.decompressOne(badShift));
    }

    // Direct malformed header / stream count / trajectory exhaustion checks on DI_ctx_split
    // 1. Empty 0-byte header
    expectRawCaseFails(RawCodecCase{ {}, { { 1, 2, 3, 4 } } });
    // 2. Passthrough mode=2 with trailing byte in header
    expectRawCaseFails(RawCodecCase{ { 0x23, 0x00 }, { { 1, 2, 3, 4 } } });
    // 3. Passthrough mode=2 with 2 streams instead of 1
    expectRawCaseFails(RawCodecCase{ { 0x23 }, { { 1, 2 }, { 3, 4 } } });
    // 4. Mode 1, nCtxLog=1 (K=2), activeMask=0x05 (bit >= K set)
    expectRawCaseFails(RawCodecCase{ { 0x11, 0x05 }, { { 1, 2 }, { 3, 4 } } });
    // 5. Mode 0 truncated header (missing minVal varint)
    expectRawCaseFails(RawCodecCase{ { 0x01, 0x03, 0x04 }, { { 1 }, { 2 } } });
    // 6. Mode 0 unterminated varint
    expectRawCaseFails(RawCodecCase{
            { 0x01, 0x03, 0x04, 0x80, 0x80, 0x80 }, { { 1 }, { 2 } } });
    // 7. Mode 1 trajectory exhaustion: prev=0 starts in class 0, but stream 0 is
    //    NOT in activeMask (activeMask = 0x02, only class 1 present) -> rem[0]==0 at i=0!
    expectRawCaseFails(RawCodecCase{ { 0x11, 0x02 }, { { 1, 2, 3, 4 } } });
    // 8. Mode 1 mid-stream trajectory exhaustion: all values in stream 0 are 0
    //    (so prev stays 0 -> class 0 forever), but stream 0 has 2 elts & stream 1 has 2 elts!
    expectRawCaseFails(
            RawCodecCase{ { 0x11, 0x03 }, { { 0, 0 }, { 5, 5 } } });
}

} // namespace
} // namespace tests
} // namespace openzl
