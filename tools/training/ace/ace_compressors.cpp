// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "tools/training/ace/ace_compressors.h"

#include <algorithm>
#include <cstdlib>
#include <string_view>

#include "openzl/openzl.hpp"
#include "openzl/zl_reflection.h"
#include "tools/training/ace/ace_sampling.h"

#include "openzl/cpp/codecs/Bitsplit.hpp"
#include "openzl/cpp/codecs/FloatMultSplit.hpp"
#include "openzl/cpp/codecs/IntMultSplit.hpp"
#include "openzl/cpp/codecs/Partition.hpp"
#include "openzl/cpp/codecs/SparseNum.hpp"

namespace openzl {
namespace training {
namespace {
template <typename NodeT>
ACENode buildNode(const NodeT& node)
{
    return ACENode(node);
}

template <typename GraphT>
ACEGraph buildGraph(const GraphT& graph)
{
    return ACEGraph(graph);
}

std::vector<ACENode> makeAllNodes()
{
    std::vector<ACENode> n;
    n.push_back(buildNode(nodes::DeltaInt{}));
    n.push_back(buildNode(nodes::TokenizeStruct{}));
    for (auto sorted : { true, false }) {
        n.push_back(buildNode(nodes::TokenizeNumeric{ sorted }));
        n.push_back(buildNode(nodes::TokenizeString{ sorted }));
    }
    for (int numBits = 1; numBits < 64; ++numBits) {
        n.push_back(buildNode(nodes::Bitunpack{ numBits }));
    }
    n.push_back(buildNode(nodes::DivideBy{}));
    n.push_back(buildNode(nodes::RangePack{}));
    for (int level = 0; level < 10; ++level) {
        n.push_back(buildNode(nodes::FieldLz{ level }));
    }
    n.push_back(buildNode(nodes::Float32Deconstruct{}));
    n.push_back(buildNode(nodes::Float16Deconstruct{}));
    n.push_back(buildNode(nodes::BFloat16Deconstruct{}));
    n.push_back(buildNode(nodes::MergeSorted{}));
    n.push_back(buildNode(nodes::ParseInt{}));
    n.push_back(buildNode(nodes::Prefix{}));
    n.push_back(buildNode(nodes::QuantizeOffsets{}));
    n.push_back(buildNode(nodes::QuantizeLengths{}));
    n.push_back(buildNode(nodes::Partition{
            ZL_PartitionParamsPreset_quantizeOffsets }));
    n.push_back(buildNode(nodes::Partition{
            ZL_PartitionParamsPreset_quantizeLengths }));
    n.push_back(buildNode(nodes::Partition{
            ZL_PartitionParamsPreset_varbyte16 }));
    n.push_back(buildNode(nodes::SparseNum{}));
    n.push_back(buildNode(nodes::SparseNumAuto{}));
    n.push_back(buildNode(nodes::LookbackDelta{}));
    n.push_back(buildNode(
            nodes::IntMultSplit{ nodes::IntMultSplit::Mode::Unsigned }));
    n.push_back(buildNode(
            nodes::IntMultSplit{ nodes::IntMultSplit::Mode::Signed }));
    n.push_back(buildNode(nodes::FloatMultSplit{}));
    n.push_back(buildNode(
            nodes::PcoBinSplit{ nodes::PcoBinSplit::LatentMode::Unsigned }));
    n.push_back(buildNode(
            nodes::PcoBinSplit{ nodes::PcoBinSplit::LatentMode::Signed }));
    n.push_back(buildNode(
            nodes::PcoBinSplit{ nodes::PcoBinSplit::LatentMode::Float }));
    n.push_back(buildNode(nodes::TransposeSplit{}));
    n.push_back(buildNode(nodes::BitsplitFP{}));
    n.push_back(buildNode(nodes::BitsplitBF16{}));
    n.push_back(buildNode(nodes::BitsplitTop8{}));
    n.push_back(buildNode(nodes::Zigzag{}));
    n.push_back(buildNode(nodes::ConvertSerialToNum8{}));
    n.push_back(buildNode(nodes::ConvertSerialToNumLE16{}));
    n.push_back(buildNode(nodes::ConvertSerialToNumLE32{}));
    n.push_back(buildNode(nodes::ConvertSerialToNumLE64{}));
    n.push_back(buildNode(nodes::ConvertSerialToNumBE16{}));
    n.push_back(buildNode(nodes::ConvertSerialToNumBE32{}));
    n.push_back(buildNode(nodes::ConvertSerialToNumBE64{}));
    for (int width = 1; width <= 32; ++width) {
        n.push_back(buildNode(nodes::ConvertSerialToStruct{ width }));
    }
    n.push_back(buildNode(nodes::ConvertStructToNumLE{}));
    n.push_back(buildNode(nodes::ConvertStructToNumBE{}));
    n.push_back(buildNode(nodes::SeparateStringComponents{}));
    n.push_back(buildNode(nodes::Hilbert3U16{}));
    n.push_back(buildNode(nodes::Hilbert3I32{}));
    n.push_back(buildNode(nodes::Morton3U16{}));
    n.push_back(buildNode(nodes::Morton3I32{}));
    return n;
}

// The Compression Transformer is off by default: on points it made trained
// compressors ~2.4x slower to compress without making them smaller.
// ACE_TRANSFORMER=1 brings back upstream's behaviour (Transformer as an ACE
// backend graph and prebuilt seed).
bool transformerAllowed()
{
    return std::getenv("ACE_TRANSFORMER") != nullptr;
}

std::vector<ACEGraph> makeAllGraphs()
{
    std::vector<ACEGraph> g;
    g.push_back(buildGraph(graphs::Compress{}));
    if (transformerAllowed()) {
        g.push_back(buildGraph(graphs::TransformerNumeric{}));
    }
    g.push_back(buildGraph(graphs::Entropy{}));
    g.push_back(buildGraph(graphs::Fse{}));
    g.push_back(buildGraph(graphs::Huffman{}));
    g.push_back(buildGraph(graphs::HuffmanPivCo{}));
    g.push_back(buildGraph(graphs::Bitpack{}));
    g.push_back(buildGraph(graphs::Constant{}));
    for (int level = 0; level < 10; ++level) {
        g.push_back(buildGraph(graphs::FieldLz{ level }));
    }
    for (int level = -5; level < 10; ++level) {
        g.push_back(buildGraph(graphs::Zstd{ level }));
    }
    // LZ4 HC above level 6 is slow on repetitive numeric data
    // (level 9 compresses at about 6 MB/s). Each ACE generation waits for
    // its slowest candidate.
    for (int level : { 1, 3, 6 }) {
        g.push_back(buildGraph(graphs::Lz4{ level }));
    }
    for (int level : { -3, 1, 3, 6, 9 }) {
        g.push_back(buildGraph(graphs::Lz{ level }));
    }
    g.push_back(buildGraph(graphs::Flatpack{}));
    g.push_back(buildGraph(graphs::Store{}));
    return g;
}

std::vector<ACECompressor> makePrebuiltNumericCompressors()
{
    std::vector<ACECompressor> compressors;
    auto graphs = getGraphsComptabileWith(Type::Numeric);
    compressors.reserve(graphs.size());
    for (const auto& graph : graphs) {
        compressors.emplace_back(graph);
    }

    ACECompressor transformer(buildGraph(graphs::TransformerNumeric{}));
    ACECompressor fieldLz(buildGraph(graphs::FieldLz{}));
    ACECompressor zstd(buildGraph(graphs::Zstd{}));
    ACECompressor transpose(buildNode(nodes::TransposeSplit{}), { zstd });
    ACECompressor lz4(buildGraph(graphs::Lz4{}));
    ACECompressor lz(buildGraph(graphs::Lz{}));
    ACECompressor deltaFieldLz(buildNode(nodes::DeltaInt{}), { fieldLz });
    ACECompressor deltaTranspose(buildNode(nodes::DeltaInt{}), { transpose });
    ACECompressor tokenizeSortedFieldLz(
            buildNode(nodes::TokenizeNumeric{ true }),
            { deltaFieldLz, fieldLz });
    ACECompressor tokenizeFieldLz(
            buildNode(nodes::TokenizeNumeric{ false }),
            { deltaFieldLz, fieldLz });
    ACECompressor zigzagFieldLz(buildNode(nodes::Zigzag{}), { fieldLz });
    ACECompressor deltaZigzagFieldLz(
            buildNode(nodes::DeltaInt{}), { zigzagFieldLz });
    ACECompressor rangePackFieldLz(buildNode(nodes::RangePack{}), { fieldLz });
    ACECompressor rangePackDeltaFieldLz(
            buildNode(nodes::RangePack{}), { deltaFieldLz });
    ACECompressor entropy(buildGraph(graphs::Entropy{}));
    ACECompressor top8bitsEntropy(
            buildNode(nodes::BitsplitTop8{}), { entropy });
    ACECompressor fpBitsEntropy(buildNode(nodes::BitsplitFP{}), { entropy });
    ACECompressor bf16BitsEntropy(
            buildNode(nodes::BitsplitBF16{}), { entropy });
    ACECompressor fse(buildGraph(graphs::Fse{}));
    ACECompressor store(buildGraph(graphs::Store{}));
    ACECompressor f32Decon(
            buildNode(nodes::Float32Deconstruct{}), { store, entropy });
    ACECompressor f16Decon(
            buildNode(nodes::Float16Deconstruct{}), { store, entropy });
    ACECompressor bf16Decon(
            buildNode(nodes::BFloat16Deconstruct{}), { store, entropy });
    ACECompressor quantizeOffsets(
            buildNode(nodes::QuantizeOffsets{}), { fse, store });
    ACECompressor quantizeLengths(
            buildNode(nodes::QuantizeLengths{}), { fse, store });
    ACECompressor pcoBinUnsigned(
            buildNode(nodes::PcoBinSplit{
                    nodes::PcoBinSplit::LatentMode::Unsigned }),
            { entropy, store });
    ACECompressor pcoBinSigned(
            buildNode(nodes::PcoBinSplit{
                    nodes::PcoBinSplit::LatentMode::Signed }),
            { entropy, store });
    ACECompressor pcoBinFloat(
            buildNode(nodes::PcoBinSplit{
                    nodes::PcoBinSplit::LatentMode::Float }),
            { entropy, store });
    ACECompressor pcoBinUnsignedFse(
            buildNode(nodes::PcoBinSplit{
                    nodes::PcoBinSplit::LatentMode::Unsigned }),
            { fse, store });
    ACECompressor pcoBinSignedFse(
            buildNode(nodes::PcoBinSplit{
                    nodes::PcoBinSplit::LatentMode::Signed }),
            { fse, store });
    ACECompressor deltaPcoBinSigned(
            buildNode(nodes::DeltaInt{}), { pcoBinSignedFse });
    ACECompressor delta2PcoBinSigned(
            buildNode(nodes::DeltaInt{}), { deltaPcoBinSigned });
    ACECompressor rangePackPcoBinUnsigned(
            buildNode(nodes::RangePack{}), { pcoBinUnsignedFse });
    ACECompressor tokenizeSortedPcoBinUnsigned(
            buildNode(nodes::TokenizeNumeric{ true }),
            { pcoBinUnsignedFse, entropy });
    ACECompressor lookbackPcoBinSigned(
            buildNode(nodes::LookbackDelta{}), { pcoBinSignedFse, entropy });
    ACECompressor lookbackPcoBin2(
            buildNode(nodes::LookbackDelta{}),
            { pcoBinSignedFse, pcoBinUnsignedFse });
    ACECompressor lookbackPcoBinFieldLz(
            buildNode(nodes::LookbackDelta{}), { pcoBinSignedFse, fieldLz });
    ACECompressor rangePackLookbackPcoBin(
            buildNode(nodes::RangePack{}), { lookbackPcoBin2 });
    ACECompressor divideByRangePackLookbackPcoBin(
            buildNode(nodes::DivideBy{}), { rangePackLookbackPcoBin });
    ACECompressor compressGeneric(buildGraph(graphs::Compress{}));
    ACECompressor intMultDeltaPcoBin(
            buildNode(nodes::IntMultSplit{
                    nodes::IntMultSplit::Mode::Unsigned }),
            { deltaPcoBinSigned, compressGeneric });
    ACECompressor intMultSignedDeltaPcoBin(
            buildNode(nodes::IntMultSplit{
                    nodes::IntMultSplit::Mode::Signed }),
            { deltaPcoBinSigned, compressGeneric });
    ACECompressor intMultLookbackPcoBin(
            buildNode(nodes::IntMultSplit{
                    nodes::IntMultSplit::Mode::Unsigned }),
            { lookbackPcoBin2, compressGeneric });
    ACECompressor floatMultLookbackPcoBin(
            buildNode(nodes::FloatMultSplit{}),
            { lookbackPcoBin2, pcoBinSignedFse });
    ACECompressor floatMultDeltaPcoBin(
            buildNode(nodes::FloatMultSplit{}),
            { deltaPcoBinSigned, pcoBinSignedFse });
    ACECompressor floatMultPcoBin(
            buildNode(nodes::FloatMultSplit{}),
            { pcoBinSignedFse, pcoBinSignedFse });
    ACECompressor intMultSignedLookbackPcoBin(
            buildNode(nodes::IntMultSplit{
                    nodes::IntMultSplit::Mode::Signed }),
            { lookbackPcoBin2, compressGeneric });
    std::vector<ACECompressor> prebuilt = {
        fieldLz,
        zstd,
        transpose,
        lz4,
        lz,
        deltaFieldLz,
        deltaTranspose,
        tokenizeSortedFieldLz,
        tokenizeFieldLz,
        zigzagFieldLz,
        deltaZigzagFieldLz,
        rangePackFieldLz,
        rangePackDeltaFieldLz,
        top8bitsEntropy,
        fpBitsEntropy,
        bf16BitsEntropy,
        f32Decon,
        f16Decon,
        bf16Decon,
        quantizeOffsets,
        quantizeLengths,
        pcoBinUnsigned,
        pcoBinSigned,
        pcoBinFloat,
        pcoBinUnsignedFse,
        pcoBinSignedFse,
        deltaPcoBinSigned,
        delta2PcoBinSigned,
        rangePackPcoBinUnsigned,
        tokenizeSortedPcoBinUnsigned,
        lookbackPcoBinSigned,
        lookbackPcoBin2,
        lookbackPcoBinFieldLz,
        rangePackLookbackPcoBin,
        divideByRangePackLookbackPcoBin,
        intMultDeltaPcoBin,
        intMultSignedDeltaPcoBin,
        intMultLookbackPcoBin,
        floatMultLookbackPcoBin,
        floatMultDeltaPcoBin,
        floatMultPcoBin,
        intMultSignedLookbackPcoBin,
    };
    if (transformerAllowed()) {
        prebuilt.insert(prebuilt.begin(), transformer);
    }
    compressors.insert(compressors.end(), prebuilt.begin(), prebuilt.end());
    // ACE_TOK_SEEDS: tokenize-first chains that won the big points slots of
    // the strongest cross-corpus model, so every slot starts with them in the
    // population.
    if (std::getenv("ACE_TOK_SEEDS") != nullptr) {
        ACECompressor transposeEntropy(
                buildNode(nodes::TransposeSplit{}), { entropy });
        ACECompressor transposeFse(buildNode(nodes::TransposeSplit{}), { fse });
        ACECompressor transposeZstd(
                buildNode(nodes::TransposeSplit{}), { zstd });
        ACECompressor dRpT(
                buildNode(nodes::DeltaInt{}),
                { ACECompressor(
                        buildNode(nodes::RangePack{}), { transposeEntropy }) });
        ACECompressor dZzT(
                buildNode(nodes::DeltaInt{}),
                { ACECompressor(
                        buildNode(nodes::Zigzag{}), { transposeEntropy }) });
        ACECompressor dT(buildNode(nodes::DeltaInt{}), { transposeEntropy });
        ACECompressor dZzTFse(
                buildNode(nodes::DeltaInt{}),
                { ACECompressor(buildNode(nodes::Zigzag{}), { transposeFse }) });
        ACECompressor rpDTZstd(
                buildNode(nodes::RangePack{}),
                { ACECompressor(
                        buildNode(nodes::DeltaInt{}), { transposeZstd }) });
        ACECompressor dZzTZstd(
                buildNode(nodes::DeltaInt{}),
                { ACECompressor(
                        buildNode(nodes::Zigzag{}), { transposeZstd }) });
        ACECompressor dQo(buildNode(nodes::DeltaInt{}), { quantizeOffsets });
        ACECompressor tokDZstdInner(
                buildNode(nodes::TokenizeNumeric{ true }),
                { ACECompressor(
                          buildNode(nodes::DeltaInt{}), { transposeZstd }),
                  entropy });
        ACECompressor tokDZstdOuter(
                buildNode(nodes::TokenizeNumeric{ true }),
                { ACECompressor(
                          buildNode(nodes::DeltaInt{}), { tokDZstdInner }),
                  transposeZstd });
        compressors.push_back(tokDZstdOuter);
        compressors.emplace_back(
                buildNode(nodes::RangePack{}),
                std::vector{ ACECompressor(
                        buildNode(nodes::DeltaInt{}),
                        { ACECompressor(
                                buildNode(nodes::Zigzag{}),
                                { tokDZstdOuter }) }) });
        compressors.emplace_back(
                buildNode(nodes::TokenizeNumeric{ true }),
                std::vector{ dT, dZzTZstd });
        compressors.emplace_back(
                buildNode(nodes::DeltaInt{}),
                std::vector{ ACECompressor(
                        buildNode(nodes::Zigzag{}),
                        { ACECompressor(
                                buildNode(nodes::TokenizeNumeric{ true }),
                                { rpDTZstd, transposeZstd }) }) });
        compressors.emplace_back(
                buildNode(nodes::LookbackDelta{}),
                std::vector{ ACECompressor(
                                     buildNode(nodes::TokenizeNumeric{ true }),
                                     { deltaPcoBinSigned, pcoBinUnsignedFse }),
                             pcoBinUnsignedFse });
        for (bool sorted : { true, false }) {
            compressors.emplace_back(
                    buildNode(nodes::TokenizeNumeric{ sorted }),
                    std::vector{ dRpT, dZzT });
            compressors.emplace_back(
                    buildNode(nodes::TokenizeNumeric{ sorted }),
                    std::vector{ dZzTFse, top8bitsEntropy });
        }
        compressors.emplace_back(
                buildNode(nodes::RangePack{}),
                std::vector{ ACECompressor(
                        buildNode(nodes::TokenizeNumeric{ true }),
                        { dT, dZzT }) });
        ACECompressor tokTail(
                buildNode(nodes::TokenizeNumeric{ true }),
                { rpDTZstd, transposeEntropy });
        compressors.emplace_back(
                buildNode(nodes::DeltaInt{}),
                std::vector{ ACECompressor(
                        buildNode(nodes::Zigzag{}), { tokTail }) });
        ACECompressor tokQo(
                buildNode(nodes::TokenizeNumeric{ true }),
                { dQo, transposeFse });
        compressors.emplace_back(
                buildNode(nodes::DivideBy{}),
                std::vector{ ACECompressor(
                        buildNode(nodes::DeltaInt{}),
                        { ACECompressor(
                                buildNode(nodes::Zigzag{}), { tokQo }) }) });
        // Delta -> Zigzag -> TokenizeNumeric(sorted) -> RangePack/Delta -> Transpose -> Zstd
        // (strong on smooth floating-point/integer streams like gps_time)
        ACECompressor dZzTokRpDTZstd(
                buildNode(nodes::DeltaInt{}),
                { ACECompressor(
                        buildNode(nodes::Zigzag{}),
                        { ACECompressor(
                                buildNode(nodes::TokenizeNumeric{ true }),
                                { ACECompressor(
                                          buildNode(nodes::RangePack{}),
                                          { ACECompressor(
                                                  buildNode(nodes::DeltaInt{}),
                                                  { transposeZstd }) }),
                                  transposeZstd }) }) });
        compressors.push_back(dZzTokRpDTZstd);
        compressors.emplace_back(
                buildNode(nodes::IntMultSplit{
                        nodes::IntMultSplit::Mode::Unsigned }),
                std::vector{ dZzTokRpDTZstd, compressGeneric });
        compressors.emplace_back(
                buildNode(nodes::IntMultSplit{
                        nodes::IntMultSplit::Mode::Unsigned }),
                std::vector{ tokenizeSortedPcoBinUnsigned, compressGeneric });
    }
    return compressors;
}

std::vector<ACECompressor> makePrebuiltStructCompressors()
{
    std::vector<ACECompressor> compressors;
    auto graphs = getGraphsComptabileWith(Type::Struct);
    compressors.reserve(graphs.size());

    ACECompressor fieldLz(buildGraph(graphs::FieldLz{}));
    ACECompressor zstd(buildGraph(graphs::Zstd{}));
    ACECompressor transpose(buildNode(nodes::TransposeSplit{}), { zstd });
    ACECompressor tokenizeFieldLz(
            buildNode(nodes::TokenizeStruct{}), { transpose, fieldLz });
    ACECompressor lz4(buildGraph(graphs::Lz4{}));
    ACECompressor lz(buildGraph(graphs::Lz{}));

    compressors.push_back(zstd);
    compressors.push_back(transpose);
    compressors.push_back(tokenizeFieldLz);
    compressors.push_back(lz4);
    compressors.push_back(lz);

    const bool bigEndian = std::getenv("ACE_BIG_ENDIAN") != nullptr;
    for (const auto& compressor : makePrebuiltNumericCompressors()) {
        compressors.push_back(ACECompressor(
                buildNode(nodes::ConvertStructToNumLE{}), { compressor }));
        if (bigEndian) {
            compressors.push_back(ACECompressor(
                    buildNode(nodes::ConvertStructToNumBE{}), { compressor }));
        }
    }
    return compressors;
}

std::vector<ACECompressor> makePrebuiltSerialCompressors()
{
    std::vector<ACECompressor> compressors;
    auto graphs = getGraphsComptabileWith(Type::Serial);
    compressors.reserve(graphs.size());
    for (const auto& graph : graphs) {
        compressors.emplace_back(graph);
    }

    ACECompressor fieldLz(buildGraph(graphs::FieldLz{}));
    ACECompressor zstd(buildGraph(graphs::Zstd{}));
    ACECompressor lz4(buildGraph(graphs::Lz4{}));
    ACECompressor entropy(buildGraph(graphs::Entropy{}));
    ACECompressor fse(buildGraph(graphs::Fse{}));
    ACECompressor store(buildGraph(graphs::Store{}));
    ACECompressor deltaFieldLz(buildNode(nodes::DeltaInt{}), { fieldLz });
    ACECompressor transpose(buildNode(nodes::TransposeSplit{}), { zstd });
    ACECompressor deltaTranspose(buildNode(nodes::DeltaInt{}), { transpose });
    ACECompressor tokenizeSortedFieldLz(
            buildNode(nodes::TokenizeNumeric{ true }),
            { deltaFieldLz, fieldLz });
    ACECompressor zigzagFieldLz(buildNode(nodes::Zigzag{}), { fieldLz });
    ACECompressor deltaZigzagFieldLz(
            buildNode(nodes::DeltaInt{}), { zigzagFieldLz });
    ACECompressor rangePackDeltaFieldLz(
            buildNode(nodes::RangePack{}), { deltaFieldLz });
    ACECompressor pcoBinUnsignedFse(
            buildNode(nodes::PcoBinSplit{
                    nodes::PcoBinSplit::LatentMode::Unsigned }),
            { fse, store });
    ACECompressor pcoBinSignedFse(
            buildNode(nodes::PcoBinSplit{
                    nodes::PcoBinSplit::LatentMode::Signed }),
            { fse, store });
    ACECompressor lookbackPcoBin2(
            buildNode(nodes::LookbackDelta{}),
            { pcoBinSignedFse, pcoBinUnsignedFse });
    ACECompressor deltaPcoBinSigned(
            buildNode(nodes::DeltaInt{}), { pcoBinSignedFse });
    ACECompressor intMultDeltaPcoBin(
            buildNode(nodes::IntMultSplit{
                    nodes::IntMultSplit::Mode::Unsigned }),
            { deltaPcoBinSigned, store });
    ACECompressor floatMultLookbackPcoBin(
            buildNode(nodes::FloatMultSplit{}),
            { lookbackPcoBin2, pcoBinSignedFse });
    std::vector<ACECompressor> coreNumeric = {
        zstd,
        fieldLz,
        lz4,
        entropy,
        deltaFieldLz,
        deltaTranspose,
        tokenizeSortedFieldLz,
        zigzagFieldLz,
        deltaZigzagFieldLz,
        rangePackDeltaFieldLz,
        pcoBinUnsignedFse,
        pcoBinSignedFse,
        lookbackPcoBin2,
        intMultDeltaPcoBin,
        floatMultLookbackPcoBin,
    };
    const bool bigEndian = std::getenv("ACE_BIG_ENDIAN") != nullptr;
    for (const auto& compressor : coreNumeric) {
        compressors.push_back(ACECompressor(
                buildNode(nodes::ConvertSerialToNum8{}), { compressor }));
        compressors.push_back(ACECompressor(
                buildNode(nodes::ConvertSerialToNumLE16{}), { compressor }));
        compressors.push_back(ACECompressor(
                buildNode(nodes::ConvertSerialToNumLE32{}), { compressor }));
        compressors.push_back(ACECompressor(
                buildNode(nodes::ConvertSerialToNumLE64{}), { compressor }));
        if (bigEndian) {
            compressors.push_back(ACECompressor(
                    buildNode(nodes::ConvertSerialToNumBE16{}), { compressor }));
            compressors.push_back(ACECompressor(
                    buildNode(nodes::ConvertSerialToNumBE32{}), { compressor }));
            compressors.push_back(ACECompressor(
                    buildNode(nodes::ConvertSerialToNumBE64{}), { compressor }));
        }
    }

    return compressors;
}

std::vector<ACECompressor> makePrebuiltStringCompressors()
{
    std::vector<ACECompressor> compressors;
    auto graphs = getGraphsComptabileWith(Type::String);
    compressors.reserve(graphs.size());
    for (const auto& graph : graphs) {
        compressors.emplace_back(graph);
    }

    ACECompressor fieldLz(buildGraph(graphs::FieldLz{}));
    ACECompressor zstd(buildGraph(graphs::Zstd{}));
    ACECompressor separate(
            buildNode(nodes::SeparateStringComponents{}), { zstd, fieldLz });
    ACECompressor prefix(buildNode(nodes::Prefix{}), { separate, fieldLz });
    ACECompressor tokenizeSorted(
            buildNode(nodes::TokenizeString{ true }), { prefix, fieldLz });
    ACECompressor tokenize(
            buildNode(nodes::TokenizeString{ false }), { separate, fieldLz });

    compressors.push_back(separate);
    compressors.push_back(prefix);
    compressors.push_back(tokenizeSorted);
    compressors.push_back(tokenize);

    return compressors;
}

std::vector<ACECompressor> makePrebuiltCompressors(Type inputType)
{
    switch (inputType) {
        case Type::Serial:
            return makePrebuiltSerialCompressors();
        case Type::Struct:
            return makePrebuiltStructCompressors();
        case Type::Numeric:
            return makePrebuiltNumericCompressors();
        case Type::String:
            return makePrebuiltStringCompressors();
        default:
            assert(false);
            return {};
    }
}
} // namespace

poly::span<const ACENode> getAllNodes()
{
    static auto nodes = new std::vector<ACENode>(makeAllNodes());
    return *nodes;
}

poly::span<const ACEGraph> getAllGraphs()
{
    static auto graphs = new std::vector<ACEGraph>(makeAllGraphs());
    return *graphs;
}

// Returns the type-compatible nodes further restricted to those usable at
// @p formatVersion (nodes with minFormatVersion 0 are treated as
// unconstrained).
std::vector<ACENode> getNodesComptabileWith(
        Type inputType,
        uint32_t formatVersion)
{
    static auto nodes = new std::unordered_map<Type, std::vector<ACENode>>([] {
        std::unordered_map<Type, std::vector<ACENode>> m;
        for (auto t :
             { Type::Serial, Type::Struct, Type::Numeric, Type::String }) {
            for (const auto& n : getAllNodes()) {
                if (isCompatible(n.inputType, t)) {
                    m[t].push_back(n);
                }
            }
        }
        return m;
    }());
    std::vector<ACENode> compatible;
    for (const auto& n : nodes->at(inputType)) {
        if (n.minFormatVersion == 0 || n.minFormatVersion <= formatVersion) {
            compatible.push_back(n);
        }
    }
    return compatible;
}

poly::span<const ACEGraph> getGraphsComptabileWith(Type inputType)
{
    static auto graphs =
            new std::unordered_map<Type, std::vector<ACEGraph>>([] {
                std::unordered_map<Type, std::vector<ACEGraph>> m;
                for (auto t : { Type::Serial,
                                Type::Struct,
                                Type::Numeric,
                                Type::String }) {
                    for (const auto& g : getAllGraphs()) {
                        if (isCompatible(g.inputTypeMask, t)) {
                            m[t].push_back(g);
                        }
                    }
                }
                return m;
            }());
    return graphs->at(inputType);
}

poly::span<const ACECompressor> getPrebuiltCompressors(Type inputType)
{
    static auto compressors =
            new std::unordered_map<Type, std::vector<ACECompressor>>([] {
                std::unordered_map<Type, std::vector<ACECompressor>> m;
                for (auto t : { Type::Serial,
                                Type::Struct,
                                Type::Numeric,
                                Type::String }) {
                    m[t] = makePrebuiltCompressors(t);
                }
                return m;
            }());
    return compressors->at(inputType);
}

const ACENode& pickRandomNodeForType(
        std::mt19937_64& rng,
        Type inputType,
        poly::span<const ACENode> nodes)
{
    assert(!nodes.empty());
    if (inputType == Type::Numeric) {
        std::vector<const ACENode*> native;
        std::vector<const ACENode*> converted;
        for (const auto& n : nodes) {
            if (n.inputType == Type::Numeric) {
                native.push_back(&n);
            } else {
                converted.push_back(&n);
            }
        }
        if (!native.empty() && !converted.empty()) {
            std::bernoulli_distribution useNative(0.85);
            const auto& bucket = useNative(rng) ? native : converted;
            std::vector<ACENode> tmp;
            tmp.reserve(bucket.size());
            for (const auto* ptr : bucket) {
                tmp.push_back(*ptr);
            }
            const ACENode& chosen = pickRandomNode(
                    rng, poly::span<const ACENode>(tmp.data(), tmp.size()));
            const size_t idx = static_cast<size_t>(&chosen - tmp.data());
            return *bucket[idx];
        }
    }
    return pickRandomNode(rng, nodes);
}

const ACENode& pickRandomNode(
        std::mt19937_64& rng,
        poly::span<const ACENode> nodes)
{
    assert(!nodes.empty());
    static const bool byFamily = std::getenv("ACE_FAMILY") != nullptr;
    if (!byFamily) {
        std::uniform_int_distribution<size_t> dist(0, nodes.size() - 1);
        return nodes[dist(rng)];
    }
    std::vector<std::string_view> names;
    for (const auto& n : nodes) {
        if (std::find(names.begin(), names.end(), n.name) == names.end()) {
            names.push_back(n.name);
        }
    }
    std::uniform_int_distribution<size_t> nameDist(0, names.size() - 1);
    const auto name = names[nameDist(rng)];
    std::vector<const ACENode*> variants;
    for (const auto& n : nodes) {
        if (n.name == name) {
            variants.push_back(&n);
        }
    }
    std::uniform_int_distribution<size_t> varDist(0, variants.size() - 1);
    return *variants[varDist(rng)];
}

bool nodeAcceptsWidth(const ACENode& node, uint32_t inputWidth)
{
    if (inputWidth == 0) {
        return true;
    }
    const auto& nm = node.name;
    if (nm == "zl.hilbert3_u16" || nm == "zl.morton3_u16") {
        return inputWidth == 6;
    }
    if (nm == "zl.hilbert3_i32" || nm == "zl.morton3_i32") {
        return inputWidth == 12;
    }
    if (nm == "zl.float32_deconstruct" || nm == "zl.bitsplit_fp") {
        return inputWidth == 4;
    }
    if (nm == "zl.float16_deconstruct" || nm == "zl.bfloat16_deconstruct"
        || nm == "zl.bitsplit_bf16") {
        return inputWidth == 2;
    }
    if (nm == "zl.convert_struct_to_num_le"
        || nm == "zl.convert_struct_to_num_be") {
        return inputWidth == 1 || inputWidth == 2 || inputWidth == 4
                || inputWidth == 8;
    }
    if (nm == "zl.bitsplit_top8") {
        return inputWidth >= 2;
    }
    return true;
}

uint32_t inferNodeOutputWidth(
        const ACENode& node,
        size_t port,
        uint32_t inputWidth)
{
    const auto& nm = node.name;
    if (nm == "zl.hilbert3_u16" || nm == "zl.hilbert3_i32"
        || nm == "zl.morton3_u16" || nm == "zl.morton3_i32") {
        return 8;
    }
    if (nm == "zl.convert_serial_to_num8") {
        return 1;
    }
    if (nm == "zl.convert_serial_to_num_le16"
        || nm == "zl.convert_serial_to_num_be16") {
        return 2;
    }
    if (nm == "zl.convert_serial_to_num_le32"
        || nm == "zl.convert_serial_to_num_be32") {
        return 4;
    }
    if (nm == "zl.convert_serial_to_num_le64"
        || nm == "zl.convert_serial_to_num_be64") {
        return 8;
    }
    if (nm == "zl.convert_serial_to_struct" && node.params.has_value()
        && node.params->localParams.has_value()) {
        for (const auto& p : node.params->localParams->getIntParams()) {
            if (p.paramId == ZL_trlip_tokenSize && p.paramValue > 0) {
                return static_cast<uint32_t>(p.paramValue);
            }
        }
    }
    if (nm == "zl.bitunpack" && node.params.has_value()
        && node.params->localParams.has_value()) {
        for (const auto& p : node.params->localParams->getIntParams()) {
            if (p.paramId == ZL_Bitunpack_numBits && p.paramValue > 0) {
                if (p.paramValue <= 8) return 1;
                if (p.paramValue <= 16) return 2;
                if (p.paramValue <= 32) return 4;
                return 8;
            }
        }
    }
    if (nm == "zl.convert_struct_to_num_le"
        || nm == "zl.convert_struct_to_num_be"
        || nm == "zl.convert_num_to_struct_le"
        || nm == "zl.convert_num_to_struct_be") {
        return inputWidth;
    }
    if (nm == "zl.float32_deconstruct") {
        return port == 0 ? 3 : 0;
    }
    if (nm == "zl.float16_deconstruct" || nm == "zl.bfloat16_deconstruct") {
        return port == 0 ? 2 : 0;
    }
    if (nm == "zl.pco_bin_split" || nm == "zl.partition") {
        return port == 0 ? 1 : 0;
    }
    if (nm == "zl.lookback_delta") {
        return port == 0 ? inputWidth : 2;
    }
    if (nm == "zl.sparse_num" || nm == "zl.sparse_num_auto") {
        return port == 0 ? 4 : inputWidth;
    }
    if (nm == "zl.separate_string_components") {
        return port == 1 ? 4 : 0;
    }
    if (nm == "zl.delta_int" || nm == "zl.zigzag" || nm == "zl.range_pack"
        || nm == "zl.divide_by" || nm == "zl.int_mult_split"
        || nm == "zl.float_mult_split" || nm == "zl.merge_sorted") {
        return inputWidth;
    }
    if (nm == "zl.tokenize_numeric") {
        return port == 1 ? inputWidth : 0;
    }
    return 0;
}

poly::span<const ACEGraph> getPolishBackends()
{
    static auto graphs = new std::vector<ACEGraph>{
        buildGraph(graphs::Entropy{}),
        buildGraph(graphs::Huffman{}),
        buildGraph(graphs::HuffmanPivCo{}),
        buildGraph(graphs::Fse{}),
        buildGraph(graphs::FieldLz{}),
        buildGraph(graphs::Zstd{}),
        buildGraph(graphs::Bitpack{}),
        buildGraph(graphs::Constant{}),
        buildGraph(graphs::Store{}),
        buildGraph(graphs::Flatpack{}),
    };
    return *graphs;
}

ACECompressor buildRandomGraphCompressor(std::mt19937_64& rng, Type inputType)
{
    const auto graphs = getGraphsComptabileWith(inputType);
    std::vector<std::string_view> families;
    families.reserve(graphs.size());
    for (const auto& g : graphs) {
        if (std::find(families.begin(), families.end(), g.name)
            == families.end()) {
            families.push_back(g.name);
        }
    }
    std::uniform_int_distribution<size_t> famDist(0, families.size() - 1);
    const auto fam = families[famDist(rng)];
    std::vector<const ACEGraph*> variants;
    for (const auto& g : graphs) {
        if (g.name == fam) {
            variants.push_back(&g);
        }
    }
    std::uniform_int_distribution<size_t> varDist(0, variants.size() - 1);
    return ACEGraphCompressor(*variants[varDist(rng)]);
}

ACECompressor buildRandomNodeCompressor(
        std::mt19937_64& rng,
        Type inputType,
        uint32_t formatVersion,
        size_t maxDepth,
        uint32_t inputWidth)
{
    if (maxDepth == 0) {
        return buildRandomGraphCompressor(rng, inputType);
    }
    const auto allCompatible = getNodesComptabileWith(inputType, formatVersion);
    std::vector<ACENode> compatible;
    compatible.reserve(allCompatible.size());
    for (const auto& cand : allCompatible) {
        if (nodeAcceptsWidth(cand, inputWidth)) {
            compatible.push_back(cand);
        }
    }
    if (compatible.empty()) {
        // No node is available at the target format version/width; fall back to
        // a single graph.
        return buildRandomGraphCompressor(rng, inputType);
    }
    auto node = pickRandomNodeForType(
            rng,
            inputType,
            poly::span<const ACENode>(compatible.data(), compatible.size()));
    assert(isCompatible(node.inputType, inputType));
    std::vector<std::unique_ptr<ACECompressor>> successors;
    successors.reserve(node.outputTypes.size());
    for (size_t i = 0; i < node.outputTypes.size(); ++i) {
        if (node.name == "zl.pco_bin_split" && i == 1) {
            std::bernoulli_distribution useStore(0.85);
            successors.push_back(std::make_unique<ACECompressor>(
                    useStore(rng) ? ACEGraphCompressor(buildGraph(graphs::Store{}))
                                  : ACEGraphCompressor(buildGraph(graphs::Entropy{}))));
        } else if (node.name == "zl.int_mult_split" && i == 1) {
            static const std::array<ACEGraph, 5> adjGraphs = {
                buildGraph(graphs::Store{}),
                buildGraph(graphs::Compress{}),
                buildGraph(graphs::Entropy{}),
                buildGraph(graphs::Bitpack{}),
                buildGraph(graphs::Constant{}),
            };
            std::uniform_int_distribution<size_t> adjDist(
                    0, adjGraphs.size() - 1);
            successors.push_back(std::make_unique<ACECompressor>(
                    ACEGraphCompressor(adjGraphs[adjDist(rng)])));
        } else if (node.name == "zl.float_mult_split" && i == 1) {
            std::uniform_int_distribution<int> adjChoice(0, 2);
            const int choice = adjChoice(rng);
            if (choice == 0) {
                successors.push_back(std::make_unique<ACECompressor>(
                        buildNode(nodes::PcoBinSplit{
                                nodes::PcoBinSplit::LatentMode::Signed }),
                        std::vector<ACECompressor>{
                                ACEGraphCompressor(buildGraph(graphs::Fse{})),
                                ACEGraphCompressor(buildGraph(graphs::Store{})) }));
            } else if (choice == 1) {
                successors.push_back(std::make_unique<ACECompressor>(
                        ACEGraphCompressor(buildGraph(graphs::Compress{}))));
            } else {
                successors.push_back(std::make_unique<ACECompressor>(
                        ACEGraphCompressor(buildGraph(graphs::Store{}))));
            }
        } else {
            const uint32_t outWidth =
                    inferNodeOutputWidth(node, i, inputWidth);
            successors.push_back(
                    std::make_unique<ACECompressor>(buildRandomCompressor(
                            rng,
                            node.outputTypes[i],
                            formatVersion,
                            maxDepth - 1,
                            outWidth)));
        }
    }
    if ((node.name == "zl.zigzag" || node.name == "zl.range_pack"
         || node.name == "zl.int_mult_split"
         || node.name == "zl.float_mult_split"
         || node.name == "zl.lookback_delta"
         || node.name == "zl.pco_bin_split")
        && successors[0]->isNode()
        && successors[0]->asNode().node.name == node.name) {
        return *successors[0];
    }
    return ACENodeCompressor(std::move(node), std::move(successors));
}

ACECompressor buildRandomCompressor(
        std::mt19937_64& rng,
        Type inputType,
        uint32_t formatVersion,
        size_t maxDepth,
        uint32_t inputWidth)
{
    if (maxDepth == 0) {
        return buildRandomGraphCompressor(rng, inputType);
    }
    std::bernoulli_distribution dist(0.5);
    if (dist(rng)) {
        auto compressor = buildRandomGraphCompressor(rng, inputType);
        assert(compressor.acceptsInputType(inputType));
        return compressor;
    } else {
        auto compressor = buildRandomNodeCompressor(
                rng, inputType, formatVersion, maxDepth, inputWidth);
        assert(compressor.acceptsInputType(inputType));
        return compressor;
    }
}

ACECompressor buildStoreCompressor()
{
    return ACEGraphCompressor(buildGraph(graphs::Store{}));
}

ACECompressor buildCompressGenericCompressor()
{
    return ACEGraphCompressor(buildGraph(graphs::Compress{}));
}

} // namespace training
} // namespace openzl
