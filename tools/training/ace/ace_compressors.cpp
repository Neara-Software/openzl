// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "tools/training/ace/ace_compressors.h"

#include <algorithm>
#include <cstdlib>
#include <string_view>

#include "openzl/openzl.hpp"
#include "openzl/zl_reflection.h"
#include "tools/training/ace/ace_sampling.h"

#include "openzl/cpp/codecs/Bitsplit.hpp"

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

// ACE_NO_TRANSFORMER: leave the Compression Transformer out of the search.
bool transformerAllowed()
{
    return std::getenv("ACE_NO_TRANSFORMER") == nullptr;
}

std::vector<ACEGraph> makeAllGraphs()
{
    std::vector<ACEGraph> g;
    g.push_back(buildGraph(graphs::Compress{}));
    if (transformerAllowed()) {
        g.push_back(buildGraph(graphs::TransformerNumeric{}));
    }
    g.push_back(buildGraph(graphs::Entropy{}));
    g.push_back(buildGraph(graphs::Bitpack{}));
    g.push_back(buildGraph(graphs::Constant{}));
    for (int level = 0; level < 10; ++level) {
        g.push_back(buildGraph(graphs::FieldLz{ level }));
    }
    for (int level = -5; level < 10; ++level) {
        g.push_back(buildGraph(graphs::Zstd{ level }));
    }
    for (int level : { 1, 3, 6, 9, 12 }) {
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
            buildNode(nodes::QuantizeOffsets{}), { fse, store });
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
        ACECompressor dQo(buildNode(nodes::DeltaInt{}), { quantizeOffsets });
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

    for (const auto& compressor : makePrebuiltNumericCompressors()) {
        compressors.push_back(ACECompressor(
                buildNode(nodes::ConvertStructToNumLE{}), { compressor }));
        compressors.push_back(ACECompressor(
                buildNode(nodes::ConvertStructToNumBE{}), { compressor }));
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

    for (const auto& compressor : makePrebuiltNumericCompressors()) {
        compressors.push_back(ACECompressor(
                buildNode(nodes::ConvertSerialToNum8{}), { compressor }));
        compressors.push_back(ACECompressor(
                buildNode(nodes::ConvertSerialToNumLE16{}), { compressor }));
        compressors.push_back(ACECompressor(
                buildNode(nodes::ConvertSerialToNumBE16{}), { compressor }));
        compressors.push_back(ACECompressor(
                buildNode(nodes::ConvertSerialToNumLE32{}), { compressor }));
        compressors.push_back(ACECompressor(
                buildNode(nodes::ConvertSerialToNumBE32{}), { compressor }));
        compressors.push_back(ACECompressor(
                buildNode(nodes::ConvertSerialToNumLE64{}), { compressor }));
        compressors.push_back(ACECompressor(
                buildNode(nodes::ConvertSerialToNumBE64{}), { compressor }));
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

poly::span<const ACEGraph> getPolishBackends()
{
    static auto graphs = new std::vector<ACEGraph>{
        buildGraph(graphs::Entropy{}),  buildGraph(graphs::Huffman{}),
        buildGraph(graphs::Fse{}),      buildGraph(graphs::FieldLz{}),
        buildGraph(graphs::Zstd{}),     buildGraph(graphs::Bitpack{}),
        buildGraph(graphs::Constant{}), buildGraph(graphs::Store{}),
        buildGraph(graphs::Flatpack{}),
    };
    return *graphs;
}

ACECompressor buildRandomGraphCompressor(std::mt19937_64& rng, Type inputType)
{
    return ACEGraphCompressor(
            randomChoice(rng, getGraphsComptabileWith(inputType)));
}

ACECompressor buildRandomNodeCompressor(
        std::mt19937_64& rng,
        Type inputType,
        uint32_t formatVersion,
        size_t maxDepth)
{
    if (maxDepth == 0) {
        return buildRandomGraphCompressor(rng, inputType);
    }
    const auto compatible = getNodesComptabileWith(inputType, formatVersion);
    if (compatible.empty()) {
        // No node is available at the target format version; fall back to a
        // single graph.
        return buildRandomGraphCompressor(rng, inputType);
    }
    auto node = pickRandomNode(
            rng,
            poly::span<const ACENode>(compatible.data(), compatible.size()));
    assert(isCompatible(node.inputType, inputType));
    std::vector<std::unique_ptr<ACECompressor>> successors;
    successors.reserve(node.outputTypes.size());
    for (size_t i = 0; i < node.outputTypes.size(); ++i) {
        successors.push_back(
                std::make_unique<ACECompressor>(buildRandomCompressor(
                        rng,
                        node.outputTypes[i],
                        formatVersion,
                        maxDepth - 1)));
    }
    return ACENodeCompressor(std::move(node), std::move(successors));
}

ACECompressor buildRandomCompressor(
        std::mt19937_64& rng,
        Type inputType,
        uint32_t formatVersion,
        size_t maxDepth)
{
    std::bernoulli_distribution dist(0.5);
    if (dist(rng)) {
        auto compressor = buildRandomGraphCompressor(rng, inputType);
        assert(compressor.acceptsInputType(inputType));
        return compressor;
    } else {
        auto compressor = buildRandomNodeCompressor(
                rng, inputType, formatVersion, maxDepth - 1);
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
