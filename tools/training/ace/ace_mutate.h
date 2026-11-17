// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <cstdlib>
#include <vector>

#include "tools/training/ace/ace_compressor.h"
#include "tools/training/ace/ace_compressors.h"
#include "tools/training/ace/ace_sampling.h"

namespace openzl {
namespace training {

/// Mutates an ACECompressor to produce a new ACECompressor with a single
/// mutation.
class ACEMutate {
   public:
    ACEMutate(
            std::mt19937_64& rng,
            Type inputType,
            uint32_t formatVersion,
            size_t maxDepth = kDefaultMaxDepth)
            : rng_(rng),
              inputType_(inputType),
              formatVersion_(formatVersion),
              maxDepth_(maxDepth)
    {
    }

    ACECompressor operator()(const ACECompressor& parent)
    {
        ACEReservoirSampler<const ACECompressor> sampler(rng_);
        visitMutable(parent, inputType_, [&](const auto& component) {
            sampler.update(component);
        });

        return replaceMutable(
                parent,
                inputType_,
                0,
                0,
                [&](const auto& component,
                    auto inputType,
                    uint32_t inputWidth,
                    size_t depth) -> poly::optional<ACECompressor> {
                    if (&component == sampler.get()) {
                        return replace(
                                component, inputType, inputWidth, depth);
                    } else {
                        return poly::nullopt;
                    }
                });
    }

   private:
    static bool isProtectedPort(const ACENode& node, size_t port)
    {
        return port == 1
                && (node.name == "zl.pco_bin_split"
                    || node.name == "zl.int_mult_split"
                    || node.name == "zl.float_mult_split");
    }

    static bool subtreeValidForWidthAndVersion(
            const ACECompressor& sub,
            uint32_t width,
            uint32_t fmtVersion)
    {
        if (!sub.isNode()) {
            return true;
        }
        const auto& nc = sub.asNode();
        if (nc.node.minFormatVersion != 0
            && nc.node.minFormatVersion > fmtVersion) {
            return false;
        }
        if (!nodeAcceptsWidth(nc.node, width)) {
            return false;
        }
        for (size_t i = 0; i < nc.successors.size(); ++i) {
            const uint32_t outW = inferNodeOutputWidth(nc.node, i, width);
            if (!subtreeValidForWidthAndVersion(
                        *nc.successors[i], outW, fmtVersion)) {
                return false;
            }
        }
        return true;
    }

    static bool isCtxSplitNodeName(std::string_view nm)
    {
        return nm == "zl.ctx_split" || nm == "zl.ctx_split_num";
    }

    template <typename VisitorFn>
    static void visitMutable(
            const ACECompressor& comp,
            Type inputType,
            VisitorFn&& visit)
    {
        visit(comp);
        if (comp.isNode()) {
            const auto& node = comp.asNode();
            for (size_t i = 0; i < node.successors.size(); ++i) {
                if (isProtectedPort(node.node, i)) {
                    continue;
                }
                visitMutable(
                        *node.successors[i], node.node.outputTypes[i], visit);
            }
        }
    }

    template <typename ReplaceFn>
    static ACECompressor replaceMutable(
            const ACECompressor& comp,
            Type inputType,
            uint32_t inputWidth,
            size_t depth,
            ReplaceFn&& replaceFn)
    {
        auto replacement = replaceFn(comp, inputType, inputWidth, depth);
        if (replacement.has_value()) {
            return std::move(*replacement);
        }
        if (comp.isNode()) {
            const auto& node = comp.asNode();
            std::vector<std::unique_ptr<ACECompressor>> successors;
            successors.reserve(node.successors.size());
            for (size_t i = 0; i < node.successors.size(); ++i) {
                if (isProtectedPort(node.node, i)) {
                    successors.push_back(
                            std::make_unique<ACECompressor>(*node.successors[i]));
                } else {
                    const uint32_t outWidth =
                            inferNodeOutputWidth(node.node, i, inputWidth);
                    auto s = replaceMutable(
                            *node.successors[i],
                            node.node.outputTypes[i],
                            outWidth,
                            depth + 1,
                            replaceFn);
                    while (isCtxSplitNodeName(node.node.name) && s.isNode()
                           && isCtxSplitNodeName(s.asNode().node.name)) {
                        ACECompressor unwrapped = *s.asNode().successors[0];
                        s                       = std::move(unwrapped);
                    }
                    successors.push_back(
                            std::make_unique<ACECompressor>(std::move(s)));
                }
            }
            return ACENodeCompressor(node.node, std::move(successors));
        } else {
            return comp;
        }
    }

    ACECompressor replace(
            const ACECompressor& component,
            Type inputType,
            uint32_t inputWidth,
            size_t depth)
    {
        std::uniform_int_distribution<int> dist(0, 99);
        const int roll = dist(rng_);
        if (roll < 35) {
            if (component.isGraph()) {
                return buildRandomGraphCompressor(rng_, inputType);
            }
            return mutateParameters(component, inputType, inputWidth, depth);
        } else if (roll < 55) {
            return mutateParameters(component, inputType, inputWidth, depth);
        } else if (roll < 75) {
            return addRandomPipeline(component, inputType, inputWidth, depth);
        } else if (roll < 85) {
            return deleteRandomPipelinePrefix(
                    component, inputType, inputWidth, depth);
        } else if (roll < 95) {
            return copySiblingPort(component, inputType, inputWidth, depth);
        } else {
            std::bernoulli_distribution useSimple(0.5);
            if (useSimple(rng_)) {
                return randomSimpleCompressor(inputType, inputWidth);
            }
            return randomCompressor(inputType, inputWidth, depth);
        }
    }

    ACECompressor copySiblingPort(
            const ACECompressor& component,
            Type inputType,
            uint32_t inputWidth,
            size_t depth)
    {
        if (!component.isNode()) {
            return mutateParameters(component, inputType, inputWidth, depth);
        }
        const auto& nc      = component.asNode();
        const size_t nPorts = nc.successors.size();
        if (nPorts < 2 || nc.node.outputTypes.size() != nPorts) {
            return mutateParameters(component, inputType, inputWidth, depth);
        }
        std::vector<std::pair<size_t, size_t>> distinctPairs;
        for (size_t src = 0; src < nPorts; ++src) {
            if (isProtectedPort(nc.node, src)) {
                continue;
            }
            for (size_t dst = 0; dst < nPorts; ++dst) {
                if (src == dst || isProtectedPort(nc.node, dst)) {
                    continue;
                }
                if (nc.node.outputTypes[src] != nc.node.outputTypes[dst]) {
                    continue;
                }
                const uint32_t dstWidth =
                        inferNodeOutputWidth(nc.node, dst, inputWidth);
                if (!subtreeValidForWidthAndVersion(
                            *nc.successors[src], dstWidth, formatVersion_)) {
                    continue;
                }
                if (*nc.successors[src] != *nc.successors[dst]) {
                    distinctPairs.emplace_back(src, dst);
                }
            }
        }
        if (distinctPairs.empty()) {
            return mutateParameters(component, inputType, inputWidth, depth);
        }
        std::uniform_int_distribution<size_t> pairDist(
                0, distinctPairs.size() - 1);
        const auto [srcPort, dstPort] = distinctPairs[pairDist(rng_)];
        std::vector<std::unique_ptr<ACECompressor>> succs;
        succs.reserve(nPorts);
        for (size_t i = 0; i < nPorts; ++i) {
            if (i == dstPort) {
                succs.push_back(std::make_unique<ACECompressor>(
                        *nc.successors[srcPort]));
            } else {
                succs.push_back(
                        std::make_unique<ACECompressor>(*nc.successors[i]));
            }
        }
        return ACENodeCompressor(nc.node, std::move(succs));
    }

    ACECompressor randomSimpleCompressor(
            Type inputType,
            uint32_t inputWidth = 0)
    {
        const auto prebuilt = getPrebuiltCompressors(inputType);
        std::vector<const ACECompressor*> valid;
        valid.reserve(prebuilt.size());
        for (const auto& c : prebuilt) {
            if (c.acceptsInputType(inputType)
                && subtreeValidForWidthAndVersion(
                        c, inputWidth, formatVersion_)) {
                valid.push_back(&c);
            }
        }
        if (valid.empty()) {
            return buildRandomGraphCompressor(rng_, inputType);
        }
        std::uniform_int_distribution<size_t> dist(0, valid.size() - 1);
        return *valid[dist(rng_)];
    }

    ACECompressor
    randomCompressor(Type inputType, uint32_t inputWidth, size_t depth)
    {
        if (depth >= maxDepth_) {
            return buildRandomGraphCompressor(rng_, inputType);
        } else {
            return buildRandomCompressor(
                    rng_,
                    inputType,
                    formatVersion_,
                    maxDepth_ - depth,
                    inputWidth);
        }
    }

    ACECompressor mutateParameters(
            const ACECompressor& component,
            Type inputType,
            uint32_t inputWidth,
            size_t depth)
    {
        if (component.isGraph()) {
            const auto& g = component.asGraph().graph;
            const auto compatible = getGraphsComptabileWith(inputType);
            std::vector<ACEGraph> variants;
            for (const auto& cand : compatible) {
                if (cand.name == g.name) {
                    variants.push_back(cand);
                }
            }
            if (variants.size() > 1) {
                return ACECompressor(randomChoice(
                        rng_,
                        poly::span<const ACEGraph>(
                                variants.data(), variants.size())));
            }
            return addRandomPipeline(component, inputType, inputWidth, depth);
        }

        const auto& nc = component.asNode();
        if (nc.node.name == "zl.pco_bin_split") {
            std::bernoulli_distribution flipOffsets(0.25);
            if (flipOffsets(rng_)) {
                std::vector<std::unique_ptr<ACECompressor>> succs;
                succs.push_back(
                        std::make_unique<ACECompressor>(*nc.successors[0]));
                const bool currentlyStore =
                        nc.successors[1]->isGraph()
                        && nc.successors[1]->asGraph().graph.name == "zl.store";
                succs.push_back(
                        std::make_unique<ACECompressor>(
                                currentlyStore ? ACECompressor(ACEGraph(
                                                         graphs::Entropy{}))
                                               : buildStoreCompressor()));
                return ACENodeCompressor(nc.node, std::move(succs));
            }
            const auto compatible =
                    getNodesComptabileWith(inputType, formatVersion_);
            std::vector<ACENode> variants;
            for (const auto& cand : compatible) {
                if (cand.name == nc.node.name) {
                    variants.push_back(cand);
                }
            }
            if (variants.size() > 1) {
                auto nextNode = randomChoice(
                        rng_,
                        poly::span<const ACENode>(
                                variants.data(), variants.size()));
                std::vector<std::unique_ptr<ACECompressor>> succs;
                for (const auto& s : nc.successors) {
                    succs.push_back(std::make_unique<ACECompressor>(*s));
                }
                return ACENodeCompressor(std::move(nextNode), std::move(succs));
            }
        } else if (nc.node.name == "zl.int_mult_split") {
            std::bernoulli_distribution flipAdj(0.5);
            if (flipAdj(rng_)) {
                static const std::array<ACEGraph, 5> adjGraphs = {
                    ACEGraph(graphs::Store{}),
                    ACEGraph(graphs::Compress{}),
                    ACEGraph(graphs::Entropy{}),
                    ACEGraph(graphs::Bitpack{}),
                    ACEGraph(graphs::Constant{}),
                };
                std::vector<std::unique_ptr<ACECompressor>> succs;
                succs.push_back(
                        std::make_unique<ACECompressor>(*nc.successors[0]));
                std::uniform_int_distribution<size_t> adjDist(
                        0, adjGraphs.size() - 1);
                succs.push_back(
                        std::make_unique<ACECompressor>(
                                ACEGraphCompressor(adjGraphs[adjDist(rng_)])));
                return ACENodeCompressor(nc.node, std::move(succs));
            }
            const auto compatible =
                    getNodesComptabileWith(inputType, formatVersion_);
            std::vector<ACENode> variants;
            for (const auto& cand : compatible) {
                if (cand.name == nc.node.name) {
                    variants.push_back(cand);
                }
            }
            if (variants.size() > 1) {
                auto nextNode = randomChoice(
                        rng_,
                        poly::span<const ACENode>(
                                variants.data(), variants.size()));
                std::vector<std::unique_ptr<ACECompressor>> succs;
                for (const auto& s : nc.successors) {
                    succs.push_back(std::make_unique<ACECompressor>(*s));
                }
                return ACENodeCompressor(std::move(nextNode), std::move(succs));
            }
        } else if (nc.node.name == "zl.float_mult_split") {
            std::vector<std::unique_ptr<ACECompressor>> succs;
            succs.push_back(
                    std::make_unique<ACECompressor>(*nc.successors[0]));
            const bool currentlyPco = nc.successors[1]->isNode()
                    && nc.successors[1]->asNode().node.name
                            == "zl.pco_bin_split";
            const bool currentlyStore = nc.successors[1]->isGraph()
                    && nc.successors[1]->asGraph().graph.name == "zl.store";
            if (currentlyPco) {
                succs.push_back(std::make_unique<ACECompressor>(
                        buildCompressGenericCompressor()));
            } else if (currentlyStore) {
                succs.push_back(std::make_unique<ACECompressor>(
                        ACENode(nodes::PcoBinSplit{
                                nodes::PcoBinSplit::LatentMode::Signed }),
                        std::vector<ACECompressor>{
                                ACECompressor(ACEGraph(graphs::Fse{})),
                                buildStoreCompressor() }));
            } else {
                succs.push_back(
                        std::make_unique<ACECompressor>(buildStoreCompressor()));
            }
            return ACENodeCompressor(nc.node, std::move(succs));
        } else if (
                nc.node.name == "zl.tokenize_numeric"
                || nc.node.name == "zl.lookback_delta"
                || nc.node.name == "zl.partition"
                || nc.node.name == "zl.sparse_num"
                || nc.node.name == "zl.bitunpack"
                || nc.node.name == "zl.ctx_split"
                || nc.node.name == "zl.ctx_split_num") {
            const auto compatible =
                    getNodesComptabileWith(inputType, formatVersion_);
            std::vector<ACENode> variants;
            for (const auto& cand : compatible) {
                if (cand.name == nc.node.name
                    && nodeAcceptsWidth(cand, inputWidth)) {
                    variants.push_back(cand);
                }
            }
            if (variants.size() > 1) {
                auto nextNode = randomChoice(
                        rng_,
                        poly::span<const ACENode>(
                                variants.data(), variants.size()));
                std::vector<std::unique_ptr<ACECompressor>> succs;
                for (const auto& s : nc.successors) {
                    succs.push_back(std::make_unique<ACECompressor>(*s));
                }
                return ACENodeCompressor(std::move(nextNode), std::move(succs));
            }
        }
        return addRandomPipeline(component, inputType, inputWidth, depth);
    }

    ACECompressor deleteRandomPipelinePrefix(
            const ACECompressor& compressor,
            Type inputType,
            uint32_t inputWidth,
            size_t depth)
    {
        const ACECompressor* pipeline = &compressor;
        // Choose the last node to delete (unwrapping port 0 when valid)
        ACEReservoirSampler<const ACECompressor> sampler(rng_);
        while (pipeline->isNode()) {
            const auto& node = pipeline->asNode();
            if (node.successors.empty()) {
                break;
            }
            if (node.successors[0]->acceptsInputType(inputType)) {
                sampler.update(*pipeline);
            }
            if (node.successors.size() != 1) {
                break;
            }
            pipeline = node.successors[0].get();
        }
        if (sampler.get() == nullptr) {
            return randomCompressor(inputType, inputWidth, depth);
        }
        auto replacement = *sampler.get()->asNode().successors[0];
        assert(replacement.acceptsInputType(inputType));
        return replacement;
    }

    ACECompressor addRandomPipeline(
            const ACECompressor& compressor,
            Type inputType,
            uint32_t inputWidth,
            size_t depth)
    {
        if (depth >= maxDepth_) {
            return randomSimpleCompressor(inputType, inputWidth);
        }
        const auto compatibleNodes =
                getNodesComptabileWith(inputType, formatVersion_);
        const std::string childNodeName =
                compressor.isNode() ? compressor.asNode().node.name : "";
        std::vector<ACENode> accepted;
        for (const auto& node : compatibleNodes) {
            if (!nodeAcceptsWidth(node, inputWidth)) {
                continue;
            }
            if ((node.name == "zl.zigzag" && childNodeName == "zl.zigzag")
                || (node.name == "zl.range_pack"
                    && childNodeName == "zl.range_pack")
                || (node.name == "zl.int_mult_split"
                    && childNodeName == "zl.int_mult_split")
                || (node.name == "zl.float_mult_split"
                    && childNodeName == "zl.float_mult_split")
                || (node.name == "zl.lookback_delta"
                    && childNodeName == "zl.lookback_delta")
                || (node.name == "zl.pco_bin_split"
                    && childNodeName == "zl.pco_bin_split")
                || (isCtxSplitNodeName(node.name)
                    && isCtxSplitNodeName(childNodeName))) {
                continue;
            }
            if (!node.outputTypes.empty()
                && compressor.acceptsInputType(node.outputTypes[0])) {
                if (node.outputTypes.size() == 1
                    || (inputType == Type::Numeric
                        && node.outputTypes[0] == Type::Numeric)) {
                    accepted.push_back(node);
                }
            }
        }
        if (accepted.empty()) {
            return randomSimpleCompressor(inputType, inputWidth);
        }
        auto node = pickRandomNodeForType(
                rng_,
                inputType,
                poly::span<const ACENode>(accepted.data(), accepted.size()));
        std::vector<ACECompressor> successors;
        successors.reserve(node.outputTypes.size());
        successors.push_back(compressor);
        const size_t childBudget =
                maxDepth_ > depth + 1 ? maxDepth_ - depth - 1 : 0;
        for (size_t i = 1; i < node.outputTypes.size(); ++i) {
            if (node.name == "zl.pco_bin_split" && i == 1) {
                std::bernoulli_distribution useStore(0.85);
                successors.push_back(
                        useStore(rng_)
                                ? buildStoreCompressor()
                                : ACECompressor(ACEGraph(graphs::Entropy{})));
            } else if (node.name == "zl.int_mult_split" && i == 1) {
                static const std::array<ACEGraph, 5> adjGraphs = {
                    ACEGraph(graphs::Store{}),
                    ACEGraph(graphs::Compress{}),
                    ACEGraph(graphs::Entropy{}),
                    ACEGraph(graphs::Bitpack{}),
                    ACEGraph(graphs::Constant{}),
                };
                std::uniform_int_distribution<size_t> adjDist(
                        0, adjGraphs.size() - 1);
                successors.push_back(ACECompressor(
                        ACEGraphCompressor(adjGraphs[adjDist(rng_)])));
            } else if (node.name == "zl.float_mult_split" && i == 1) {
                std::uniform_int_distribution<int> adjChoice(0, 2);
                const int choice = adjChoice(rng_);
                if (choice == 0) {
                    successors.push_back(ACECompressor(
                            ACENode(nodes::PcoBinSplit{
                                    nodes::PcoBinSplit::LatentMode::Signed }),
                            std::vector<ACECompressor>{
                                    ACECompressor(ACEGraph(graphs::Fse{})),
                                    buildStoreCompressor() }));
                } else if (choice == 1) {
                    successors.push_back(buildCompressGenericCompressor());
                } else {
                    successors.push_back(buildStoreCompressor());
                }
            } else {
                const uint32_t outWidth =
                        inferNodeOutputWidth(node, i, inputWidth);
                successors.push_back(buildRandomCompressor(
                        rng_,
                        node.outputTypes[i],
                        formatVersion_,
                        childBudget,
                        outWidth));
            }
        }
        return ACECompressor(std::move(node), std::move(successors));
    }

    std::mt19937_64& rng_;
    Type inputType_;
    uint32_t formatVersion_;
    size_t maxDepth_;
};

} // namespace training
} // namespace openzl
