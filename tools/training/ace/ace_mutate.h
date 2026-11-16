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
                [&](const auto& component,
                    auto inputType,
                    size_t depth) -> poly::optional<ACECompressor> {
                    if (&component == sampler.get()) {
                        return replace(component, inputType, depth);
                    } else {
                        return poly::nullopt;
                    }
                });
    }

   private:
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
                if ((node.node.name == "zl.pco_bin_split"
                     || node.node.name == "zl.int_mult_split"
                     || node.node.name == "zl.float_mult_split")
                    && i == 1) {
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
            size_t depth,
            ReplaceFn&& replaceFn)
    {
        auto replacement = replaceFn(comp, inputType, depth);
        if (replacement.has_value()) {
            return std::move(*replacement);
        }
        if (comp.isNode()) {
            const auto& node = comp.asNode();
            std::vector<std::unique_ptr<ACECompressor>> successors;
            successors.reserve(node.successors.size());
            for (size_t i = 0; i < node.successors.size(); ++i) {
                if ((node.node.name == "zl.pco_bin_split"
                     || node.node.name == "zl.int_mult_split"
                     || node.node.name == "zl.float_mult_split")
                    && i == 1) {
                    successors.push_back(
                            std::make_unique<ACECompressor>(*node.successors[i]));
                } else {
                    auto s = replaceMutable(
                            *node.successors[i],
                            node.node.outputTypes[i],
                            depth + 1,
                            replaceFn);
                    successors.push_back(
                            std::make_unique<ACECompressor>(std::move(s)));
                }
            }
            return ACENodeCompressor(node.node, std::move(successors));
        } else {
            return comp;
        }
    }

    ACECompressor
    replace(const ACECompressor& component, Type inputType, size_t depth)
    {
        std::uniform_int_distribution<int> dist(0, 4);
        auto choice = dist(rng_);
        if (choice == 0) {
            return randomSimpleCompressor(inputType);
        } else if (choice == 1) {
            return randomCompressor(inputType, depth);
        } else if (choice == 2) {
            return deleteRandomPipelinePrefix(component, inputType, depth);
        } else if (choice == 3) {
            return addRandomPipeline(component, inputType, depth);
        } else {
            return mutateParameters(component, inputType, depth);
        }
    }

    ACECompressor randomSimpleCompressor(Type inputType)
    {
        return randomChoice(rng_, getPrebuiltCompressors(inputType));
    }

    ACECompressor randomCompressor(Type inputType, size_t depth)
    {
        if (depth >= maxDepth_) {
            return buildRandomGraphCompressor(rng_, inputType);
        } else {
            return buildRandomCompressor(
                    rng_, inputType, formatVersion_, maxDepth_ - depth);
        }
    }

    ACECompressor mutateParameters(
            const ACECompressor& component,
            Type inputType,
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
            return addRandomPipeline(component, inputType, depth);
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
                std::vector<std::unique_ptr<ACECompressor>> succs;
                succs.push_back(
                        std::make_unique<ACECompressor>(*nc.successors[0]));
                const bool currentlyStore =
                        nc.successors[1]->isGraph()
                        && nc.successors[1]->asGraph().graph.name == "zl.store";
                succs.push_back(
                        std::make_unique<ACECompressor>(
                                currentlyStore ? ACECompressor(ACEGraph(
                                                         graphs::Compress{}))
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
                || nc.node.name == "zl.lookback_delta") {
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
        }
        return addRandomPipeline(component, inputType, depth);
    }

    ACECompressor deleteRandomPipelinePrefix(
            const ACECompressor& compressor,
            Type inputType,
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
            return randomCompressor(inputType, depth);
        }
        auto replacement = *sampler.get()->asNode().successors[0];
        assert(replacement.acceptsInputType(inputType));
        return replacement;
    }

    ACECompressor addRandomPipeline(
            const ACECompressor& compressor,
            Type inputType,
            size_t depth)
    {
        if (depth >= maxDepth_) {
            return randomSimpleCompressor(inputType);
        }
        const auto compatibleNodes =
                getNodesComptabileWith(inputType, formatVersion_);
        const std::string childNodeName =
                compressor.isNode() ? compressor.asNode().node.name : "";
        std::vector<ACENode> accepted;
        for (const auto& node : compatibleNodes) {
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
                    && childNodeName == "zl.pco_bin_split")) {
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
            return randomSimpleCompressor(inputType);
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
                std::bernoulli_distribution useCompress(0.5);
                successors.push_back(
                        useCompress(rng_)
                                ? ACECompressor(ACEGraph(graphs::Compress{}))
                                : buildStoreCompressor());
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
                successors.push_back(buildRandomCompressor(
                        rng_,
                        node.outputTypes[i],
                        formatVersion_,
                        childBudget));
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
