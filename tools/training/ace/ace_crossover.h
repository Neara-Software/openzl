// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include "tools/training/ace/ace_compressor.h"
#include "tools/training/ace/ace_mutate.h"
#include "tools/training/ace/ace_sampling.h"

namespace openzl {
namespace training {

/// Crosses over two ACECompressors to produce a new ACECompressor that inherits
/// a comination of their traits.
class ACECrossover {
   public:
    ACECrossover(
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

    ACECompressor operator()(
            const ACECompressor& parent1,
            const ACECompressor& parent2)
    {
        // Shuffle the parents
        std::uniform_int_distribution<int> dist(0, 1);
        if (dist(rng_)) {
            return crossover(parent1, parent2);
        } else {
            return crossover(parent2, parent1);
        }
    }

   private:
    ACECompressor crossover(
            const ACECompressor& donor,
            const ACECompressor& recipient)
    {
        for (size_t attempt = 0; attempt < 5; ++attempt) {
            auto donorComponent = getRandomComponent(donor);
            auto child          = replaceRandomComponent(
                    recipient, std::move(donorComponent));
            if (child.has_value()) {
                return std::move(*child);
            }
        }
        return ACEMutate(rng_, inputType_, formatVersion_, maxDepth_)(recipient);
    }

    static bool isProtectedPort(const ACENodeCompressor& node, size_t port)
    {
        return port == 1
                && (node.node.name == "zl.pco_bin_split"
                    || node.node.name == "zl.int_mult_split"
                    || node.node.name == "zl.float_mult_split");
    }

    template <typename Fn>
    static void forEachMutable(
            const ACECompressor& comp,
            Type inputType,
            Fn&& fn)
    {
        fn(comp, inputType);
        if (comp.isNode()) {
            const auto& node = comp.asNode();
            for (size_t i = 0; i < node.successors.size(); ++i) {
                if (isProtectedPort(node, i)) {
                    continue;
                }
                forEachMutable(
                        *node.successors[i], node.node.outputTypes[i], fn);
            }
        }
    }

    template <typename Fn>
    static ACECompressor replaceMutable(
            const ACECompressor& comp,
            Type inputType,
            Fn&& fn)
    {
        auto replacement = fn(comp, inputType);
        if (replacement.has_value()) {
            return std::move(*replacement);
        }
        if (comp.isNode()) {
            const auto& node = comp.asNode();
            std::vector<std::unique_ptr<ACECompressor>> successors;
            successors.reserve(node.successors.size());
            for (size_t i = 0; i < node.successors.size(); ++i) {
                if (isProtectedPort(node, i)) {
                    successors.push_back(
                            std::make_unique<ACECompressor>(*node.successors[i]));
                } else {
                    successors.push_back(std::make_unique<ACECompressor>(
                            replaceMutable(
                                    *node.successors[i],
                                    node.node.outputTypes[i],
                                    fn)));
                }
            }
            return ACENodeCompressor(node.node, std::move(successors));
        }
        return comp;
    }

    ACECompressor getRandomComponent(const ACECompressor& donor)
    {
        ACEReservoirSampler<const ACECompressor> sampler(rng_);
        forEachMutable(donor, inputType_, [&](const auto& component, auto) {
            sampler.update(component);
        });
        return *sampler.get();
    }

    poly::optional<ACECompressor> replaceRandomComponent(
            const ACECompressor& recipient,
            ACECompressor&& donorComponent)
    {
        ACEReservoirSampler<const ACECompressor> sampler(rng_);
        forEachMutable(
                recipient, inputType_, [&](const auto& component, auto inputType) {
                    if (donorComponent.acceptsInputType(inputType)) {
                        sampler.update(component);
                    }
                });
        if (sampler.get()) {
            return replaceMutable(
                    recipient,
                    inputType_,
                    [&donorComponent, target = sampler.get()](
                            const auto& component,
                            auto inputType) mutable -> poly::optional<ACECompressor> {
                        if (&component == target) {
                            assert(donorComponent.acceptsInputType(inputType));
                            return std::move(donorComponent);
                        } else {
                            return poly::nullopt;
                        }
                    });
        } else {
            return poly::nullopt;
        }
    }

    std::mt19937_64& rng_;
    Type inputType_;
    uint32_t formatVersion_;
    size_t maxDepth_;
};

} // namespace training
} // namespace openzl
