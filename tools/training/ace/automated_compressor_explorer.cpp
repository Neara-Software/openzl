// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "tools/training/ace/automated_compressor_explorer.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <vector>

#include "openzl/common/a1cbor_helpers.h"
#include "openzl/common/allocation.h"
#include "openzl/shared/a1cbor.h"

namespace openzl {
namespace training {

std::vector<ACECompressor> AutomatedCompressorExplorer::initialPopulation()
{
    // Use all prebuilt compressors
    auto prebuilt = getPrebuiltCompressors(inputType());
    std::vector<ACECompressor> population;
    population.insert(population.end(), prebuilt.begin(), prebuilt.end());

    // Use populationSize() random compressors
    for (size_t i = 0; i < populationSize(); ++i) {
        population.push_back(
                buildRandomCompressor(rng(), inputType(), formatVersion_));
    }
    return population;
}

namespace {
/// @returns Up to @p maxParts contiguous runs of @p inputs with about the
/// same number of bytes each.
std::vector<poly::span<const Input>> splitInputs(
        poly::span<const Input> inputs,
        size_t maxParts)
{
    const size_t numParts = std::max<size_t>(
            1, std::min(maxParts, inputs.size()));
    size_t total = 0;
    for (const auto& input : inputs) {
        total += input.contentSize();
    }
    std::vector<poly::span<const Input>> parts;
    size_t begin = 0;
    size_t bytes = 0;
    for (size_t i = 0; i < inputs.size(); ++i) {
        bytes += inputs[i].contentSize();
        const bool last = parts.size() + 1 == numParts;
        if (!last && bytes * numParts >= total * (parts.size() + 1)) {
            parts.push_back(inputs.subspan(begin, i + 1 - begin));
            begin = i + 1;
        }
    }
    if (begin < inputs.size()) {
        parts.push_back(inputs.subspan(begin));
    }
    return parts;
}

/// Put small evolutionary pressure on the results to be simpler.
/// Adjusts cSize on the scale of 0.1%, and times on the scale of 1%, given a
/// typical number of components of ~10.
void adjustResults(const ACECompressor& gene, std::vector<float>& results)
{
    std::array<float, 3> scale = { 0.0001, 0.001, 0.001 };
    const size_t numComponents = gene.numComponents();
    for (size_t i = 0; i < results.size(); ++i) {
        const auto delta = results[i] * scale[i] * numComponents;
        results[i] += std::max<float>(numComponents, delta);
    }
}
} // namespace

/* static */ std::vector<float> AutomatedCompressorExplorer::fitnessOf(
        const ACECompressor& gene,
        const poly::optional<ACECompressionResult>& result)
{
    if (!result.has_value()) {
        return std::vector<float>(3, std::numeric_limits<float>::infinity());
    }
    auto fitness = result->asFloatVector();
    adjustResults(gene, fitness);
    // ACE_SIZE_ONLY: time axes stay constant so dominance and crowding use
    // size alone. ACE_NO_CTIME: drop only the compression-time objective, so
    // the front trades size against decompression time.
    static const bool sizeOnly = std::getenv("ACE_SIZE_ONLY") != nullptr;
    static const bool noCTime  = std::getenv("ACE_NO_CTIME") != nullptr;
    if (sizeOnly || noCTime) {
        fitness[1] = 0;
    }
    if (sizeOnly) {
        fitness[2] = 0;
    }
    return fitness;
}

/* static */ std::vector<float> AutomatedCompressorExplorer::computeFitness(
        const ACECompressor& gene,
        poly::span<const Input> inputs,
        uint32_t formatVersion,
        int compressionLevel)
{
    return fitnessOf(
            gene, gene.benchmark(inputs, formatVersion, compressionLevel));
}

std::vector<float> AutomatedCompressorExplorer::computeFitness(
        const ACECompressor& gene)
{
    return computeFitness(gene, inputs_, formatVersion_, compressionLevel_);
}

std::vector<std::vector<float>> AutomatedCompressorExplorer::computeFitness(
        poly::span<const ACECompressor> genes)
{
    // Each gene is benchmarked in up to numThreads parts of the inputs, so a
    // slow gene does not hold the generation on one thread. Sizes and times
    // are integer sums, so the parts add up to the whole-input result.
    const auto parts = splitInputs(inputs_, threadPool_->numThreads);
    using PartResult = poly::optional<ACECompressionResult>;
    std::vector<poly::optional<std::vector<float>>> cached(genes.size());
    std::vector<std::vector<std::future<PartResult>>> futures(genes.size());
    for (size_t g = 0; g < genes.size(); ++g) {
        const auto& gene = genes[g];
        auto cache       = cachedFitness_.find(gene.hash());
        if (cache != cachedFitness_.end()) {
            std::uniform_int_distribution<size_t> dist(0, cache->second.first);
            if (dist(rng()) != 0) {
                cached[g] = cache->second.second;
                continue;
            }
        }
        for (const auto& part : parts) {
            futures[g].push_back(threadPool_->run(
                    [part,
                     &gene,
                     formatVersion    = formatVersion_,
                     compressionLevel = compressionLevel_] {
                        return gene.benchmark(
                                part, formatVersion, compressionLevel);
                    }));
        }
    }

    std::vector<std::vector<float>> results;
    results.reserve(genes.size());
    for (size_t i = 0; i < genes.size(); ++i) {
        if (cached[i].has_value()) {
            results.push_back(std::move(*cached[i]));
            continue;
        }
        PartResult total = ACECompressionResult{};
        for (auto& future : futures[i]) {
            auto part = future.get();
            if (!part.has_value()) {
                total = poly::nullopt;
            } else if (total.has_value()) {
                *total += *part;
            }
        }
        auto result            = fitnessOf(genes[i], total);
        auto [cache, inserted] = cachedFitness_.emplace(
                genes[i].hash(), std::make_pair(1, result));
        if (!inserted) {
            for (size_t j = 0; j < result.size(); ++j) {
                cache->second.second[j] =
                        std::min(cache->second.second[j], result[j]);
            }
            ++cache->second.first;
        }
        results.push_back(std::move(result));
    }
    return results;
}

namespace {
struct ArenaDeleter {
    void operator()(Arena* arena) const
    {
        ALLOC_Arena_freeArena(arena);
    }
};
} // namespace

namespace {
/// @returns @p gene with its @p leaf-th graph component (in forEachComponent
/// order) replaced by @p graph.
ACECompressor replaceLeaf(
        const ACECompressor& gene,
        Type inputType,
        size_t leaf,
        const ACEGraph& graph)
{
    size_t seen = 0;
    return gene.replace(
            inputType,
            [&](const ACECompressor& c,
                Type,
                size_t) -> poly::optional<ACECompressor> {
                if (c.isNode()) {
                    return poly::nullopt;
                }
                if (seen++ == leaf) {
                    return ACECompressor(ACEGraphCompressor(graph));
                }
                return c;
            });
}
} // namespace

ACECompressor AutomatedCompressorExplorer::polishBackends(
        const ACECompressor& gene,
        float size)
{
    const auto backends = getPolishBackends();
    std::vector<Type> leafTypes;
    gene.forEachComponent(inputType(), [&](const ACECompressor& c, Type t) {
        if (!c.isNode()) {
            leafTypes.push_back(t);
        }
    });
    ACECompressor best = gene;
    for (size_t leaf = 0; leaf < leafTypes.size(); ++leaf) {
        std::vector<ACECompressor> tries;
        for (const auto& b : backends) {
            if (isCompatible(b.inputTypeMask, leafTypes[leaf])) {
                tries.push_back(replaceLeaf(best, inputType(), leaf, b));
            }
        }
        std::vector<std::future<std::vector<float>>> futures;
        for (const auto& t : tries) {
            futures.push_back(threadPool_->run([&inputs = inputs_,
                                               &t,
                                               formatVersion = formatVersion_,
                                               compressionLevel =
                                                       compressionLevel_] {
                return computeFitness(
                        t, inputs, formatVersion, compressionLevel);
            }));
        }
        for (size_t i = 0; i < tries.size(); ++i) {
            auto f = futures[i].get();
            if (f[0] < size) {
                size = f[0];
                best = tries[i];
            }
        }
    }
    return best;
}

std::unordered_set<ACECompressor> AutomatedCompressorExplorer::reproduce(
        size_t numChildren)
{
    static const size_t numPolish = [] {
        const char* p = std::getenv("ACE_POLISH");
        return p ? (size_t)std::strtoul(p, nullptr, 10) : 0;
    }();
    auto children = Base::reproduce(numChildren);
    if (numPolish == 0) {
        return children;
    }
    std::vector<size_t> order;
    for (size_t i = 0; i < population().size(); ++i) {
        if (rank()[i] == 0 && polished_.count(population()[i].hash()) == 0) {
            order.push_back(i);
        }
    }
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return fitness()[a][0] < fitness()[b][0];
    });
    for (size_t k = 0; k < std::min(numPolish, order.size()); ++k) {
        const auto& gene = population()[order[k]];
        polished_.insert(gene.hash());
        auto better = polishBackends(gene, fitness()[order[k]][0]);
        polished_.insert(better.hash());
        if (better != gene && populationSet().count(better) == 0) {
            children.insert(std::move(better));
        }
    }
    return children;
}

std::string AutomatedCompressorExplorer::savePopulation() const
{
    return savePopulation(population().size());
}

std::string AutomatedCompressorExplorer::savePopulation(size_t limit) const
{
    std::vector<size_t> order(population().size());
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    if (limit < order.size()) {
        std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) {
            return fitness()[x][0] < fitness()[y][0];
        });
        order.resize(limit);
    }
    std::unique_ptr<Arena, ArenaDeleter> arena(ALLOC_HeapArena_create());
    auto a         = A1C_Arena_wrap(arena.get());
    A1C_Item* item = A1C_Item_root(&a);
    if (item == nullptr) {
        throw std::bad_alloc();
    }
    A1C_Item* array = A1C_Item_array(item, order.size(), &a);
    if (array == nullptr) {
        throw std::bad_alloc();
    }
    for (size_t idx : order) {
        const auto s = population()[idx].serialize();
        if (!A1C_Item_string_copy(array++, s.data(), s.size(), &a)) {
            throw std::bad_alloc();
        }
    }
    const size_t size = A1C_Item_encodedSize(item);
    std::string serialized(size, '\0');
    const size_t actual = A1C_Item_encode(
            item, (uint8_t*)serialized.data(), serialized.size(), nullptr);
    if (size != actual) {
        throw Exception("Serialization failed");
    }
    return serialized;
}

void AutomatedCompressorExplorer::loadPopulation(poly::string_view snapshot)
{
    std::unique_ptr<Arena, ArenaDeleter> arena(ALLOC_HeapArena_create());
    A1C_Decoder decoder;
    A1C_Decoder_init(&decoder, A1C_Arena_wrap(arena.get()), {});
    const A1C_Item* item = A1C_Decoder_decode(
            &decoder, (const uint8_t*)snapshot.data(), snapshot.size());
    if (item == nullptr) {
        throw Exception("Failed to deserialize");
    }
    if (item->type != A1C_ItemType_array) {
        throw Exception("Must be an array");
    }
    std::vector<ACECompressor> population;
    population.reserve(item->array.size);
    for (size_t i = 0; i < item->array.size; ++i) {
        const A1C_Item& p = item->array.items[i];
        if (p.type != A1C_ItemType_string) {
            throw Exception("Array elements must be strings");
        }
        population.emplace_back(
                poly::string_view{ p.string.data, p.string.size });
    }
    extendPopulation(std::move(population));
}

} // namespace training
} // namespace openzl
