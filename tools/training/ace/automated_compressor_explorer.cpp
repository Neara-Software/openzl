// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "tools/training/ace/automated_compressor_explorer.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <map>
#include <memory>
#include <vector>

#include "openzl/common/a1cbor_helpers.h"
#include "openzl/common/allocation.h"
#include "openzl/shared/a1cbor.h"
#include "tools/training/ace/ace_features.h"

namespace openzl {
namespace training {

/// ACE_TF_COST=c: each Transformer graph in @p gene scales its size by
/// (1 + c), so the search keeps a Transformer only where it saves more than
/// c. Its compression cost is otherwise invisible under ACE_NO_CTIME.
double transformerSizePenalty(const ACECompressor& gene)
{
    static const double cost = [] {
        const char* c = std::getenv("ACE_TF_COST");
        return c ? std::strtod(c, nullptr) : 0.0;
    }();
    if (cost <= 0) {
        return 1;
    }
    size_t n = 0;
    gene.forEachComponent(Type::Serial, [&n](const ACECompressor& c, auto) {
        if (c.isGraph() && c.asGraph().graph.name == "zl.transformer_numeric") {
            ++n;
        }
    });
    return 1 + cost * (double)n;
}

std::vector<ACECompressor> AutomatedCompressorExplorer::initialPopulation()
{
    // Use all prebuilt compressors
    auto prebuilt = getPrebuiltCompressors(inputType());
    std::vector<ACECompressor> population;
    population.insert(population.end(), prebuilt.begin(), prebuilt.end());

    // Append targeted feature-guided seeds before extendPopulation deduplicates
    auto featureSeeds =
            buildFeatureGuidedSeeds(inputs_, inputType(), formatVersion_);
    population.insert(
            population.end(),
            std::make_move_iterator(featureSeeds.begin()),
            std::make_move_iterator(featureSeeds.end()));

    // Use populationSize() random compressors
    for (size_t i = 0; i < populationSize(); ++i) {
        population.push_back(
                buildRandomCompressor(rng(), inputType(), formatVersion_));
    }
    return population;
}

namespace {
/// Put small evolutionary pressure on the results to be simpler.
/// Adjusts cSize on the scale of 0.1%, and times on the scale of 1%, given a
/// typical number of components of ~10.
double parsimonyScale()
{
    static const double s = [] {
        const char* p = std::getenv("ACE_PARSIMONY_SCALE");
        return p ? std::strtod(p, nullptr) : 1.0;
    }();
    return s;
}

void adjustResults(const ACECompressor& gene, std::vector<float>& results)
{
    const double ps = parsimonyScale();
    if (ps <= 0.0) {
        return;
    }
    std::array<float, 3> scale = { 0.0001, 0.001, 0.001 };
    const size_t numComponents = gene.numComponents();
    for (size_t i = 0; i < results.size(); ++i) {
        const auto delta = results[i] * scale[i] * numComponents;
        results[i] += static_cast<float>(ps) * std::max<float>(numComponents, delta);
    }
}

std::vector<float> fitnessFromResult(
        const ACECompressor& gene,
        const poly::optional<ACECompressionResult>& result)
{
    if (!result.has_value()) {
        return std::vector<float>(3, std::numeric_limits<float>::infinity());
    }
    auto fitness = result->asFloatVector();
    adjustResults(gene, fitness);
    const float compPenalty = static_cast<float>(
            1.0 + parsimonyScale() * 1e-5 * static_cast<double>(gene.numComponents()));
    fitness[0] *= (float)transformerSizePenalty(gene) * compPenalty;
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

/// The bytes that `benchmark()` compresses for @p input. For a String input,
/// that includes the string lengths.
size_t benchmarkBytes(const Input& input)
{
    size_t bytes = input.contentSize();
    if (input.type() == Type::String) {
        bytes += input.numElts() * sizeof(uint32_t);
    }
    return bytes;
}

/// Benchmarks @p batch with contexts that the calling thread keeps across
/// calls, so that only a thread's first benchmark runs on cold contexts.
poly::optional<ACECompressionResult> benchmarkOnThreadContexts(
        const Compressor& compressor,
        poly::span<const Input> batch)
{
    thread_local CCtx cctx;
    thread_local DCtx dctx;
    return benchmark(cctx, dctx, compressor, batch);
}
} // namespace

/* static */ std::vector<poly::span<const Input>>
AutomatedCompressorExplorer::splitInputs(
        poly::span<const Input> inputs,
        size_t numBatches)
{
    size_t totalBytes = 0;
    for (const auto& input : inputs) {
        totalBytes += benchmarkBytes(input);
    }
    numBatches = std::clamp<size_t>(
            totalBytes / kMinBatchBytes, 1, std::max<size_t>(numBatches, 1));

    // Batch k ends before the first input that starts at or after k /
    // numBatches of the total bytes.
    std::vector<poly::span<const Input>> batches;
    size_t begin = 0;
    size_t bytes = 0;
    size_t batch = 1;
    for (size_t i = 0; i < inputs.size(); ++i) {
        if (i > begin && batch < numBatches
            && bytes * numBatches >= batch * totalBytes) {
            batches.push_back(inputs.subspan(begin, i - begin));
            begin = i;
            while (batch < numBatches
                   && bytes * numBatches >= batch * totalBytes) {
                ++batch;
            }
        }
        bytes += benchmarkBytes(inputs[i]);
    }
    if (begin < inputs.size()) {
        batches.push_back(inputs.subspan(begin, inputs.size() - begin));
    }
    return batches;
}

/* static */ std::vector<float> AutomatedCompressorExplorer::computeFitness(
        const ACECompressor& gene,
        poly::span<const Input> inputs,
        uint32_t formatVersion,
        int compressionLevel)
{
    return fitnessFromResult(
            gene, gene.benchmark(inputs, formatVersion, compressionLevel));
}

std::vector<float> AutomatedCompressorExplorer::computeFitness(
        const ACECompressor& gene)
{
    return fitnessFromResult(
            gene, gene.benchmark(inputs_, formatVersion_, compressionLevel_));
}

std::vector<std::vector<float>> AutomatedCompressorExplorer::computeFitness(
        poly::span<const ACECompressor> genes)
{
    static const int probeEnv = [] {
        const char* p = std::getenv("ACE_PROBE_PRUNE");
        if (!p) {
            return -1;
        }
        return std::atoi(p) != 0 ? 1 : 0;
    }();
    static const double probeMargin = [] {
        const char* p = std::getenv("ACE_PROBE_MARGIN");
        return p ? std::strtod(p, nullptr) : 2.0;
    }();
    const size_t numBatches = inputBatches_.size();
    const bool probeEnabled = generation() > 0 && numBatches >= 4
            && (probeEnv == 1 || (probeEnv != 0 && inputs_.size() >= 16));
    const size_t probeIdx0 = 0;
    const size_t probeIdx1 = numBatches / 2;

    double totalAllBytes = 0.0;
    for (const auto& input : inputs_) {
        totalAllBytes += static_cast<double>(benchmarkBytes(input));
    }
    double probeInputBytes = 0.0;
    if (numBatches >= 2) {
        for (const auto& in : inputBatches_[probeIdx0]) {
            probeInputBytes += static_cast<double>(benchmarkBytes(in));
        }
        if (probeIdx1 != probeIdx0) {
            for (const auto& in : inputBatches_[probeIdx1]) {
                probeInputBytes += static_cast<double>(benchmarkBytes(in));
            }
        }
    }

    struct GeneEval {
        std::reference_wrapper<const ACECompressor> gene;
        poly::optional<std::vector<float>> cachedFitness;
        std::shared_ptr<const Compressor> compressor;
        std::vector<std::future<poly::optional<ACECompressionResult>>>
                batchResults;
        poly::optional<ACECompressionResult> probeRes0;
        poly::optional<ACECompressionResult> probeRes1;
        double probeSize{ std::numeric_limits<double>::infinity() };
        bool probeOk{ false };
        bool pruned{ false };
    };
    // A gene that reuses its cached fitness gets no tasks. When probe pruning is
    // enabled, uncached genes first evaluate two representative batches; genes
    // far worse than bestProbeSoFar_ skip the remaining batches.
    std::vector<GeneEval> evals;
    evals.reserve(genes.size());
    for (const auto& gene : genes) {
        auto& eval = evals.emplace_back(
                GeneEval{ std::cref(gene), poly::nullopt, nullptr, {} });
        auto cache = cachedFitness_.find(gene.hash());
        if (cache != cachedFitness_.end()) {
            std::uniform_int_distribution<size_t> dist(0, cache->second.first);
            if (dist(rng()) != 0) {
                eval.cachedFitness = cache->second.second;
                continue;
            }
        }
        eval.compressor = std::make_shared<Compressor>(
                gene.buildBenchmarkCompressor(
                        formatVersion_, compressionLevel_));
        eval.batchResults.resize(numBatches);
        if (probeEnabled) {
            eval.batchResults[probeIdx0] = threadPool_->run(
                    [c = eval.compressor, b = inputBatches_[probeIdx0]] {
                        return benchmarkOnThreadContexts(*c, b);
                    });
            if (probeIdx1 != probeIdx0) {
                eval.batchResults[probeIdx1] = threadPool_->run(
                        [c = eval.compressor, b = inputBatches_[probeIdx1]] {
                            return benchmarkOnThreadContexts(*c, b);
                        });
            }
        } else {
            for (size_t b = 0; b < numBatches; ++b) {
                eval.batchResults[b] = threadPool_->run(
                        [c = eval.compressor, batch = inputBatches_[b]] {
                            return benchmarkOnThreadContexts(*c, batch);
                        });
            }
        }
    }

    if (probeEnabled) {
        double minProbeThisBatch = bestProbeSoFar_;
        for (auto& eval : evals) {
            if (eval.cachedFitness.has_value()) {
                continue;
            }
            eval.probeRes0 = eval.batchResults[probeIdx0].get();
            bool ok        = eval.probeRes0.has_value();
            double psize   = ok ? static_cast<double>(
                                        eval.probeRes0->compressedSize)
                                : std::numeric_limits<double>::infinity();
            if (probeIdx1 != probeIdx0) {
                eval.probeRes1 = eval.batchResults[probeIdx1].get();
                if (!eval.probeRes1.has_value()) {
                    ok = false;
                } else if (ok) {
                    psize += static_cast<double>(
                            eval.probeRes1->compressedSize);
                }
            }
            eval.probeOk = ok;
            if (ok) {
                psize *= transformerSizePenalty(eval.gene.get());
                eval.probeSize    = psize;
                minProbeThisBatch = std::min(minProbeThisBatch, psize);
            }
        }
        bestProbeSoFar_ = minProbeThisBatch;
        for (auto& eval : evals) {
            if (eval.cachedFitness.has_value()) {
                continue;
            }
            if (!eval.probeOk
                || (std::isfinite(bestProbeSoFar_)
                    && eval.probeSize > bestProbeSoFar_ * probeMargin)) {
                eval.pruned = true;
                continue;
            }
            for (size_t b = 0; b < numBatches; ++b) {
                if (b == probeIdx0 || b == probeIdx1) {
                    continue;
                }
                eval.batchResults[b] = threadPool_->run(
                        [c = eval.compressor, batch = inputBatches_[b]] {
                            return benchmarkOnThreadContexts(*c, batch);
                        });
            }
        }
    }

    std::vector<std::vector<float>> results;
    results.reserve(genes.size());
    for (auto& eval : evals) {
        std::vector<float> result;
        const uint64_t h = eval.gene.get().hash();
        if (eval.cachedFitness.has_value()) {
            result = std::move(*eval.cachedFitness);
        } else if (eval.pruned) {
            if (!eval.probeOk || probeInputBytes <= 0.0) {
                result = std::vector<float>(
                        3, std::numeric_limits<float>::infinity());
            } else {
                const float est = static_cast<float>(
                        eval.probeSize * (totalAllBytes / probeInputBytes)
                        * 2.0);
                result = { est, 1e9f, 1e9f };
            }
        } else {
            poly::optional<ACECompressionResult> total = ACECompressionResult{};
            std::vector<float> batchSizes(numBatches, 0.0f);
            for (size_t b = 0; b < numBatches; ++b) {
                poly::optional<ACECompressionResult> batchResult;
                if (probeEnabled && b == probeIdx0) {
                    batchResult = std::move(eval.probeRes0);
                } else if (probeEnabled && b == probeIdx1) {
                    batchResult = std::move(eval.probeRes1);
                } else {
                    batchResult = eval.batchResults[b].get();
                }
                if (!batchResult.has_value()) {
                    total = poly::nullopt;
                } else {
                    batchSizes[b] = static_cast<float>(
                                            batchResult->compressedSize)
                            * static_cast<float>(
                                    transformerSizePenalty(eval.gene.get()));
                    if (total.has_value()) {
                        *total += *batchResult;
                    }
                }
            }
            if (total.has_value()) {
                updateBatchArchive(eval.gene.get(), batchSizes);
                perBatchSizes_[h] = std::move(batchSizes);
                if (numBatches >= 4) {
                    double psize = static_cast<double>(
                            perBatchSizes_[h][probeIdx0]);
                    if (probeIdx1 != probeIdx0) {
                        psize += static_cast<double>(
                                perBatchSizes_[h][probeIdx1]);
                    }
                    bestProbeSoFar_ = std::min(bestProbeSoFar_, psize);
                }
            }
            result = fitnessFromResult(eval.gene.get(), total);
        }
        if (!eval.pruned) {
            auto [cache, inserted] =
                    cachedFitness_.emplace(h, std::make_pair(1, result));
            if (!inserted) {
                const float n = static_cast<float>(cache->second.first + 1);
                cache->second.second[0] =
                        std::min(cache->second.second[0], result[0]);
                for (size_t j = 1; j < result.size(); ++j) {
                    cache->second.second[j] +=
                            (result[j] - cache->second.second[j]) / n;
                }
                ++cache->second.first;
            }
        }
        auto opIt = pendingOps_.find(h);
        if (opIt != pendingOps_.end()) {
            const auto& info = opIt->second;
            double reward    = 0.0;
            if (std::isfinite(result[0]) && std::isfinite(info.parentFitness0)
                && info.parentFitness0 > 0.0f
                && result[0] < info.parentFitness0) {
                const double rel =
                        static_cast<double>(info.parentFitness0 - result[0])
                        / static_cast<double>(info.parentFitness0);
                reward = std::min(1.0, 0.25 + rel * 20.0);
            }
            opStats_[info.op].pulls += 1.0;
            opStats_[info.op].rewardSum += reward;
            pendingOps_.erase(opIt);
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

size_t AutomatedCompressorExplorer::selectParentIndex()
{
    static const int lexEnv = [] {
        const char* p = std::getenv("ACE_LEXICASE");
        if (!p) {
            return -1;
        }
        return std::atoi(p) != 0 ? 1 : 0;
    }();
    const bool useLexicase = inputBatches_.size() >= 2
            && (lexEnv == 1 || (lexEnv != 0 && inputBatches_.size() >= 4
                                && inputs_.size() >= 8));
    if (useLexicase) {
        std::bernoulli_distribution pickLex(0.5);
        if (pickLex(rng())) {
            const size_t B = inputBatches_.size();
            std::vector<size_t> pool;
            pool.reserve(population().size());
            for (size_t i = 0; i < population().size(); ++i) {
                if (!std::isfinite(fitness()[i][0])) {
                    continue;
                }
                auto it = perBatchSizes_.find(population()[i].hash());
                if (it != perBatchSizes_.end() && it->second.size() == B) {
                    pool.push_back(i);
                }
            }
            if (!pool.empty()) {
                std::vector<size_t> batchOrder(B);
                for (size_t b = 0; b < B; ++b) {
                    batchOrder[b] = b;
                }
                std::shuffle(batchOrder.begin(), batchOrder.end(), rng());
                for (size_t b : batchOrder) {
                    float minSz = std::numeric_limits<float>::infinity();
                    for (size_t idx : pool) {
                        minSz = std::min(
                                minSz,
                                perBatchSizes_[population()[idx].hash()][b]);
                    }
                    const float thresh = minSz * 1.02f + 1.0f;
                    std::vector<size_t> nextPool;
                    nextPool.reserve(pool.size());
                    for (size_t idx : pool) {
                        if (perBatchSizes_[population()[idx].hash()][b]
                            <= thresh) {
                            nextPool.push_back(idx);
                        }
                    }
                    pool = std::move(nextPool);
                    if (pool.size() <= 1) {
                        break;
                    }
                }
                if (pool.size() > 1) {
                    std::uniform_int_distribution<size_t> tieDist(
                            0, pool.size() - 1);
                    return pool[tieDist(rng())];
                }
                if (!pool.empty()) {
                    return pool[0];
                }
            }
        }
    }
    return selector().select(rank(), crowdingDistance());
}

const ACECompressor& AutomatedCompressorExplorer::selectParent()
{
    return population()[selectParentIndex()];
}

size_t AutomatedCompressorExplorer::selectBanditOp()
{
    double totalPulls = 0.0;
    for (size_t op = 0; op < kNumBanditOps; ++op) {
        if (opStats_[op].pulls < 1.0) {
            return op;
        }
        totalPulls += opStats_[op].pulls;
    }
    // Small epsilon-greedy floor ensures every operator keeps getting sampled.
    std::bernoulli_distribution explore(0.10);
    if (explore(rng())) {
        std::uniform_int_distribution<size_t> opDist(0, kNumBanditOps - 1);
        return opDist(rng());
    }
    const double logTotal = std::log(totalPulls + 1.0);
    size_t bestOp         = 0;
    double bestScore      = -1.0;
    for (size_t op = 0; op < kNumBanditOps; ++op) {
        const double pulls = opStats_[op].pulls;
        const double mean  = opStats_[op].rewardSum / pulls;
        const double ucb   = mean + 0.6 * std::sqrt(logTotal / pulls);
        if (ucb > bestScore) {
            bestScore = ucb;
            bestOp    = op;
        }
    }
    return bestOp;
}

ACECompressor AutomatedCompressorExplorer::graftMotif(
        const ACECompressor& parent)
{
    if (motifs_.empty()) {
        return mutate(parent);
    }
    struct Target {
        const ACECompressor* ptr;
        Type type;
    };
    std::vector<Target> targets;
    parent.forEachComponent(inputType(), [&](const ACECompressor& c, Type t) {
        for (const auto& m : motifs_) {
            if (m.inputType == t && m.compressor.hash() != c.hash()) {
                targets.push_back(Target{ &c, t });
                break;
            }
        }
    });
    if (targets.empty()) {
        return mutate(parent);
    }
    std::uniform_int_distribution<size_t> tDist(0, targets.size() - 1);
    const auto target = targets[tDist(rng())];
    std::vector<const ACECompressor*> matching;
    for (const auto& m : motifs_) {
        if (m.inputType == target.type
            && m.compressor.hash() != target.ptr->hash()) {
            matching.push_back(&m.compressor);
        }
    }
    std::uniform_int_distribution<size_t> mDist(0, matching.size() - 1);
    const ACECompressor chosen = *matching[mDist(rng())];
    return parent.replace(
            inputType(),
            [&](const ACECompressor& c,
                Type,
                size_t) -> poly::optional<ACECompressor> {
                if (&c == target.ptr) {
                    return chosen;
                }
                return poly::nullopt;
            });
}

void AutomatedCompressorExplorer::mineMotifs()
{
    if (population().empty()) {
        return;
    }
    std::vector<size_t> order(population().size());
    for (size_t i = 0; i < order.size(); ++i) {
        order[i] = i;
    }
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return fitness()[a][0] < fitness()[b][0];
    });
    const float bestF0 = fitness()[order[0]][0];
    if (!std::isfinite(bestF0) || bestF0 <= 0.0f) {
        return;
    }
    const size_t topLimit =
            std::min(order.size(), std::max<size_t>(4, order.size() / 4));
    std::unordered_set<size_t> selected;
    for (size_t k = 0; k < topLimit; ++k) {
        selected.insert(order[k]);
    }
    for (size_t i = 0; i < population().size(); ++i) {
        if (rank()[i] == 0 && std::isfinite(fitness()[i][0])) {
            selected.insert(i);
        }
    }
    struct MotifAccum {
        Type inputType;
        ACECompressor compressor;
        double score{ 0.0 };
    };
    std::unordered_map<uint64_t, MotifAccum> table;
    for (size_t idx : selected) {
        const double weight = static_cast<double>(bestF0)
                / static_cast<double>(std::max(1.0f, fitness()[idx][0]));
        population()[idx].forEachComponent(
                inputType(), [&](const ACECompressor& sub, Type t) {
                    if (!sub.isNode()) {
                        return;
                    }
                    const size_t nc = sub.numComponents();
                    if (nc < 2 || nc > 12) {
                        return;
                    }
                    const uint64_t h = sub.hash();
                    auto [it, ins]   = table.emplace(
                            h, MotifAccum{ t, sub, weight });
                    if (!ins) {
                        it->second.score += weight;
                    }
                });
    }
    std::vector<MotifAccum> ranked;
    ranked.reserve(table.size());
    for (auto& kv : table) {
        ranked.push_back(std::move(kv.second));
    }
    std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) {
        return a.score > b.score;
    });
    constexpr size_t kMaxMotifs = 16;
    motifs_.clear();
    for (size_t i = 0; i < std::min(kMaxMotifs, ranked.size()); ++i) {
        motifs_.push_back(
                MinedMotif{ ranked[i].inputType,
                            std::move(ranked[i].compressor) });
    }
}

std::unordered_set<ACECompressor> AutomatedCompressorExplorer::reproduce(
        size_t numChildren)
{
    static const bool useBandit = [] {
        const char* p = std::getenv("ACE_BANDIT");
        return !p || std::atoi(p) != 0;
    }();
    static const size_t numPolish = [] {
        const char* p = std::getenv("ACE_POLISH");
        return p ? (size_t)std::strtoul(p, nullptr, 10) : 0;
    }();

    std::unordered_set<ACECompressor> children;
    if (!useBandit || generation() == 0) {
        children = Base::reproduce(numChildren);
    } else {
        size_t maxIterations = 2 * numChildren;
        while (children.size() < numChildren && maxIterations-- > 0) {
            const size_t op    = selectBanditOp();
            const size_t p1Idx = selectParentIndex();
            const float p1Fit  = fitness()[p1Idx][0];
            ACECompressor child = buildStoreCompressor();
            float parentFit0    = p1Fit;
            if (op == 0) {
                const size_t p2Idx = selectParentIndex();
                parentFit0         = std::min(p1Fit, fitness()[p2Idx][0]);
                child = crossover(population()[p1Idx], population()[p2Idx]);
                std::bernoulli_distribution mutDist(mutationProbability());
                if (mutDist(rng())) {
                    child = mutate(child);
                }
            } else if (op == 1) {
                child = mutate(population()[p1Idx]);
            } else if (op == 2) {
                child = graftMotif(population()[p1Idx]);
            } else {
                child = buildRandomCompressor(
                        rng(), inputType(), formatVersion_, kDefaultMaxDepth);
            }
            if (populationSet().count(child) == 0) {
                pendingOps_[child.hash()] = PendingChildInfo{ op, parentFit0 };
                children.insert(std::move(child));
            }
        }
    }

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

void AutomatedCompressorExplorer::updateBatchArchive(
        const ACECompressor& gene,
        const std::vector<float>& batchSizes)
{
    if (batchArchive_.size() < batchSizes.size()) {
        batchArchive_.resize(batchSizes.size());
    }
    const uint64_t h = gene.hash();
    auto bySize      = [](const auto& a, const auto& b) {
        return a.first < b.first;
    };
    for (size_t b = 0; b < batchSizes.size(); ++b) {
        const float sz = batchSizes[b];
        if (!std::isfinite(sz)) {
            continue;
        }
        auto& slot   = batchArchive_[b];
        bool updated = false;
        for (auto& entry : slot) {
            if (entry.second.hash() == h) {
                if (sz < entry.first) {
                    entry.first = sz;
                    std::sort(slot.begin(), slot.end(), bySize);
                }
                updated = true;
                break;
            }
        }
        if (updated) {
            continue;
        }
        if (slot.size() < kArchivePerBatch) {
            slot.emplace_back(sz, gene);
            std::sort(slot.begin(), slot.end(), bySize);
        } else if (sz < slot.back().first) {
            slot.back() = std::make_pair(sz, gene);
            std::sort(slot.begin(), slot.end(), bySize);
        }
    }
}

std::vector<ACECompressor> AutomatedCompressorExplorer::archivedWinners() const
{
    std::vector<ACECompressor> winners;
    std::unordered_set<uint64_t> seen;
    for (size_t rankIdx = 0; rankIdx < kArchivePerBatch; ++rankIdx) {
        for (const auto& slot : batchArchive_) {
            if (rankIdx < slot.size()) {
                const auto& comp = slot[rankIdx].second;
                if (seen.insert(comp.hash()).second) {
                    winners.push_back(comp);
                }
            }
        }
    }
    return winners;
}

void AutomatedCompressorExplorer::step()
{
    Base::step();
    mineMotifs();
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
