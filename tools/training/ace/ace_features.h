// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <unordered_set>
#include <vector>

#include "openzl/cpp/codecs/Bitpack.hpp"
#include "openzl/cpp/codecs/Compress.hpp"
#include "openzl/cpp/codecs/Conversion.hpp"
#include "openzl/cpp/codecs/Delta.hpp"
#include "openzl/cpp/codecs/Entropy.hpp"
#include "openzl/cpp/codecs/Hilbert3.hpp"
#include "openzl/cpp/codecs/IntMultSplit.hpp"
#include "openzl/cpp/codecs/LookbackDelta.hpp"
#include "openzl/cpp/codecs/Morton3.hpp"
#include "openzl/cpp/codecs/PcoBinSplit.hpp"
#include "openzl/cpp/codecs/SparseNum.hpp"
#include "openzl/cpp/codecs/Store.hpp"
#include "openzl/cpp/codecs/Tokenize.hpp"
#include "openzl/cpp/codecs/Transpose.hpp"
#include "openzl/cpp/codecs/Zigzag.hpp"
#include "openzl/openzl.hpp"
#include "tools/training/ace/ace_compressor.h"
#include "tools/training/ace/ace_compressors.h"

namespace openzl {
namespace training {

struct ACEInputFeatures {
    uint32_t eltWidth{ 0 };
    double sortedFraction{ 0.0 };
    double zeroDiffFraction{ 0.0 };
    bool lowCardinality{ false };
    uint64_t commonGcd{ 1 };
    size_t sampledElements{ 0 };
};

namespace detail {

inline uint64_t readUnsignedSample(const uint8_t* p, size_t width)
{
    switch (width) {
        case 1: {
            uint8_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return v;
        }
        case 2: {
            uint16_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return v;
        }
        case 4: {
            uint32_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return v;
        }
        case 8: {
            uint64_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return v;
        }
        default:
            return 0;
    }
}

inline int64_t readSignedSample(const uint8_t* p, size_t width)
{
    switch (width) {
        case 1: {
            int8_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return v;
        }
        case 2: {
            int16_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return v;
        }
        case 4: {
            int32_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return v;
        }
        case 8: {
            int64_t v = 0;
            std::memcpy(&v, p, sizeof(v));
            return v;
        }
        default:
            return 0;
    }
}

inline bool compressorNodesValidForInput(
        const ACECompressor& comp,
        uint32_t inputWidth,
        uint32_t formatVersion)
{
    if (!comp.isNode()) {
        return true;
    }
    const auto& nc = comp.asNode();
    if (nc.node.minFormatVersion != 0
        && nc.node.minFormatVersion > formatVersion) {
        return false;
    }
    if (!nodeAcceptsWidth(nc.node, inputWidth)) {
        return false;
    }
    for (size_t i = 0; i < nc.successors.size(); ++i) {
        const uint32_t outWidth =
                inferNodeOutputWidth(nc.node, i, inputWidth);
        if (!compressorNodesValidForInput(
                    *nc.successors[i], outWidth, formatVersion)) {
            return false;
        }
    }
    return true;
}

inline bool isSeedValid(
        const ACECompressor& seed,
        Type inputType,
        uint32_t eltWidth,
        uint32_t formatVersion)
{
    if (!seed.acceptsInputType(inputType)) {
        return false;
    }
    if (!compressorNodesValidForInput(seed, eltWidth, formatVersion)) {
        return false;
    }
    try {
        (void)seed.buildBenchmarkCompressor(formatVersion, 0);
        return true;
    } catch (...) {
        return false;
    }
}

} // namespace detail

/// Extracts cheap structural features by sampling up to ~64 KiB across
/// representative inputs (0, n/2, n-1).
inline ACEInputFeatures extractACEInputFeatures(
        poly::span<const Input> inputs,
        Type inputType)
{
    ACEInputFeatures features{};
    if (inputs.empty()) {
        return features;
    }

    if (inputType == Type::Numeric || inputType == Type::Struct) {
        size_t commonWidth = 0;
        bool hasNonEmpty   = false;
        bool sameWidth     = true;
        for (const auto& in : inputs) {
            if (in.numElts() == 0) {
                continue;
            }
            if (!hasNonEmpty) {
                commonWidth = in.eltWidth();
                hasNonEmpty = true;
            } else if (in.eltWidth() != commonWidth) {
                sameWidth = false;
                break;
            }
        }
        if (!hasNonEmpty) {
            commonWidth = inputs[0].eltWidth();
        }
        if (sameWidth && commonWidth > 0) {
            features.eltWidth = static_cast<uint32_t>(commonWidth);
        }
    }

    const size_t n = inputs.size();
    std::vector<size_t> repIndices;
    repIndices.reserve(3);
    for (size_t idx : { size_t(0), n / 2, n - 1 }) {
        if (std::find(repIndices.begin(), repIndices.end(), idx)
            == repIndices.end()) {
            repIndices.push_back(idx);
        }
    }
    bool anySampleNonEmpty = false;
    for (size_t idx : repIndices) {
        if (inputs[idx].contentSize() > 0) {
            anySampleNonEmpty = true;
            break;
        }
    }
    if (!anySampleNonEmpty) {
        repIndices.clear();
        for (size_t i = 0; i < n && repIndices.size() < 3; ++i) {
            if (inputs[i].contentSize() > 0) {
                repIndices.push_back(i);
            }
        }
    }
    if (repIndices.empty()) {
        return features;
    }

    constexpr size_t kMaxSampleBytes = 64 * 1024;
    const size_t perInputBudget =
            std::max<size_t>(64, kMaxSampleBytes / repIndices.size());

    if (inputType == Type::Numeric
        && (features.eltWidth == 1 || features.eltWidth == 2
            || features.eltWidth == 4 || features.eltWidth == 8)) {
        const size_t w         = features.eltWidth;
        size_t totalElts       = 0;
        size_t totalPairs      = 0;
        size_t unsignedNonDec  = 0;
        size_t signedNonDec    = 0;
        size_t equalPairs      = 0;
        uint64_t diffGcdU      = 0;
        uint64_t diffGcdS      = 0;
        std::vector<uint64_t> distinct;
        distinct.reserve(65);

        for (size_t idx : repIndices) {
            const auto& in = inputs[idx];
            if (in.eltWidth() != w || in.numElts() == 0
                || in.ptr() == nullptr) {
                continue;
            }
            const auto* bytes = static_cast<const uint8_t*>(in.ptr());
            const size_t take = std::min(
                    in.numElts(), std::max<size_t>(1, perInputBudget / w));
            uint64_t prevU = 0;
            int64_t prevS  = 0;
            for (size_t i = 0; i < take; ++i) {
                const uint64_t u =
                        detail::readUnsignedSample(bytes + i * w, w);
                const int64_t s = detail::readSignedSample(bytes + i * w, w);
                ++totalElts;
                if (distinct.size() <= 64
                    && std::find(distinct.begin(), distinct.end(), u)
                            == distinct.end()) {
                    distinct.push_back(u);
                }
                if (i > 0) {
                    ++totalPairs;
                    if (u == prevU) {
                        ++equalPairs;
                    }
                    if (u >= prevU) {
                        ++unsignedNonDec;
                    }
                    if (s >= prevS) {
                        ++signedNonDec;
                    }
                    const uint64_t du =
                            (u >= prevU) ? (u - prevU) : (prevU - u);
                    const uint64_t ds = (s >= prevS)
                            ? (static_cast<uint64_t>(s)
                               - static_cast<uint64_t>(prevS))
                            : (static_cast<uint64_t>(prevS)
                               - static_cast<uint64_t>(s));
                    diffGcdU = std::gcd(diffGcdU, du);
                    diffGcdS = std::gcd(diffGcdS, ds);
                }
                prevU = u;
                prevS = s;
            }
        }

        features.sampledElements = totalElts;
        if (totalPairs > 0) {
            features.sortedFraction =
                    static_cast<double>(std::max(unsignedNonDec, signedNonDec))
                    / static_cast<double>(totalPairs);
            features.zeroDiffFraction =
                    static_cast<double>(equalPairs)
                    / static_cast<double>(totalPairs);
        }
        if (totalElts > 0) {
            features.lowCardinality = distinct.size() <= 64
                    && (totalElts > 64 || distinct.size() < totalElts);
        }
        const uint64_t bestDiffGcd = std::max(diffGcdU, diffGcdS);
        if (distinct.size() >= 2 && bestDiffGcd > 1) {
            features.commonGcd = bestDiffGcd;
        }
    } else if (inputType == Type::Struct && features.eltWidth > 0) {
        const size_t w    = features.eltWidth;
        size_t totalElts  = 0;
        size_t totalPairs = 0;
        size_t equalPairs = 0;
        for (size_t idx : repIndices) {
            const auto& in = inputs[idx];
            if (in.eltWidth() != w || in.numElts() == 0
                || in.ptr() == nullptr) {
                continue;
            }
            const auto* bytes = static_cast<const uint8_t*>(in.ptr());
            const size_t take = std::min(
                    in.numElts(), std::max<size_t>(1, perInputBudget / w));
            totalElts += take;
            for (size_t i = 1; i < take; ++i) {
                ++totalPairs;
                if (std::memcmp(bytes + i * w, bytes + (i - 1) * w, w) == 0) {
                    ++equalPairs;
                }
            }
        }
        features.sampledElements = totalElts;
        if (totalPairs > 0) {
            features.zeroDiffFraction =
                    static_cast<double>(equalPairs)
                    / static_cast<double>(totalPairs);
        }
    }

    return features;
}

/// Builds targeted ACECompressor seed pipelines guided by @p features, filtered
/// to seeds compatible with (@p inputType, @p features.eltWidth,
/// @p formatVersion).
inline std::vector<ACECompressor> buildFeatureGuidedSeeds(
        const ACEInputFeatures& features,
        Type inputType,
        uint32_t formatVersion)
{
    std::vector<ACECompressor> candidates;
    const ACECompressor fse(ACEGraph(graphs::Fse{}));
    const ACECompressor store(ACEGraph(graphs::Store{}));
    const ACECompressor bitpack(ACEGraph(graphs::Bitpack{}));
    const ACECompressor compressGeneric(ACEGraph(graphs::Compress{}));
    const ACECompressor pcoUnsigned(
            ACENode(nodes::PcoBinSplit{
                    nodes::PcoBinSplit::LatentMode::Unsigned }),
            std::vector<ACECompressor>{ fse, store });
    const ACECompressor pcoSigned(
            ACENode(nodes::PcoBinSplit{
                    nodes::PcoBinSplit::LatentMode::Signed }),
            std::vector<ACECompressor>{ fse, store });

    if (inputType == Type::Numeric) {
        if (features.sortedFraction >= 0.7) {
            candidates.emplace_back(
                    ACENode(nodes::DeltaInt{}),
                    std::vector<ACECompressor>{ pcoUnsigned });
            candidates.emplace_back(
                    ACENode(nodes::DeltaInt{}),
                    std::vector<ACECompressor>{ ACECompressor(
                            ACENode(nodes::Zigzag{}),
                            std::vector<ACECompressor>{ pcoUnsigned }) });
            candidates.emplace_back(
                    ACENode(nodes::LookbackDelta{}),
                    std::vector<ACECompressor>{ pcoSigned, fse });
        }
        if (features.commonGcd > 1) {
            candidates.emplace_back(
                    ACENode(nodes::IntMultSplit{
                            nodes::IntMultSplit::Mode::Unsigned }),
                    std::vector<ACECompressor>{ pcoSigned, compressGeneric });
            candidates.emplace_back(
                    ACENode(nodes::IntMultSplit{
                            nodes::IntMultSplit::Mode::Signed }),
                    std::vector<ACECompressor>{ pcoSigned, compressGeneric });
            candidates.emplace_back(
                    ACENode(nodes::IntMultSplit{
                            nodes::IntMultSplit::Mode::Unsigned }),
                    std::vector<ACECompressor>{
                            ACECompressor(
                                    ACENode(nodes::DeltaInt{}),
                                    std::vector<ACECompressor>{ pcoSigned }),
                            bitpack });
        }
        if (features.lowCardinality) {
            candidates.emplace_back(
                    ACENode(nodes::TokenizeNumeric{ true }),
                    std::vector<ACECompressor>{
                            ACECompressor(
                                    ACENode(nodes::DeltaInt{}),
                                    std::vector<ACECompressor>{ bitpack }),
                            fse });
            candidates.emplace_back(
                    ACENode(nodes::TokenizeNumeric{ true }),
                    std::vector<ACECompressor>{ pcoUnsigned, bitpack });
        }
        if (features.zeroDiffFraction >= 0.5 && formatVersion >= 26) {
            if (formatVersion >= 27) {
                candidates.emplace_back(
                        ACENode(nodes::SparseNumAuto{}),
                        std::vector<ACECompressor>{ pcoUnsigned, fse });
            } else {
                candidates.emplace_back(
                        ACENode(nodes::SparseNumAuto{}),
                        std::vector<ACECompressor>{ bitpack, fse });
            }
        }
    } else if (inputType == Type::Struct) {
        if (features.eltWidth == 6) {
            candidates.emplace_back(
                    ACENode(nodes::Hilbert3U16{}),
                    std::vector<ACECompressor>{ pcoUnsigned });
            candidates.emplace_back(
                    ACENode(nodes::Morton3U16{}),
                    std::vector<ACECompressor>{ pcoUnsigned });
            candidates.emplace_back(
                    ACENode(nodes::TransposeSplit{}),
                    std::vector<ACECompressor>{ fse });
            candidates.emplace_back(
                    ACENode(nodes::TransposeSplit{}),
                    std::vector<ACECompressor>{ ACECompressor(
                            ACENode(nodes::ConvertSerialToNum8{}),
                            std::vector<ACECompressor>{ ACECompressor(
                                    ACENode(nodes::DeltaInt{}),
                                    std::vector<ACECompressor>{ fse }) }) });
        } else if (features.eltWidth == 12) {
            candidates.emplace_back(
                    ACENode(nodes::Hilbert3I32{}),
                    std::vector<ACECompressor>{ pcoUnsigned });
            candidates.emplace_back(
                    ACENode(nodes::Morton3I32{}),
                    std::vector<ACECompressor>{ pcoUnsigned });
            candidates.emplace_back(
                    ACENode(nodes::TransposeSplit{}),
                    std::vector<ACECompressor>{ fse });
        }
    }

    std::vector<ACECompressor> valid;
    valid.reserve(candidates.size());
    std::unordered_set<uint64_t> seen;
    seen.reserve(candidates.size());
    for (auto& cand : candidates) {
        if (!detail::isSeedValid(
                    cand, inputType, features.eltWidth, formatVersion)) {
            continue;
        }
        if (seen.insert(cand.hash()).second) {
            valid.push_back(std::move(cand));
        }
    }
    return valid;
}

inline std::vector<ACECompressor> buildFeatureGuidedSeeds(
        poly::span<const Input> inputs,
        Type inputType,
        uint32_t formatVersion)
{
    return buildFeatureGuidedSeeds(
            extractACEInputFeatures(inputs, inputType),
            inputType,
            formatVersion);
}

} // namespace training
} // namespace openzl
