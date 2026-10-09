// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "cardinality.h"
#include "numeric_stats.h"

#include "openzl/shared/estimate.h"
#include "openzl/shared/mem.h"

/*
 * These hashes intentionally duplicate the shared estimator hashes used when
 * the Transformer models were trained. They are frozen model feature
 * definitions, not aliases: later shared-estimator changes must not propagate
 * here without regenerating features, retraining, and evaluation.
 *
 * Each estimate feeds hashes into `ZL_HashedCardinality`, whose result
 * depends only on the set of hashes. Loops therefore skip an element whose
 * hashed bytes equal the previous element's; this never changes the set.
 */
#define XXH_INLINE_ALL
#define XXH_STATIC_LINKING_ONLY
#include "openzl/shared/xxhash.h"

#include "openzl/common/assertion.h"

static uint64_t linear_hash_u64(uint64_t value)
{
    uint64_t const hash = value * 0x9E3779B185EBCA87ULL;
    return hash ^ (hash << 47);
}

static uint64_t xxh3_hash_u64(uint64_t value)
{
    uint8_t little_endian[sizeof(value)];
    ZL_writeLE64(little_endian, value);
    return XXH3_64bits(little_endian, sizeof(little_endian));
}

static uint64_t xxh3_hash_pair_u64(uint64_t first, uint64_t second)
{
    uint8_t pair[2 * sizeof(uint64_t)];
    ZL_writeLE64(pair, first);
    ZL_writeLE64(pair + sizeof(uint64_t), second);
    return XXH3_64bits(pair, sizeof(pair));
}

/* Bump the hash of every value whose bit is set, in increasing order. */
static void bump_bitmap_values(
        ZL_HashedCardinality* sketch,
        const uint64_t* words,
        size_t word_count)
{
    for (size_t word = 0; word < word_count; ++word) {
        uint64_t bits = words[word];
        while (bits != 0) {
            uint64_t const value =
                    (uint64_t)(word * 64 + (size_t)ZL_ctz64(bits));
            bits &= bits - 1;
            ZL_HashedCardinality_bump(
                    sketch,
                    sketch->useHll ? xxh3_hash_u64(value)
                                   : linear_hash_u64(value));
        }
    }
}

static uint64_t estimate_bitmap_cardinality(
        const uint64_t* words,
        size_t word_count,
        size_t cardinality_upper_bound)
{
    if (cardinality_upper_bound == 0)
        return 0;
    ZL_HashedCardinality sketch;
    ZL_HashedCardinality_init(&sketch, cardinality_upper_bound);
    bump_bitmap_values(&sketch, words, word_count);
    return ZL_HashedCardinality_estimate(&sketch, cardinality_upper_bound);
}

uint64_t TRS_estimate_cardinality_u64(const uint64_t* data, size_t n_elements)
{
    if (n_elements == 0)
        return 0;
    ZL_HashedCardinality sketch;
    ZL_HashedCardinality_init(&sketch, n_elements);
    uint64_t prev = ~data[0];
    for (size_t i = 0; i < n_elements; ++i) {
        uint64_t const value = data[i];
        if (value == prev)
            continue;
        prev = value;
        ZL_HashedCardinality_bump(
                &sketch,
                sketch.useHll ? xxh3_hash_u64(value) : linear_hash_u64(value));
    }
    return ZL_HashedCardinality_estimate(&sketch, n_elements);
}

uint64_t TRS_estimate_cardinality_u32(const uint32_t* data, size_t n_elements)
{
    if (n_elements == 0)
        return 0;
    ZL_HashedCardinality sketch;
    ZL_HashedCardinality_init(&sketch, n_elements);
    uint32_t prev = ~data[0];
    for (size_t i = 0; i < n_elements; ++i) {
        uint32_t const value = data[i];
        if (value == prev)
            continue;
        prev = value;
        ZL_HashedCardinality_bump(
                &sketch,
                sketch.useHll ? xxh3_hash_u64(value) : linear_hash_u64(value));
    }
    return ZL_HashedCardinality_estimate(&sketch, n_elements);
}

/* Pair and window features use XXH3 for both estimator paths, matching the
 * trained feature contract. */
uint64_t TRS_estimate_pair_cardinality_u64(
        const uint64_t* data,
        size_t n_elements)
{
    if (n_elements < 2)
        return 0;
    size_t const n_pairs = n_elements - 1;
    ZL_HashedCardinality sketch;
    ZL_HashedCardinality_init(&sketch, n_pairs);
    ZL_HashedCardinality_bump(&sketch, xxh3_hash_pair_u64(data[0], data[1]));
    for (size_t i = 1; i < n_pairs; ++i) {
        /* Pair i equals pair i - 1 inside a run of three equal values. */
        if (data[i + 1] == data[i] && data[i] == data[i - 1])
            continue;
        ZL_HashedCardinality_bump(
                &sketch, xxh3_hash_pair_u64(data[i], data[i + 1]));
    }
    return ZL_HashedCardinality_estimate(&sketch, n_pairs);
}

uint64_t TRS_estimate_pair_cardinality_u32(
        const uint32_t* data,
        size_t n_elements)
{
    if (n_elements < 2)
        return 0;
    size_t const n_pairs = n_elements - 1;
    ZL_HashedCardinality sketch;
    ZL_HashedCardinality_init(&sketch, n_pairs);
    ZL_HashedCardinality_bump(&sketch, xxh3_hash_pair_u64(data[0], data[1]));
    for (size_t i = 1; i < n_pairs; ++i) {
        if (data[i + 1] == data[i] && data[i] == data[i - 1])
            continue;
        ZL_HashedCardinality_bump(
                &sketch, xxh3_hash_pair_u64(data[i], data[i + 1]));
    }
    return ZL_HashedCardinality_estimate(&sketch, n_pairs);
}

uint64_t TRS_estimate_cardinality_u8_bitmap(
        const uint64_t seen_values[TRS_CARDINALITY_U8_BITMAP_WORDS],
        size_t n_elements)
{
    return estimate_bitmap_cardinality(
            seen_values, TRS_CARDINALITY_U8_BITMAP_WORDS, n_elements);
}

uint64_t TRS_estimate_cardinality_u16_bitmap(
        const uint64_t seen_values[TRS_CARDINALITY_U16_BITMAP_WORDS],
        size_t n_elements)
{
    return estimate_bitmap_cardinality(
            seen_values, TRS_CARDINALITY_U16_BITMAP_WORDS, n_elements);
}

uint64_t TRS_estimate_pair_cardinality_u8_bitmap(
        const uint64_t seen_pairs[TRS_CARDINALITY_U8_PAIR_BITMAP_WORDS],
        size_t n_elements)
{
    if (n_elements < 2)
        return 0;
    size_t const n_pairs = n_elements - 1;
    ZL_HashedCardinality sketch;
    ZL_HashedCardinality_init(&sketch, n_pairs);
    for (size_t word = 0; word < TRS_CARDINALITY_U8_PAIR_BITMAP_WORDS;
         ++word) {
        uint64_t bits = seen_pairs[word];
        while (bits != 0) {
            uint32_t const key = (uint32_t)(word * 64 + (size_t)ZL_ctz64(bits));
            bits &= bits - 1;
            ZL_HashedCardinality_bump(
                    &sketch, xxh3_hash_pair_u64(key >> 8, key & 0xFF));
        }
    }
    return ZL_HashedCardinality_estimate(&sketch, n_pairs);
}

uint64_t TRS_estimate_d8_cardinality(
        const uint8_t* data,
        size_t n_bytes,
        size_t elt_width)
{
    ZL_ASSERT(
            elt_width == 1 || elt_width == 2 || elt_width == 4
            || elt_width == 8);
    if (elt_width != 1 && elt_width != 2 && elt_width != 4 && elt_width != 8)
        return 0;
    ZL_ASSERT_EQ(n_bytes % elt_width, 0);
    if (n_bytes % elt_width != 0)
        return 0;
    if (n_bytes < 8)
        return 0;
    size_t const n_windows = n_bytes - 7;
    ZL_HashedCardinality sketch;
    ZL_HashedCardinality_init(&sketch, n_windows);
    if (ZL_isLittleEndian()) {
        uint64_t prev = ~ZL_read64(data);
        for (size_t i = 0; i < n_windows; ++i) {
            uint64_t const window = ZL_read64(data + i);
            if (window == prev)
                continue;
            prev = window;
            ZL_HashedCardinality_bump(&sketch, XXH3_64bits(data + i, 8));
        }
    } else {
        for (size_t i = 0; i < n_windows; ++i) {
            uint8_t window[8];
            for (size_t b = 0; b < sizeof(window); ++b) {
                window[b] = TRS_numeric_canonical_byte_from_big_endian(
                        data, i + b, elt_width);
            }
            ZL_HashedCardinality_bump(
                    &sketch, XXH3_64bits(window, sizeof(window)));
        }
    }
    return ZL_HashedCardinality_estimate(&sketch, n_windows);
}
