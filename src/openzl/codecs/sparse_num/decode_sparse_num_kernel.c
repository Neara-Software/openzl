// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "decode_sparse_num_kernel.h"

#include <assert.h>
#include <stdint.h>
#include <string.h>

#if defined(__AVX2__)
#    include <immintrin.h>
#elif defined(__aarch64__)
#    include <arm_neon.h>
#elif defined(__wasm_simd128__)
#    include <wasm_simd128.h>
#endif

/*
 * Distances are numeric stream elements, so the binding guarantees alignment
 * for their element width. Use typed loads instead of byte copies to keep the
 * kernel dependency-free and to match the numeric-stream host-endian contract.
 */
static uint32_t
ZL_sparseNumReadDistance(const void* src, size_t index, size_t width)
{
    const void* const ptr = (const uint8_t*)src + index * width;
    assert(ZL_sparseNumIsAlignedForWidth(ptr, width));
    switch (width) {
        case 1:
            return *(const uint8_t*)ptr;
        case 2:
            return *(const uint16_t*)ptr;
        case 4:
            return *(const uint32_t*)ptr;
        default:
            assert(false);
            return 0;
    }
}

/*
 * Unchecked sums of the distance stream. The caller guarantees that the sum
 * cannot overflow 64 bits (at most 2^32 distances below 2^32 each).
 */
static uint64_t ZL_sparseNumSumDistances8(const uint8_t* d, size_t n)
{
    uint64_t sum = 0;
    size_t i     = 0;
#if defined(__AVX2__)
    __m256i acc        = _mm256_setzero_si256();
    __m256i const zero = _mm256_setzero_si256();
    for (; i + 32 <= n; i += 32) {
        __m256i const v = _mm256_loadu_si256((const __m256i*)(d + i));
        acc             = _mm256_add_epi64(acc, _mm256_sad_epu8(v, zero));
    }
    __m128i const acc2 = _mm_add_epi64(
            _mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    sum = (uint64_t)_mm_cvtsi128_si64(acc2)
            + (uint64_t)_mm_extract_epi64(acc2, 1);
#elif defined(__aarch64__)
    uint64x2_t acc = vdupq_n_u64(0);
    for (; i + 16 <= n; i += 16) {
        acc = vpadalq_u32(acc, vpaddlq_u16(vpaddlq_u8(vld1q_u8(d + i))));
    }
    sum = vgetq_lane_u64(acc, 0) + vgetq_lane_u64(acc, 1);
#elif defined(__wasm_simd128__)
    v128_t acc = wasm_i64x2_const(0, 0);
    for (; i + 16 <= n; i += 16) {
        v128_t const s32 = wasm_u32x4_extadd_pairwise_u16x8(
                wasm_u16x8_extadd_pairwise_u8x16(wasm_v128_load(d + i)));
        acc = wasm_i64x2_add(
                acc,
                wasm_i64x2_add(
                        wasm_u64x2_extend_low_u32x4(s32),
                        wasm_u64x2_extend_high_u32x4(s32)));
    }
    sum = (uint64_t)wasm_i64x2_extract_lane(acc, 0)
            + (uint64_t)wasm_i64x2_extract_lane(acc, 1);
#endif
    for (; i < n; ++i) {
        sum += d[i];
    }
    return sum;
}

static uint64_t ZL_sparseNumSumDistances16(const uint16_t* d, size_t n)
{
    uint64_t sum = 0;
    size_t i     = 0;
#if defined(__AVX2__)
    __m256i acc        = _mm256_setzero_si256();
    __m256i const zero = _mm256_setzero_si256();
    for (; i + 16 <= n; i += 16) {
        __m256i const v   = _mm256_loadu_si256((const __m256i*)(d + i));
        __m256i const s32 = _mm256_add_epi32(
                _mm256_unpacklo_epi16(v, zero), _mm256_unpackhi_epi16(v, zero));
        acc = _mm256_add_epi64(
                acc,
                _mm256_add_epi64(
                        _mm256_unpacklo_epi32(s32, zero),
                        _mm256_unpackhi_epi32(s32, zero)));
    }
    __m128i const acc2 = _mm_add_epi64(
            _mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
    sum = (uint64_t)_mm_cvtsi128_si64(acc2)
            + (uint64_t)_mm_extract_epi64(acc2, 1);
#elif defined(__aarch64__)
    uint64x2_t acc = vdupq_n_u64(0);
    for (; i + 8 <= n; i += 8) {
        acc = vpadalq_u32(acc, vpaddlq_u16(vld1q_u16(d + i)));
    }
    sum = vgetq_lane_u64(acc, 0) + vgetq_lane_u64(acc, 1);
#elif defined(__wasm_simd128__)
    v128_t acc = wasm_i64x2_const(0, 0);
    for (; i + 8 <= n; i += 8) {
        v128_t const s32 =
                wasm_u32x4_extadd_pairwise_u16x8(wasm_v128_load(d + i));
        acc = wasm_i64x2_add(
                acc,
                wasm_i64x2_add(
                        wasm_u64x2_extend_low_u32x4(s32),
                        wasm_u64x2_extend_high_u32x4(s32)));
    }
    sum = (uint64_t)wasm_i64x2_extract_lane(acc, 0)
            + (uint64_t)wasm_i64x2_extract_lane(acc, 1);
#endif
    for (; i < n; ++i) {
        sum += d[i];
    }
    return sum;
}

static uint64_t ZL_sparseNumSumDistances32(const uint32_t* d, size_t n)
{
    uint64_t sum = 0;
    for (size_t i = 0; i < n; ++i) {
        sum += d[i];
    }
    return sum;
}

/*
 * Compute the reconstructed element count before allocation. The public error
 * value is SIZE_MAX, so this rejects both real overflow and the
 * otherwise-valid SIZE_MAX count, keeping the API a single-value return.
 *
 * At most 2^32 distances below 2^32 each cannot overflow a 64-bit sum, so the
 * common case sums without a per-element check and validates the total once.
 * The running count only grows, so checking the total is equivalent to
 * checking every step.
 */
size_t ZL_sparseNumDecodeOutputCount(
        const void* distances,
        size_t numDistances,
        size_t distanceWidth,
        size_t numLiterals)
{
    assert(ZL_sparseNumValidDistanceWidth(distanceWidth));

    if ((uint64_t)numDistances <= UINT32_MAX) {
        uint64_t sum;
        switch (distanceWidth) {
            case 1:
                sum = ZL_sparseNumSumDistances8(
                        (const uint8_t*)distances, numDistances);
                break;
            case 2:
                sum = ZL_sparseNumSumDistances16(
                        (const uint16_t*)distances, numDistances);
                break;
            default:
                sum = ZL_sparseNumSumDistances32(
                        (const uint32_t*)distances, numDistances);
                break;
        }
        if (sum
            >= (uint64_t)(ZL_SPARSE_NUM_DECODE_OUTPUT_COUNT_ERROR
                          - numLiterals)) {
            return ZL_SPARSE_NUM_DECODE_OUTPUT_COUNT_ERROR;
        }
        return numLiterals + (size_t)sum;
    }

    size_t count = numLiterals;
    for (size_t i = 0; i < numDistances; ++i) {
        uint32_t const distance =
                ZL_sparseNumReadDistance(distances, i, distanceWidth);
        if ((size_t)distance
            >= ZL_SPARSE_NUM_DECODE_OUTPUT_COUNT_ERROR - count) {
            return ZL_SPARSE_NUM_DECODE_OUTPUT_COUNT_ERROR;
        }
        count += (size_t)distance;
    }

    return count;
}

/*
 * Fill the whole destination with the dominant symbol. Dominant value 0 is the
 * sparse-zero mode and maps directly to memset(0).
 */
static void ZL_sparseNumDecodeFill(
        void* dst,
        size_t dstSize,
        uint64_t dominant,
        size_t valueWidth)
{
    if (dstSize == 0) {
        return;
    }
    assert(dst != NULL);
    if (dominant == 0 || valueWidth == 1) {
        memset(dst, (uint8_t)dominant, dstSize);
        return;
    }
    assert(ZL_sparseNumIsAlignedForWidth(dst, valueWidth));
    size_t const count = dstSize / valueWidth;
    switch (valueWidth) {
        case 2: {
            uint16_t* const out  = (uint16_t*)dst;
            uint16_t const value = (uint16_t)dominant;
            for (size_t i = 0; i < count; ++i) {
                out[i] = value;
            }
            return;
        }
        case 4: {
            uint32_t* const out  = (uint32_t*)dst;
            uint32_t const value = (uint32_t)dominant;
            for (size_t i = 0; i < count; ++i) {
                out[i] = value;
            }
            return;
        }
        case 8: {
            uint64_t* const out = (uint64_t*)dst;
            for (size_t i = 0; i < count; ++i) {
                out[i] = dominant;
            }
            return;
        }
        default:
            assert(false);
            return;
    }
}

/*
 * Write the literals into a destination that already holds the dominant symbol
 * everywhere. Each distance is the number of dominant symbols before the next
 * literal, so the only serial dependency is one add per literal. Returns the
 * number of elements covered, up to and including the last literal.
 */
#define ZL_SPARSE_NUM_DEFINE_SCATTER(DW, DT, VW, VT)                           \
    static size_t ZL_sparseNumScatterD##DW##V##VW(                             \
            VT* restrict dst,                                                  \
            const DT* restrict distances,                                      \
            const VT* restrict values,                                         \
            size_t numValues)                                                  \
    {                                                                          \
        size_t pos = 0;                                                        \
        for (size_t i = 0; i < numValues; ++i) {                               \
            pos += distances[i];                                               \
            dst[pos++] = values[i];                                            \
        }                                                                      \
        return pos;                                                            \
    }

#define ZL_SPARSE_NUM_DEFINE_SCATTERS(DW, DT)                                  \
    ZL_SPARSE_NUM_DEFINE_SCATTER(DW, DT, 1, uint8_t)                           \
    ZL_SPARSE_NUM_DEFINE_SCATTER(DW, DT, 2, uint16_t)                          \
    ZL_SPARSE_NUM_DEFINE_SCATTER(DW, DT, 4, uint32_t)                          \
    ZL_SPARSE_NUM_DEFINE_SCATTER(DW, DT, 8, uint64_t)                          \
    static size_t ZL_sparseNumScatterD##DW(                                    \
            void* dst,                                                         \
            const void* distances,                                             \
            const void* values,                                                \
            size_t numValues,                                                  \
            size_t valueWidth)                                                 \
    {                                                                          \
        const DT* const d = (const DT*)distances;                              \
        switch (valueWidth) {                                                  \
            case 1:                                                            \
                return ZL_sparseNumScatterD##DW##V1(                           \
                        (uint8_t*)dst, d, (const uint8_t*)values, numValues);  \
            case 2:                                                            \
                return ZL_sparseNumScatterD##DW##V2(                           \
                        (uint16_t*)dst,                                        \
                        d,                                                     \
                        (const uint16_t*)values,                               \
                        numValues);                                            \
            case 4:                                                            \
                return ZL_sparseNumScatterD##DW##V4(                           \
                        (uint32_t*)dst,                                        \
                        d,                                                     \
                        (const uint32_t*)values,                               \
                        numValues);                                            \
            case 8:                                                            \
                return ZL_sparseNumScatterD##DW##V8(                           \
                        (uint64_t*)dst,                                        \
                        d,                                                     \
                        (const uint64_t*)values,                               \
                        numValues);                                            \
            default:                                                           \
                assert(false);                                                 \
                return 0;                                                      \
        }                                                                      \
    }

ZL_SPARSE_NUM_DEFINE_SCATTERS(1, uint8_t)
ZL_SPARSE_NUM_DEFINE_SCATTERS(2, uint16_t)
ZL_SPARSE_NUM_DEFINE_SCATTERS(4, uint32_t)

/*
 * Public kernel entry point. The destination is first filled with the dominant
 * symbol in one pass, then the literals are written at their positions. This
 * replaces one small fill per run by one large fill, which is much cheaper
 * when runs are short.
 */
void ZL_sparseNumDecode(
        void* dst,
        size_t expectedDstSize,
        const void* distances,
        size_t numDistances,
        size_t distanceWidth,
        uint64_t dominant,
        const void* values,
        size_t numValues,
        size_t valueWidth)
{
    assert(ZL_sparseNumValidValueWidth(valueWidth));
    assert(ZL_sparseNumValidDistanceWidth(distanceWidth));
    assert(numDistances == numValues
           || (numValues < SIZE_MAX && numDistances == numValues + 1));

    ZL_sparseNumDecodeFill(dst, expectedDstSize, dominant, valueWidth);

    size_t produced;
    switch (distanceWidth) {
        case 1:
            produced = ZL_sparseNumScatterD1(
                    dst, distances, values, numValues, valueWidth);
            break;
        case 2:
            produced = ZL_sparseNumScatterD2(
                    dst, distances, values, numValues, valueWidth);
            break;
        case 4:
            produced = ZL_sparseNumScatterD4(
                    dst, distances, values, numValues, valueWidth);
            break;
        default:
            assert(false);
            return;
    }

    /* The optional final distance is the trailing dominant run. */
    if (numDistances != numValues) {
        produced += ZL_sparseNumReadDistance(
                distances, numValues, distanceWidth);
    }
    assert(produced * valueWidth == expectedDstSize);
    (void)produced;
}
