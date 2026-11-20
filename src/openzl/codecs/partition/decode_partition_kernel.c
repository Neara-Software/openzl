// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/partition/decode_partition_kernel.h"

#include "openzl/codecs/common/bitstream/ff_bitstream.h"
#include "openzl/shared/mem.h"
#include "openzl/zl_errors.h"

/// The bitstream accumulator holds 63 bits, and after a reload up to 7 bits
/// may be consumed. A single read must satisfy: residual + nbBits <= 64, so the
/// maximum safe read after reload is 57 bits. Use 56 to match the encoder.
#define ZL_PARTITION_BITS_SPLIT 56

/// Read an offset value from the bitstream, splitting into two reads if the
/// number of bits exceeds ZL_PARTITION_BITS_SPLIT.
static uint64_t ZL_partitionReadBits(ZS_BitDStreamFF* bitstream, size_t nbBits)
{
    if (nbBits <= ZL_PARTITION_BITS_SPLIT) {
        return (uint64_t)ZS_BitDStreamFF_read(bitstream, nbBits);
    } else {
        uint64_t const lo = (uint64_t)ZS_BitDStreamFF_read(bitstream, 32);
        ZS_BitDStreamFF_reload(bitstream);
        uint64_t const hi =
                (uint64_t)ZS_BitDStreamFF_read(bitstream, nbBits - 32);
        return lo | (hi << 32);
    }
}

#if defined(__AVX2__)
#    include <immintrin.h>
#    define ZL_PARTITION_BLOCK 32
#elif defined(__aarch64__)
#    include <arm_neon.h>
#    define ZL_PARTITION_BLOCK 16
#elif defined(__wasm_simd128__)
#    include <wasm_simd128.h>
#    define ZL_PARTITION_BLOCK 16
#else
#    define ZL_PARTITION_BLOCK 16
#endif

/// @returns true iff all ZL_PARTITION_BLOCK buckets are < @p limit.
/// @pre limit >= 1
static bool ZL_partitionBlockAllBelow(uint8_t const* buckets, uint8_t limit)
{
#if defined(__AVX2__)
    __m256i const v   = _mm256_loadu_si256((const __m256i*)buckets);
    __m256i const max = _mm256_set1_epi8((char)(limit - 1));
    __m256i const ok  = _mm256_cmpeq_epi8(_mm256_min_epu8(v, max), v);
    return _mm256_movemask_epi8(ok) == -1;
#elif defined(__aarch64__)
    return vminvq_u8(vcltq_u8(vld1q_u8(buckets), vdupq_n_u8(limit))) != 0;
#elif defined(__wasm_simd128__)
    return wasm_i8x16_all_true(wasm_u8x16_lt(
            wasm_v128_load(buckets), wasm_i8x16_splat((int8_t)limit)));
#else
    unsigned above = 0;
    for (size_t i = 0; i < ZL_PARTITION_BLOCK; ++i) {
        above |= (unsigned)(buckets[i] >= limit);
    }
    return above == 0;
#endif
}

/// Decode one block of buckets that all select single-value partitions in the
/// leading run of single-value partitions. Those partitions have consecutive
/// bases and no offset bits, so value = start + bucket.
static void ZL_partitionDecodeDirectBlock8(
        uint8_t* dst,
        uint8_t const* buckets,
        uint8_t start)
{
#if defined(__AVX2__)
    _mm256_storeu_si256(
            (__m256i*)dst,
            _mm256_add_epi8(
                    _mm256_loadu_si256((const __m256i*)buckets),
                    _mm256_set1_epi8((char)start)));
#elif defined(__aarch64__)
    vst1q_u8(dst, vaddq_u8(vld1q_u8(buckets), vdupq_n_u8(start)));
#elif defined(__wasm_simd128__)
    wasm_v128_store(
            dst,
            wasm_i8x16_add(
                    wasm_v128_load(buckets), wasm_i8x16_splat((int8_t)start)));
#else
    for (size_t i = 0; i < ZL_PARTITION_BLOCK; ++i) {
        dst[i] = (uint8_t)(start + buckets[i]);
    }
#endif
}

static void ZL_partitionDecodeDirectBlock16(
        uint16_t* dst,
        uint8_t const* buckets,
        uint16_t start)
{
#if defined(__AVX2__)
    __m256i const vstart = _mm256_set1_epi16((short)start);
    __m128i const lo     = _mm_loadu_si128((const __m128i*)buckets);
    __m128i const hi     = _mm_loadu_si128((const __m128i*)(buckets + 16));
    _mm256_storeu_si256(
            (__m256i*)dst,
            _mm256_add_epi16(_mm256_cvtepu8_epi16(lo), vstart));
    _mm256_storeu_si256(
            (__m256i*)(dst + 16),
            _mm256_add_epi16(_mm256_cvtepu8_epi16(hi), vstart));
#elif defined(__aarch64__)
    uint8x16_t const v       = vld1q_u8(buckets);
    uint16x8_t const vstart = vdupq_n_u16(start);
    vst1q_u16(dst, vaddw_u8(vstart, vget_low_u8(v)));
    vst1q_u16(dst + 8, vaddw_u8(vstart, vget_high_u8(v)));
#elif defined(__wasm_simd128__)
    v128_t const v      = wasm_v128_load(buckets);
    v128_t const vstart = wasm_i16x8_splat((int16_t)start);
    wasm_v128_store(
            dst, wasm_i16x8_add(wasm_u16x8_extend_low_u8x16(v), vstart));
    wasm_v128_store(
            dst + 8, wasm_i16x8_add(wasm_u16x8_extend_high_u8x16(v), vstart));
#else
    for (size_t i = 0; i < ZL_PARTITION_BLOCK; ++i) {
        dst[i] = (uint16_t)(start + buckets[i]);
    }
#endif
}

static void ZL_partitionDecodeDirectBlock32(
        uint32_t* dst,
        uint8_t const* buckets,
        uint32_t start)
{
    for (size_t i = 0; i < ZL_PARTITION_BLOCK; ++i) {
        dst[i] = start + buckets[i];
    }
}

static void ZL_partitionDecodeDirectBlock64(
        uint64_t* dst,
        uint8_t const* buckets,
        uint64_t start)
{
    for (size_t i = 0; i < ZL_PARTITION_BLOCK; ++i) {
        dst[i] = start + buckets[i];
    }
}

/// Decode loop for one element width.
///
/// Partitions with no offset bits do not touch the bitstream, so the reload is
/// skipped for them. Blocks whose buckets all fall in the first @p numDirect
/// partitions (each of size 1) are decoded without the bitstream at all.
#define ZL_PARTITION_DEFINE_DECODE(W, T)                                       \
    static void ZL_partitionDecode##W(                                         \
            T* dst,                                                            \
            uint8_t const* buckets,                                            \
            size_t nbValues,                                                   \
            ZS_BitDStreamFF* bitstream,                                        \
            uint64_t const* bases,                                             \
            uint8_t const* bits,                                               \
            size_t numDirect)                                                  \
    {                                                                          \
        size_t i = 0;                                                          \
        if (numDirect > 0) {                                                   \
            uint8_t const limit =                                              \
                    (uint8_t)(numDirect > 255 ? 255 : numDirect);              \
            for (; i + ZL_PARTITION_BLOCK <= nbValues;                         \
                 i += ZL_PARTITION_BLOCK) {                                    \
                if (ZL_partitionBlockAllBelow(buckets + i, limit)) {           \
                    ZL_partitionDecodeDirectBlock##W(                          \
                            dst + i, buckets + i, (T)bases[0]);                \
                    continue;                                                  \
                }                                                              \
                for (size_t j = i; j < i + ZL_PARTITION_BLOCK; ++j) {          \
                    uint8_t const bucket = buckets[j];                         \
                    uint8_t const nbBits = bits[bucket];                       \
                    uint64_t val         = bases[bucket];                      \
                    if (nbBits != 0) {                                         \
                        val += ZL_partitionReadBits(bitstream, nbBits);        \
                        ZS_BitDStreamFF_reload(bitstream);                     \
                    }                                                          \
                    dst[j] = (T)val;                                           \
                }                                                              \
            }                                                                  \
        }                                                                      \
        for (; i < nbValues; ++i) {                                            \
            uint8_t const bucket = buckets[i];                                 \
            uint8_t const nbBits = bits[bucket];                               \
            uint64_t val         = bases[bucket];                              \
            if (nbBits != 0) {                                                 \
                val += ZL_partitionReadBits(bitstream, nbBits);                \
                ZS_BitDStreamFF_reload(bitstream);                             \
            }                                                                  \
            dst[i] = (T)val;                                                   \
        }                                                                      \
    }

ZL_PARTITION_DEFINE_DECODE(8, uint8_t)
ZL_PARTITION_DEFINE_DECODE(16, uint16_t)
ZL_PARTITION_DEFINE_DECODE(32, uint32_t)
ZL_PARTITION_DEFINE_DECODE(64, uint64_t)

ZL_Report ZL_partitionDecode(
        void* dst,
        size_t eltWidth,
        uint8_t const* buckets,
        size_t nbValues,
        uint8_t const* offsets,
        size_t offsetsSize,
        ZL_PartitionParams const* params)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(NULL);
    ZL_ASSERT(ZL_PartitionParams_validate(params));

    uint64_t bases[ZL_PARTITION_MAX_PARTITIONS];
    uint8_t bits[ZL_PARTITION_MAX_PARTITIONS];
    ZL_PartitionParams_computeBasesU64(params, bases);
    ZL_PartitionParams_computeBits(params, bits);
    /* Buckets are untrusted: give out-of-range buckets a defined result. */
    for (size_t i = params->numPartitions; i < ZL_PARTITION_MAX_PARTITIONS;
         ++i) {
        bases[i] = 0;
        bits[i]  = 0;
    }

    /* Length of the leading run of size-1 partitions. */
    size_t numDirect = 0;
    while (numDirect < params->numPartitions
           && params->partitionSizes[numDirect] == 1) {
        ++numDirect;
    }

    ZS_BitDStreamFF bitstream = ZS_BitDStreamFF_init(offsets, offsetsSize);

    switch (eltWidth) {
        case 1:
            ZL_partitionDecode8(
                    (uint8_t*)dst,
                    buckets,
                    nbValues,
                    &bitstream,
                    bases,
                    bits,
                    numDirect);
            break;
        case 2:
            ZL_partitionDecode16(
                    (uint16_t*)dst,
                    buckets,
                    nbValues,
                    &bitstream,
                    bases,
                    bits,
                    numDirect);
            break;
        case 4:
            ZL_partitionDecode32(
                    (uint32_t*)dst,
                    buckets,
                    nbValues,
                    &bitstream,
                    bases,
                    bits,
                    numDirect);
            break;
        case 8:
            ZL_partitionDecode64(
                    (uint64_t*)dst,
                    buckets,
                    nbValues,
                    &bitstream,
                    bases,
                    bits,
                    numDirect);
            break;
        default:
            break;
    }

    ZL_Report const ret = ZS_BitDStreamFF_finish(&bitstream);
    ZL_ERR_IF(ZL_isError(ret), srcSize_tooSmall);

    return ZL_returnSuccess();
}
