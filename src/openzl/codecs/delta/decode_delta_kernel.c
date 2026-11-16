// Copyright (c) Meta Platforms, Inc. and affiliates.

// note : relative-path include (same directory by default)
#include "decode_delta_kernel.h"

#include <assert.h>
#include <stdint.h> // uint32_t

#include "openzl/shared/mem.h"
#include "openzl/shared/portability.h"

// note : some template capability would be welcomed here

/* Vectorized helpers are optional accelerators for ZS_deltaDecodeXX().
 * They expect dst[0] to be already written, decode as many whole vectors as
 * possible starting from element 1, and return the index of the first element
 * not yet decoded. The scalar loop in ZS_deltaDecodeXX() finishes the job, so
 * removing a helper changes speed, not results. */

#if ZL_HAS_SSSE3

#    include <tmmintrin.h>

/* Each iteration computes the prefix sums of one vector of deltas, then adds
 * them to `prev`, the last value decoded so far. `prev` is updated from the
 * local prefix sums rather than from the stored result, so that only one add
 * sits on the dependency chain between iterations, not the shuffle. */

static size_t
ZS_deltaDecode8_ssse3(uint8_t* dst, uint8_t const* deltas, size_t nbElts)
{
    assert(nbElts > 0);
    size_t const kEltsPerIter = sizeof(__m128i) / sizeof(*deltas);
    __m128i prev              = _mm_set1_epi8((char)dst[0]);
    size_t n                  = 1;
    for (; n + kEltsPerIter <= nbElts; n += kEltsPerIter) {
        __m128i values = _mm_loadu_si128((__m128i_u const*)&deltas[n - 1]);
        values         = _mm_add_epi8(values, _mm_slli_si128(values, 8));
        values         = _mm_add_epi8(values, _mm_slli_si128(values, 4));
        values         = _mm_add_epi8(values, _mm_slli_si128(values, 2));
        values         = _mm_add_epi8(values, _mm_slli_si128(values, 1));
        _mm_storeu_si128((__m128i_u*)&dst[n], _mm_add_epi8(values, prev));
        prev = _mm_add_epi8(
                prev, _mm_shuffle_epi8(values, _mm_set1_epi8(0x0f)));
    }
    return n;
}

static size_t
ZS_deltaDecode16_ssse3(uint16_t* dst, uint16_t const* deltas, size_t nbElts)
{
    assert(nbElts > 0);
    size_t const kEltsPerIter = sizeof(__m128i) / sizeof(*deltas);
    __m128i prev              = _mm_set1_epi16((int16_t)dst[0]);
    size_t n                  = 1;
    for (; n + kEltsPerIter <= nbElts; n += kEltsPerIter) {
        __m128i values = _mm_loadu_si128((__m128i_u const*)&deltas[n - 1]);
        values         = _mm_add_epi16(values, _mm_slli_si128(values, 8));
        values         = _mm_add_epi16(values, _mm_slli_si128(values, 4));
        values         = _mm_add_epi16(values, _mm_slli_si128(values, 2));
        _mm_storeu_si128((__m128i_u*)&dst[n], _mm_add_epi16(values, prev));
        prev = _mm_add_epi16(
                prev, _mm_shuffle_epi8(values, _mm_set1_epi16(0x0f0e)));
    }
    return n;
}

static size_t
ZS_deltaDecode32_ssse3(uint32_t* dst, uint32_t const* deltas, size_t nbElts)
{
    // TODO: Consider aligning either the source or the destination buffer,
    // or both. I've measured that this loop is 10% slower when working on
    // unaligned output buffers. Interestingly, the other vectorized delta
    // decoders aren't affected by the alignment.
    // We could either align here by decoding a few scalar elements first.
    // Or ask zstrong to give us aligned buffers when it doesn't need an
    // extra memcpy to do so.
    assert(nbElts > 0);
    size_t const kEltsPerIter = sizeof(__m128i) / sizeof(*deltas);
    __m128i prev              = _mm_set1_epi32((int32_t)dst[0]);
    size_t n                  = 1;
    for (; n + kEltsPerIter <= nbElts; n += kEltsPerIter) {
        __m128i values = _mm_loadu_si128((__m128i_u const*)&deltas[n - 1]);
        values         = _mm_add_epi32(values, _mm_slli_si128(values, 8));
        values         = _mm_add_epi32(values, _mm_slli_si128(values, 4));
        _mm_storeu_si128((__m128i_u*)&dst[n], _mm_add_epi32(values, prev));
        prev = _mm_add_epi32(prev, _mm_shuffle_epi32(values, 0xff));
    }
    return n;
}

#endif // ZL_HAS_SSSE3

#if ZL_HAS_AVX2

static size_t
ZS_deltaDecode64_avx2(uint64_t* dst, uint64_t const* deltas, size_t nbElts)
{
    assert(nbElts > 0);
    size_t const kEltsPerIter = sizeof(__m256i) / sizeof(*deltas);
    __m256i prev              = _mm256_set1_epi64x((long long)dst[0]);
    __m256i const zero        = _mm256_setzero_si256();
    size_t n                  = 1;
    for (; n + kEltsPerIter <= nbElts; n += kEltsPerIter) {
        __m256i values  = _mm256_loadu_si256((__m256i_u const*)&deltas[n - 1]);
        values          = _mm256_add_epi64(values, _mm256_slli_si256(values, 8));
        __m256i low_sum = _mm256_permute4x64_epi64(values, 0x55);
        low_sum         = _mm256_blend_epi32(zero, low_sum, 0xF0);
        values          = _mm256_add_epi64(values, low_sum);
        _mm256_storeu_si256((__m256i_u*)&dst[n], _mm256_add_epi64(values, prev));
        prev = _mm256_add_epi64(prev, _mm256_permute4x64_epi64(values, 0xFF));
    }
    return n;
}

#endif // ZL_HAS_AVX2

#if ZL_HAS_WASM_SIMD128

static size_t
ZS_deltaDecode8_wasm(uint8_t* dst, uint8_t const* deltas, size_t nbElts)
{
    assert(nbElts > 0);
    size_t const kEltsPerIter = sizeof(v128_t) / sizeof(*deltas);
    v128_t prev               = wasm_u8x16_splat(dst[0]);
    v128_t const zero         = wasm_u8x16_splat(0);
    size_t n                  = 1;
    for (; n + kEltsPerIter <= nbElts; n += kEltsPerIter) {
        v128_t values = wasm_v128_load(&deltas[n - 1]);
        values        = wasm_i8x16_add(
                values,
                wasm_i8x16_shuffle(
                        zero, values, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23));
        values = wasm_i8x16_add(
                values,
                wasm_i8x16_shuffle(
                        zero, values, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27));
        values = wasm_i8x16_add(
                values,
                wasm_i8x16_shuffle(
                        zero, values, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29));
        values = wasm_i8x16_add(
                values,
                wasm_i8x16_shuffle(
                        zero, values, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30));
        wasm_v128_store(&dst[n], wasm_i8x16_add(values, prev));
        prev = wasm_i8x16_add(
                prev,
                wasm_i8x16_shuffle(
                        values, values, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15));
    }
    return n;
}

static size_t
ZS_deltaDecode16_wasm(uint16_t* dst, uint16_t const* deltas, size_t nbElts)
{
    assert(nbElts > 0);
    size_t const kEltsPerIter = sizeof(v128_t) / sizeof(*deltas);
    v128_t prev               = wasm_u16x8_splat(dst[0]);
    v128_t const zero         = wasm_u16x8_splat(0);
    size_t n                  = 1;
    for (; n + kEltsPerIter <= nbElts; n += kEltsPerIter) {
        v128_t values = wasm_v128_load(&deltas[n - 1]);
        values        = wasm_i16x8_add(
                values,
                wasm_i8x16_shuffle(
                        zero, values, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23));
        values = wasm_i16x8_add(
                values,
                wasm_i8x16_shuffle(
                        zero, values, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27));
        values = wasm_i16x8_add(
                values,
                wasm_i8x16_shuffle(
                        zero, values, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29));
        wasm_v128_store(&dst[n], wasm_i16x8_add(values, prev));
        prev = wasm_i16x8_add(
                prev, wasm_i16x8_shuffle(values, values, 7, 7, 7, 7, 7, 7, 7, 7));
    }
    return n;
}

static size_t
ZS_deltaDecode32_wasm(uint32_t* dst, uint32_t const* deltas, size_t nbElts)
{
    assert(nbElts > 0);
    size_t const kEltsPerIter = sizeof(v128_t) / sizeof(*deltas);
    v128_t prev               = wasm_u32x4_splat(dst[0]);
    v128_t const zero         = wasm_u32x4_splat(0);
    size_t n                  = 1;
    for (; n + kEltsPerIter <= nbElts; n += kEltsPerIter) {
        v128_t values = wasm_v128_load(&deltas[n - 1]);
        values        = wasm_i32x4_add(
                values,
                wasm_i8x16_shuffle(
                        zero, values, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23));
        values = wasm_i32x4_add(
                values,
                wasm_i8x16_shuffle(
                        zero, values, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27));
        wasm_v128_store(&dst[n], wasm_i32x4_add(values, prev));
        prev = wasm_i32x4_add(
                prev, wasm_i32x4_shuffle(values, values, 3, 3, 3, 3));
    }
    return n;
}

static size_t
ZS_deltaDecode64_wasm(uint64_t* dst, uint64_t const* deltas, size_t nbElts)
{
    assert(nbElts > 0);
    size_t const kEltsPerIter = sizeof(v128_t) / sizeof(*deltas);
    v128_t prev               = wasm_u64x2_splat(dst[0]);
    v128_t const zero         = wasm_u64x2_splat(0);
    size_t n                  = 1;
    for (; n + kEltsPerIter <= nbElts; n += kEltsPerIter) {
        v128_t values = wasm_v128_load(&deltas[n - 1]);
        values        = wasm_i64x2_add(
                values, wasm_i64x2_shuffle(zero, values, 0, 2));
        wasm_v128_store(&dst[n], wasm_i64x2_add(values, prev));
        prev = wasm_i64x2_add(prev, wasm_i64x2_shuffle(values, values, 1, 1));
    }
    return n;
}

#endif // ZL_HAS_WASM_SIMD128

void ZS_deltaDecode8(
        uint8_t* dst,
        uint8_t first,
        uint8_t const* deltas,
        size_t nbElts)
{
    if (nbElts == 0) {
        return;
    }
    dst[0]   = first;
    size_t n = 1;
#if ZL_HAS_SSSE3
    n = ZS_deltaDecode8_ssse3(dst, deltas, nbElts);
#elif ZL_HAS_WASM_SIMD128
    n = ZS_deltaDecode8_wasm(dst, deltas, nbElts);
#endif
    for (; n < nbElts; ++n) {
        dst[n] = (uint8_t)(dst[n - 1] + deltas[n - 1]);
    }
}

void ZS_deltaDecode16(
        uint16_t* dst,
        uint16_t first,
        uint16_t const* deltas,
        size_t nbElts)
{
    if (nbElts == 0) {
        return;
    }
    dst[0]   = first;
    size_t n = 1;
#if ZL_HAS_SSSE3
    n = ZS_deltaDecode16_ssse3(dst, deltas, nbElts);
#elif ZL_HAS_WASM_SIMD128
    n = ZS_deltaDecode16_wasm(dst, deltas, nbElts);
#endif
    for (; n < nbElts; ++n) {
        dst[n] = (uint16_t)(dst[n - 1] + deltas[n - 1]);
    }
}

void ZS_deltaDecode32(
        uint32_t* dst,
        uint32_t first,
        uint32_t const* deltas,
        size_t nbElts)
{
    if (nbElts == 0) {
        return;
    }
    dst[0]   = first;
    size_t n = 1;
#if ZL_HAS_SSSE3
    n = ZS_deltaDecode32_ssse3(dst, deltas, nbElts);
#elif ZL_HAS_WASM_SIMD128
    n = ZS_deltaDecode32_wasm(dst, deltas, nbElts);
#endif
    for (; n < nbElts; ++n) {
        dst[n] = dst[n - 1] + deltas[n - 1];
    }
}

void ZS_deltaDecode64(
        uint64_t* dst,
        uint64_t first,
        uint64_t const* deltas,
        size_t nbElts)
{
    if (nbElts == 0) {
        return;
    }
    dst[0]   = first;
    size_t n = 1;
#if ZL_HAS_AVX2
    n = ZS_deltaDecode64_avx2(dst, deltas, nbElts);
#elif ZL_HAS_WASM_SIMD128
    n = ZS_deltaDecode64_wasm(dst, deltas, nbElts);
#endif
    for (; n < nbElts; ++n) {
        dst[n] = dst[n - 1] + deltas[n - 1];
    }
}

void ZS_deltaDecode(
        void* dst,
        void const* first,
        void const* deltas,
        size_t nbElts,
        size_t eltWidth)
{
    if (nbElts == 0) {
        return;
    }
    switch (eltWidth) {
        case 1:
            ZS_deltaDecode8(
                    (uint8_t*)dst,
                    ZL_read8(first),
                    (uint8_t const*)deltas,
                    nbElts);
            break;
        case 2:
            ZS_deltaDecode16(
                    (uint16_t*)dst,
                    ZL_readLE16(first),
                    (uint16_t const*)deltas,
                    nbElts);
            break;
        case 4:
            ZS_deltaDecode32(
                    (uint32_t*)dst,
                    ZL_readLE32(first),
                    (uint32_t const*)deltas,
                    nbElts);
            break;
        case 8:
            ZS_deltaDecode64(
                    (uint64_t*)dst,
                    ZL_readLE64(first),
                    (uint64_t const*)deltas,
                    nbElts);
            break;
        default:
            assert(false);
            break;
    }
}
