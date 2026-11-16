// Copyright (c) Meta Platforms, Inc. and affiliates.

#include <assert.h>
#include <string.h> // memcpy

#include "openzl/codecs/splitByStruct/encode_splitByStruct_kernel.h"

#include "openzl/shared/mem.h"
#include "openzl/shared/portability.h"

#if ZL_HAS_SSSE3
#    include <tmmintrin.h>
#endif

static void split_3ch_2b(
        void* restrict xPtr,
        void* restrict yPtr,
        void* restrict zPtr,
        const void* srcPtr,
        size_t nbElts)
{
    uint8_t* x         = (uint8_t*)xPtr;
    uint8_t* y         = (uint8_t*)yPtr;
    uint8_t* z         = (uint8_t*)zPtr;
    const uint8_t* src = (const uint8_t*)srcPtr;
    size_t i           = 0;

#if ZL_HAS_AVX2
    __m256i const em2_0_0 = _mm256_setr_epi8(0, 1, 6, 7, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 6, 7, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m256i const em2_0_1 = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 3, 8, 9, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3, 8, 9, 14, 15, -1, -1, -1, -1);
    __m256i const em2_0_2 = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 10, 11);
    __m256i const em2_1_0 = _mm256_setr_epi8(2, 3, 8, 9, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3, 8, 9, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m256i const em2_1_1 = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, 4, 5, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 10, 11, -1, -1, -1, -1, -1, -1);
    __m256i const em2_1_2 = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 6, 7, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 6, 7, 12, 13);
    __m256i const em2_2_0 = _mm256_setr_epi8(4, 5, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m256i const em2_2_1 = _mm256_setr_epi8(-1, -1, -1, -1, 0, 1, 6, 7, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 6, 7, 12, 13, -1, -1, -1, -1, -1, -1);
    __m256i const em2_2_2 = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3, 8, 9, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3, 8, 9, 14, 15);
#endif
#if ZL_HAS_SSSE3
    __m128i const em2_128_0_0 = _mm_setr_epi8(0, 1, 6, 7, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m128i const em2_128_0_1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 3, 8, 9, 14, 15, -1, -1, -1, -1);
    __m128i const em2_128_0_2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 10, 11);
    __m128i const em2_128_1_0 = _mm_setr_epi8(2, 3, 8, 9, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m128i const em2_128_1_1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 4, 5, 10, 11, -1, -1, -1, -1, -1, -1);
    __m128i const em2_128_1_2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 6, 7, 12, 13);
    __m128i const em2_128_2_0 = _mm_setr_epi8(4, 5, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m128i const em2_128_2_1 = _mm_setr_epi8(-1, -1, -1, -1, 0, 1, 6, 7, 12, 13, -1, -1, -1, -1, -1, -1);
    __m128i const em2_128_2_2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3, 8, 9, 14, 15);
#endif
#if ZL_HAS_WASM_SIMD128
    v128_t const em2_w_0_0 = wasm_i8x16_const(0, 1, 6, 7, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    v128_t const em2_w_0_1 = wasm_i8x16_const(-1, -1, -1, -1, -1, -1, 2, 3, 8, 9, 14, 15, -1, -1, -1, -1);
    v128_t const em2_w_0_2 = wasm_i8x16_const(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 10, 11);
    v128_t const em2_w_1_0 = wasm_i8x16_const(2, 3, 8, 9, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    v128_t const em2_w_1_1 = wasm_i8x16_const(-1, -1, -1, -1, -1, -1, 4, 5, 10, 11, -1, -1, -1, -1, -1, -1);
    v128_t const em2_w_1_2 = wasm_i8x16_const(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 6, 7, 12, 13);
    v128_t const em2_w_2_0 = wasm_i8x16_const(4, 5, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    v128_t const em2_w_2_1 = wasm_i8x16_const(-1, -1, -1, -1, 0, 1, 6, 7, 12, 13, -1, -1, -1, -1, -1, -1);
    v128_t const em2_w_2_2 = wasm_i8x16_const(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3, 8, 9, 14, 15);
#endif

#if ZL_HAS_AVX2
    for (; i + 16u <= nbElts; i += 16u) {
        __m256i const a0 = _mm256_loadu_si256((const __m256i_u*)(src + i * 6u + 0u));
        __m256i const a1 = _mm256_loadu_si256((const __m256i_u*)(src + i * 6u + 32u));
        __m256i const a2 = _mm256_loadu_si256((const __m256i_u*)(src + i * 6u + 64u));
        __m256i const t0 = _mm256_permute2x128_si256(a0, a1, 0x30);
        __m256i const t1 = _mm256_permute2x128_si256(a0, a2, 0x21);
        __m256i const t2 = _mm256_permute2x128_si256(a1, a2, 0x30);
        __m256i const vx = _mm256_or_si256(
                _mm256_or_si256(
                        _mm256_shuffle_epi8(t0, em2_0_0),
                        _mm256_shuffle_epi8(t1, em2_0_1)),
                _mm256_shuffle_epi8(t2, em2_0_2));
        __m256i const vy = _mm256_or_si256(
                _mm256_or_si256(
                        _mm256_shuffle_epi8(t0, em2_1_0),
                        _mm256_shuffle_epi8(t1, em2_1_1)),
                _mm256_shuffle_epi8(t2, em2_1_2));
        __m256i const vz = _mm256_or_si256(
                _mm256_or_si256(
                        _mm256_shuffle_epi8(t0, em2_2_0),
                        _mm256_shuffle_epi8(t1, em2_2_1)),
                _mm256_shuffle_epi8(t2, em2_2_2));
        _mm256_storeu_si256((__m256i_u*)(x + i * 2u), vx);
        _mm256_storeu_si256((__m256i_u*)(y + i * 2u), vy);
        _mm256_storeu_si256((__m256i_u*)(z + i * 2u), vz);
    }
#endif
#if ZL_HAS_SSSE3
    for (; i + 8u <= nbElts; i += 8u) {
        __m128i const t0 = _mm_loadu_si128((const __m128i_u*)(src + i * 6u + 0u));
        __m128i const t1 = _mm_loadu_si128((const __m128i_u*)(src + i * 6u + 16u));
        __m128i const t2 = _mm_loadu_si128((const __m128i_u*)(src + i * 6u + 32u));
        __m128i const vx = _mm_or_si128(
                _mm_or_si128(
                        _mm_shuffle_epi8(t0, em2_128_0_0),
                        _mm_shuffle_epi8(t1, em2_128_0_1)),
                _mm_shuffle_epi8(t2, em2_128_0_2));
        __m128i const vy = _mm_or_si128(
                _mm_or_si128(
                        _mm_shuffle_epi8(t0, em2_128_1_0),
                        _mm_shuffle_epi8(t1, em2_128_1_1)),
                _mm_shuffle_epi8(t2, em2_128_1_2));
        __m128i const vz = _mm_or_si128(
                _mm_or_si128(
                        _mm_shuffle_epi8(t0, em2_128_2_0),
                        _mm_shuffle_epi8(t1, em2_128_2_1)),
                _mm_shuffle_epi8(t2, em2_128_2_2));
        _mm_storeu_si128((__m128i_u*)(x + i * 2u), vx);
        _mm_storeu_si128((__m128i_u*)(y + i * 2u), vy);
        _mm_storeu_si128((__m128i_u*)(z + i * 2u), vz);
    }
#elif ZL_HAS_WASM_SIMD128
    for (; i + 8u <= nbElts; i += 8u) {
        v128_t const t0 = wasm_v128_load(src + i * 6u + 0u);
        v128_t const t1 = wasm_v128_load(src + i * 6u + 16u);
        v128_t const t2 = wasm_v128_load(src + i * 6u + 32u);
        v128_t const vx = wasm_v128_or(
                wasm_v128_or(
                        wasm_i8x16_swizzle(t0, em2_w_0_0),
                        wasm_i8x16_swizzle(t1, em2_w_0_1)),
                wasm_i8x16_swizzle(t2, em2_w_0_2));
        v128_t const vy = wasm_v128_or(
                wasm_v128_or(
                        wasm_i8x16_swizzle(t0, em2_w_1_0),
                        wasm_i8x16_swizzle(t1, em2_w_1_1)),
                wasm_i8x16_swizzle(t2, em2_w_1_2));
        v128_t const vz = wasm_v128_or(
                wasm_v128_or(
                        wasm_i8x16_swizzle(t0, em2_w_2_0),
                        wasm_i8x16_swizzle(t1, em2_w_2_1)),
                wasm_i8x16_swizzle(t2, em2_w_2_2));
        wasm_v128_store(x + i * 2u, vx);
        wasm_v128_store(y + i * 2u, vy);
        wasm_v128_store(z + i * 2u, vz);
    }
#endif
    for (; i < nbElts; i++) {
        ZL_write16(x + i * 2u, ZL_read16(src + i * 6u + 0u));
        ZL_write16(y + i * 2u, ZL_read16(src + i * 6u + 2u));
        ZL_write16(z + i * 2u, ZL_read16(src + i * 6u + 4u));
    }
}

static void split_3ch_4b(
        void* restrict xPtr,
        void* restrict yPtr,
        void* restrict zPtr,
        const void* srcPtr,
        size_t nbElts)
{
    uint8_t* x         = (uint8_t*)xPtr;
    uint8_t* y         = (uint8_t*)yPtr;
    uint8_t* z         = (uint8_t*)zPtr;
    const uint8_t* src = (const uint8_t*)srcPtr;
    size_t i           = 0;

#if ZL_HAS_AVX2
    __m256i const em4_0_0 = _mm256_setr_epi8(0, 1, 2, 3, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    __m256i const em4_0_1 = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, -1, -1, -1, -1);
    __m256i const em4_0_2 = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 6, 7);
    __m256i const em4_1_0 = _mm256_setr_epi8(4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m256i const em4_1_1 = _mm256_setr_epi8(-1, -1, -1, -1, 0, 1, 2, 3, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 12, 13, 14, 15, -1, -1, -1, -1);
    __m256i const em4_1_2 = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11);
    __m256i const em4_2_0 = _mm256_setr_epi8(8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m256i const em4_2_1 = _mm256_setr_epi8(-1, -1, -1, -1, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1);
    __m256i const em4_2_2 = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 12, 13, 14, 15);
#endif
#if ZL_HAS_SSSE3
    __m128i const em4_128_0_0 = _mm_setr_epi8(0, 1, 2, 3, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    __m128i const em4_128_0_1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, -1, -1, -1, -1);
    __m128i const em4_128_0_2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 6, 7);
    __m128i const em4_128_1_0 = _mm_setr_epi8(4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m128i const em4_128_1_1 = _mm_setr_epi8(-1, -1, -1, -1, 0, 1, 2, 3, 12, 13, 14, 15, -1, -1, -1, -1);
    __m128i const em4_128_1_2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11);
    __m128i const em4_128_2_0 = _mm_setr_epi8(8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m128i const em4_128_2_1 = _mm_setr_epi8(-1, -1, -1, -1, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1);
    __m128i const em4_128_2_2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 12, 13, 14, 15);
#endif
#if ZL_HAS_WASM_SIMD128
    v128_t const em4_w_0_0 = wasm_i8x16_const(0, 1, 2, 3, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    v128_t const em4_w_0_1 = wasm_i8x16_const(-1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, -1, -1, -1, -1);
    v128_t const em4_w_0_2 = wasm_i8x16_const(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 6, 7);
    v128_t const em4_w_1_0 = wasm_i8x16_const(4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    v128_t const em4_w_1_1 = wasm_i8x16_const(-1, -1, -1, -1, 0, 1, 2, 3, 12, 13, 14, 15, -1, -1, -1, -1);
    v128_t const em4_w_1_2 = wasm_i8x16_const(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11);
    v128_t const em4_w_2_0 = wasm_i8x16_const(8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    v128_t const em4_w_2_1 = wasm_i8x16_const(-1, -1, -1, -1, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1);
    v128_t const em4_w_2_2 = wasm_i8x16_const(-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 12, 13, 14, 15);
#endif

#if ZL_HAS_AVX2
    for (; i + 8u <= nbElts; i += 8u) {
        __m256i const a0 = _mm256_loadu_si256((const __m256i_u*)(src + i * 12u + 0u));
        __m256i const a1 = _mm256_loadu_si256((const __m256i_u*)(src + i * 12u + 32u));
        __m256i const a2 = _mm256_loadu_si256((const __m256i_u*)(src + i * 12u + 64u));
        __m256i const t0 = _mm256_permute2x128_si256(a0, a1, 0x30);
        __m256i const t1 = _mm256_permute2x128_si256(a0, a2, 0x21);
        __m256i const t2 = _mm256_permute2x128_si256(a1, a2, 0x30);
        __m256i const vx = _mm256_or_si256(
                _mm256_or_si256(
                        _mm256_shuffle_epi8(t0, em4_0_0),
                        _mm256_shuffle_epi8(t1, em4_0_1)),
                _mm256_shuffle_epi8(t2, em4_0_2));
        __m256i const vy = _mm256_or_si256(
                _mm256_or_si256(
                        _mm256_shuffle_epi8(t0, em4_1_0),
                        _mm256_shuffle_epi8(t1, em4_1_1)),
                _mm256_shuffle_epi8(t2, em4_1_2));
        __m256i const vz = _mm256_or_si256(
                _mm256_or_si256(
                        _mm256_shuffle_epi8(t0, em4_2_0),
                        _mm256_shuffle_epi8(t1, em4_2_1)),
                _mm256_shuffle_epi8(t2, em4_2_2));
        _mm256_storeu_si256((__m256i_u*)(x + i * 4u), vx);
        _mm256_storeu_si256((__m256i_u*)(y + i * 4u), vy);
        _mm256_storeu_si256((__m256i_u*)(z + i * 4u), vz);
    }
#endif
#if ZL_HAS_SSSE3
    for (; i + 4u <= nbElts; i += 4u) {
        __m128i const t0 = _mm_loadu_si128((const __m128i_u*)(src + i * 12u + 0u));
        __m128i const t1 = _mm_loadu_si128((const __m128i_u*)(src + i * 12u + 16u));
        __m128i const t2 = _mm_loadu_si128((const __m128i_u*)(src + i * 12u + 32u));
        __m128i const vx = _mm_or_si128(
                _mm_or_si128(
                        _mm_shuffle_epi8(t0, em4_128_0_0),
                        _mm_shuffle_epi8(t1, em4_128_0_1)),
                _mm_shuffle_epi8(t2, em4_128_0_2));
        __m128i const vy = _mm_or_si128(
                _mm_or_si128(
                        _mm_shuffle_epi8(t0, em4_128_1_0),
                        _mm_shuffle_epi8(t1, em4_128_1_1)),
                _mm_shuffle_epi8(t2, em4_128_1_2));
        __m128i const vz = _mm_or_si128(
                _mm_or_si128(
                        _mm_shuffle_epi8(t0, em4_128_2_0),
                        _mm_shuffle_epi8(t1, em4_128_2_1)),
                _mm_shuffle_epi8(t2, em4_128_2_2));
        _mm_storeu_si128((__m128i_u*)(x + i * 4u), vx);
        _mm_storeu_si128((__m128i_u*)(y + i * 4u), vy);
        _mm_storeu_si128((__m128i_u*)(z + i * 4u), vz);
    }
#elif ZL_HAS_WASM_SIMD128
    for (; i + 4u <= nbElts; i += 4u) {
        v128_t const t0 = wasm_v128_load(src + i * 12u + 0u);
        v128_t const t1 = wasm_v128_load(src + i * 12u + 16u);
        v128_t const t2 = wasm_v128_load(src + i * 12u + 32u);
        v128_t const vx = wasm_v128_or(
                wasm_v128_or(
                        wasm_i8x16_swizzle(t0, em4_w_0_0),
                        wasm_i8x16_swizzle(t1, em4_w_0_1)),
                wasm_i8x16_swizzle(t2, em4_w_0_2));
        v128_t const vy = wasm_v128_or(
                wasm_v128_or(
                        wasm_i8x16_swizzle(t0, em4_w_1_0),
                        wasm_i8x16_swizzle(t1, em4_w_1_1)),
                wasm_i8x16_swizzle(t2, em4_w_1_2));
        v128_t const vz = wasm_v128_or(
                wasm_v128_or(
                        wasm_i8x16_swizzle(t0, em4_w_2_0),
                        wasm_i8x16_swizzle(t1, em4_w_2_1)),
                wasm_i8x16_swizzle(t2, em4_w_2_2));
        wasm_v128_store(x + i * 4u, vx);
        wasm_v128_store(y + i * 4u, vy);
        wasm_v128_store(z + i * 4u, vz);
    }
#endif
    for (; i < nbElts; i++) {
        ZL_write32(x + i * 4u, ZL_read32(src + i * 12u + 0u));
        ZL_write32(y + i * 4u, ZL_read32(src + i * 12u + 4u));
        ZL_write32(z + i * 4u, ZL_read32(src + i * 12u + 8u));
    }
}


// Function dedicated to structures with all member sizes <= 8
static void ZS_dafss_nosmsgt8(
        void* restrict dstBuffers[],
        size_t nbStructMembers,
        const void* src,
        size_t nbStructs,
        const size_t* structMemberSizes);

// General Function for structures with any member size
static void ZS_dafss_anysms(
        void* restrict dstBuffers[],
        size_t nbStructMembers,
        const void* src,
        size_t nbStructs,
        const size_t* structMemberSizes);

/* ZS_dispatchArrayFixedSizeStruct():
 *
 * See interface descrition in *.h.
 *
 * It's a simple though efficient kernel,
 * which loops through the structure definition
 * to copy each member into its own target buffer.
 *
 * It features 2 variants (automatically selected),
 * a generic one, for structures featuring members of any size,
 * which runs at ~1.4 GB/s on my laptop,
 * and a specialized one, when all members have a size <= 8,
 * which is fairly common (ex: sao, vrs),
 * in which case, speed jumps to 5.4 GB/s.
 *
 * A JIT compiler could do better,
 * by taking full advantage of the fixed structure of fixed size
 * to produce a more compact code with less branches,
 * resulting in faster speed (~9 GB/s for sao ingestion on my laptop).
 * But it's a much more complex undertaking.
 *
 * Another strategy not attempted here is inlining,
 * hoping that the structure definition could be determined at compile time
 * and result in an automatic simplification of the loop.
 * At this stage, it's unclear if this is a reasonable expectation,
 * as the rest of the code can act as an obfuscator,
 * making it impossible for the compiler to keep track of structure definition.
 *
 * The kernel relies on cpu's branch predictor to correctly detect the
 * instruction path. This is probably fine up to a certain (?) amount of
 * elements, as the loop becomes bigger with larger structs, thus making branch
 * history more complex. This issue is simplified in the nosmsgt8 variant, which
 * only has the loop branches to care about, thus expanding branch history by a
 * large factor, but could still become a problem for very large structures with
 * many members.
 *
 * Another potential topic of concern for large structures could be the TLB :
 * as the loop writes into many output buffers, each tracked by a pointer,
 * which is fine as long as the nb of buffers doesn't overflow the TLB.
 * Otherwise, it costs a TLB miss every time, increasing latency.
 * The size of the TLB varies with each cpu architecture, so no single limit.
 * On Skylake, it's 64 (16x4) entries, leaving some room for reasonable
 * structures.
 *
 * If TLB thrashing is a concern, a potential strategy could be prefetching,
 * but it's not guaranteed to make a positive impact,
 * as the amount of prefetching per work unit is very high in my opinion.
 *
 * Another strategy is to fill one @dstBuffer at a time,
 * scanning the entire input and extracting target member at each scan.
 * It requires as many scans are structure members,
 * which becomes more costly as this number increases,
 * so it doesn't seem promising for performance.
 * But it can also be multi-threaded.
 * And rather than doing one member at a time,
 * it could extract a "batch" of members per scan.
 *
 * So, a potential advanced strategy for structures with a _lot_ of members
 * could be to cut the job into several "batches",
 * in charge of a subset of members suitable for TLB capacity,
 * and run them in parallel in different threads.
 */

void ZS_dispatchArrayFixedSizeStruct(
        void* restrict dstBuffers[],
        size_t nbStructMembers,
        const void* src,
        size_t srcSize,
        const size_t* structMemberSizes)
{
    // Special case : empty input ==> no dispatch
    // support src==NULL, and empty structs.
    if (srcSize == 0) {
        return;
    }
    assert(src != NULL);

    // Special case: only one member, effectively a copy-paste
    // Dealing with it here free the caller from special-casing this situation
    // on its own side
    if (nbStructMembers == 1) {
        memcpy(dstBuffers[0], src, srcSize);
        return;
    }

    int smsgt8        = 0;
    size_t structSize = 0;

    for (size_t n = 0; n < nbStructMembers; n++) {
        smsgt8 |= (structMemberSizes[n] > 8);
        structSize += structMemberSizes[n];
    }
    // Since srcSize > 0, only non-empty struct are valid at this stage
    assert(structSize > 0);

    size_t const nbStructs = srcSize / structSize;
    assert(nbStructs * structSize == srcSize); /* exact multiple */

    if (nbStructMembers == 3
        && structMemberSizes[0] == structMemberSizes[1]
        && structMemberSizes[1] == structMemberSizes[2]) {
        size_t const fs = structMemberSizes[0];
        if (fs == 2 || fs == 4) {
            if (fs == 2) {
                split_3ch_2b(
                        dstBuffers[0],
                        dstBuffers[1],
                        dstBuffers[2],
                        src,
                        nbStructs);
            } else {
                split_3ch_4b(
                        dstBuffers[0],
                        dstBuffers[1],
                        dstBuffers[2],
                        src,
                        nbStructs);
            }
            size_t const chBytes = nbStructs * fs;
            dstBuffers[0]        = (char*)dstBuffers[0] + chBytes;
            dstBuffers[1]        = (char*)dstBuffers[1] + chBytes;
            dstBuffers[2]        = (char*)dstBuffers[2] + chBytes;
            return;
        }
    }

    if (!smsgt8) {
        ZS_dafss_nosmsgt8(
                dstBuffers, nbStructMembers, src, nbStructs, structMemberSizes);
        return;
    }

    ZS_dafss_anysms(
            dstBuffers, nbStructMembers, src, nbStructs, structMemberSizes);
}

static void ZS_dafss_nosmsgt8(
        void* restrict dstBuffers[],
        size_t nbStructMembers,
        const void* src,
        size_t nbStructs,
        const size_t* structMemberSizes)
{
    size_t smallestSms = (size_t)(-1);

    for (size_t n = 0; n < nbStructMembers; n++) {
        if (structMemberSizes[n] < smallestSms) {
            smallestSms = structMemberSizes[n];
        }
    }

    // Only non-empty structs are allowed at this stage
    // All fixed-size members are presumed to have a size > 0
    assert(smallestSms > 0);
    // Condition to enter this function : all members must have a size <= 8
    assert(smallestSms <= 8);

    size_t const nbSafeRounds_precalc[8] = { 8, 4, 3, 2, 2, 2, 2, 1 };
    size_t const nbSafeRounds = nbSafeRounds_precalc[smallestSms - 1];

    if (nbStructs > nbSafeRounds) {
        for (size_t n = 0; n < (nbStructs - nbSafeRounds); n++) {
            for (size_t bufid = 0; bufid < nbStructMembers; bufid++) {
                size_t const sms = structMemberSizes[bufid];
                memcpy(dstBuffers[bufid], src, 8);
                dstBuffers[bufid] = (char*)(dstBuffers[bufid]) + sms;
                src               = (const char*)src + sms;
            }
        }
    }

    /* last rounds use exact field size, to control buffer overflow */
    size_t const nbLastRounds =
            (nbStructs < nbSafeRounds) ? nbStructs : nbSafeRounds;
    ZS_dafss_anysms(
            dstBuffers, nbStructMembers, src, nbLastRounds, structMemberSizes);
}

static void ZS_dafss_anysms(
        void* restrict dstBuffers[],
        size_t nbStructMembers,
        const void* src,
        size_t nbStructs,
        const size_t* structMemberSizes)
{
    for (size_t n = 0; n < nbStructs; n++) {
        for (size_t bufid = 0; bufid < nbStructMembers; bufid++) {
            size_t const sms = structMemberSizes[bufid];
            memcpy(dstBuffers[bufid], src, sms);
            dstBuffers[bufid] = (char*)(dstBuffers[bufid]) + sms;
            src               = (const char*)src + sms;
        }
    }
}
