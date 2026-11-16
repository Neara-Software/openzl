// Copyright (c) Meta Platforms, Inc. and affiliates.

// note : this is the raw kernel transform,
//        which implies minimal #include policy
#include "decode_splitByStruct_kernel.h"

#include <assert.h>
#include <string.h> // memcpy

#include "openzl/shared/mem.h"
#include "openzl/shared/portability.h"

#if ZL_HAS_SSSE3
#    include <tmmintrin.h>
#endif

static void rejoin_3ch_2b(
        void* restrict dst,
        const void* restrict xPtr,
        const void* restrict yPtr,
        const void* restrict zPtr,
        size_t nbElts)
{
    uint8_t* out       = (uint8_t*)dst;
    const uint8_t* x   = (const uint8_t*)xPtr;
    const uint8_t* y   = (const uint8_t*)yPtr;
    const uint8_t* z   = (const uint8_t*)zPtr;
    size_t i           = 0;

#if ZL_HAS_AVX2
    __m256i const dm2_0_0 = _mm256_setr_epi8(0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5, -1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5, -1, -1);
    __m256i const dm2_0_1 = _mm256_setr_epi8(-1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5, -1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5);
    __m256i const dm2_0_2 = _mm256_setr_epi8(-1, -1, -1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1);
    __m256i const dm2_1_0 = _mm256_setr_epi8(-1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1, 10, 11, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1, 10, 11);
    __m256i const dm2_1_1 = _mm256_setr_epi8(-1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1, -1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1);
    __m256i const dm2_1_2 = _mm256_setr_epi8(4, 5, -1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, 4, 5, -1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1);
    __m256i const dm2_2_0 = _mm256_setr_epi8(-1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1, -1, -1);
    __m256i const dm2_2_1 = _mm256_setr_epi8(10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1, 10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1);
    __m256i const dm2_2_2 = _mm256_setr_epi8(-1, -1, 10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1, 10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15);
#endif
#if ZL_HAS_SSSE3
    __m128i const dm2_128_0_0 = _mm_setr_epi8(0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5, -1, -1);
    __m128i const dm2_128_0_1 = _mm_setr_epi8(-1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5);
    __m128i const dm2_128_0_2 = _mm_setr_epi8(-1, -1, -1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1);
    __m128i const dm2_128_1_0 = _mm_setr_epi8(-1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1, 10, 11);
    __m128i const dm2_128_1_1 = _mm_setr_epi8(-1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1);
    __m128i const dm2_128_1_2 = _mm_setr_epi8(4, 5, -1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1);
    __m128i const dm2_128_2_0 = _mm_setr_epi8(-1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1, -1, -1);
    __m128i const dm2_128_2_1 = _mm_setr_epi8(10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1);
    __m128i const dm2_128_2_2 = _mm_setr_epi8(-1, -1, 10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15);
#endif
#if ZL_HAS_WASM_SIMD128
    v128_t const dm2_w_0_0 = wasm_i8x16_const(0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5, -1, -1);
    v128_t const dm2_w_0_1 = wasm_i8x16_const(-1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5);
    v128_t const dm2_w_0_2 = wasm_i8x16_const(-1, -1, -1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1);
    v128_t const dm2_w_1_0 = wasm_i8x16_const(-1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1, 10, 11);
    v128_t const dm2_w_1_1 = wasm_i8x16_const(-1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1);
    v128_t const dm2_w_1_2 = wasm_i8x16_const(4, 5, -1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1);
    v128_t const dm2_w_2_0 = wasm_i8x16_const(-1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1, -1, -1);
    v128_t const dm2_w_2_1 = wasm_i8x16_const(10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1);
    v128_t const dm2_w_2_2 = wasm_i8x16_const(-1, -1, 10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15);
#endif

#if ZL_HAS_AVX2
    for (; i + 16u <= nbElts; i += 16u) {
        __m256i const vx = _mm256_loadu_si256((const __m256i_u*)(x + i * 2u));
        __m256i const vy = _mm256_loadu_si256((const __m256i_u*)(y + i * 2u));
        __m256i const vz = _mm256_loadu_si256((const __m256i_u*)(z + i * 2u));
        __m256i const t0 = _mm256_or_si256(
                _mm256_or_si256(
                        _mm256_shuffle_epi8(vx, dm2_0_0),
                        _mm256_shuffle_epi8(vy, dm2_0_1)),
                _mm256_shuffle_epi8(vz, dm2_0_2));
        __m256i const t1 = _mm256_or_si256(
                _mm256_or_si256(
                        _mm256_shuffle_epi8(vx, dm2_1_0),
                        _mm256_shuffle_epi8(vy, dm2_1_1)),
                _mm256_shuffle_epi8(vz, dm2_1_2));
        __m256i const t2 = _mm256_or_si256(
                _mm256_or_si256(
                        _mm256_shuffle_epi8(vx, dm2_2_0),
                        _mm256_shuffle_epi8(vy, dm2_2_1)),
                _mm256_shuffle_epi8(vz, dm2_2_2));
        _mm256_storeu_si256(
                (__m256i_u*)(out + i * 6u + 0u),
                _mm256_permute2x128_si256(t0, t1, 0x20));
        _mm256_storeu_si256(
                (__m256i_u*)(out + i * 6u + 32u),
                _mm256_permute2x128_si256(t2, t0, 0x30));
        _mm256_storeu_si256(
                (__m256i_u*)(out + i * 6u + 64u),
                _mm256_permute2x128_si256(t1, t2, 0x31));
    }
#endif
#if ZL_HAS_SSSE3
    for (; i + 8u <= nbElts; i += 8u) {
        __m128i const vx = _mm_loadu_si128((const __m128i_u*)(x + i * 2u));
        __m128i const vy = _mm_loadu_si128((const __m128i_u*)(y + i * 2u));
        __m128i const vz = _mm_loadu_si128((const __m128i_u*)(z + i * 2u));
        __m128i const o0 = _mm_or_si128(
                _mm_or_si128(
                        _mm_shuffle_epi8(vx, dm2_128_0_0),
                        _mm_shuffle_epi8(vy, dm2_128_0_1)),
                _mm_shuffle_epi8(vz, dm2_128_0_2));
        __m128i const o1 = _mm_or_si128(
                _mm_or_si128(
                        _mm_shuffle_epi8(vx, dm2_128_1_0),
                        _mm_shuffle_epi8(vy, dm2_128_1_1)),
                _mm_shuffle_epi8(vz, dm2_128_1_2));
        __m128i const o2 = _mm_or_si128(
                _mm_or_si128(
                        _mm_shuffle_epi8(vx, dm2_128_2_0),
                        _mm_shuffle_epi8(vy, dm2_128_2_1)),
                _mm_shuffle_epi8(vz, dm2_128_2_2));
        _mm_storeu_si128((__m128i_u*)(out + i * 6u + 0u), o0);
        _mm_storeu_si128((__m128i_u*)(out + i * 6u + 16u), o1);
        _mm_storeu_si128((__m128i_u*)(out + i * 6u + 32u), o2);
    }
#elif ZL_HAS_WASM_SIMD128
    for (; i + 8u <= nbElts; i += 8u) {
        v128_t const vx = wasm_v128_load(x + i * 2u);
        v128_t const vy = wasm_v128_load(y + i * 2u);
        v128_t const vz = wasm_v128_load(z + i * 2u);
        v128_t const o0 = wasm_v128_or(
                wasm_v128_or(
                        wasm_i8x16_swizzle(vx, dm2_w_0_0),
                        wasm_i8x16_swizzle(vy, dm2_w_0_1)),
                wasm_i8x16_swizzle(vz, dm2_w_0_2));
        v128_t const o1 = wasm_v128_or(
                wasm_v128_or(
                        wasm_i8x16_swizzle(vx, dm2_w_1_0),
                        wasm_i8x16_swizzle(vy, dm2_w_1_1)),
                wasm_i8x16_swizzle(vz, dm2_w_1_2));
        v128_t const o2 = wasm_v128_or(
                wasm_v128_or(
                        wasm_i8x16_swizzle(vx, dm2_w_2_0),
                        wasm_i8x16_swizzle(vy, dm2_w_2_1)),
                wasm_i8x16_swizzle(vz, dm2_w_2_2));
        wasm_v128_store(out + i * 6u + 0u, o0);
        wasm_v128_store(out + i * 6u + 16u, o1);
        wasm_v128_store(out + i * 6u + 32u, o2);
    }
#endif
    for (; i < nbElts; i++) {
        ZL_write16(out + i * 6u + 0u, ZL_read16(x + i * 2u));
        ZL_write16(out + i * 6u + 2u, ZL_read16(y + i * 2u));
        ZL_write16(out + i * 6u + 4u, ZL_read16(z + i * 2u));
    }
}

static void rejoin_3ch_4b(
        void* restrict dst,
        const void* restrict xPtr,
        const void* restrict yPtr,
        const void* restrict zPtr,
        size_t nbElts)
{
    uint8_t* out       = (uint8_t*)dst;
    const uint8_t* x   = (const uint8_t*)xPtr;
    const uint8_t* y   = (const uint8_t*)yPtr;
    const uint8_t* z   = (const uint8_t*)zPtr;
    size_t i           = 0;

#if ZL_HAS_AVX2
    __m256i const dm4_0_0 = _mm256_setr_epi8(0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 6, 7, 0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 6, 7);
    __m256i const dm4_0_1 = _mm256_setr_epi8(-1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1);
    __m256i const dm4_0_2 = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1);
    __m256i const dm4_1_0 = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, -1, -1, -1, -1);
    __m256i const dm4_1_1 = _mm256_setr_epi8(4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11);
    __m256i const dm4_1_2 = _mm256_setr_epi8(-1, -1, -1, -1, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1);
    __m256i const dm4_2_0 = _mm256_setr_epi8(-1, -1, -1, -1, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    __m256i const dm4_2_1 = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 13, 14, 15, -1, -1, -1, -1);
    __m256i const dm4_2_2 = _mm256_setr_epi8(8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, 12, 13, 14, 15, 8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, 12, 13, 14, 15);
#endif
#if ZL_HAS_SSSE3
    __m128i const dm4_128_0_0 = _mm_setr_epi8(0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 6, 7);
    __m128i const dm4_128_0_1 = _mm_setr_epi8(-1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1);
    __m128i const dm4_128_0_2 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1);
    __m128i const dm4_128_1_0 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, -1, -1, -1, -1);
    __m128i const dm4_128_1_1 = _mm_setr_epi8(4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11);
    __m128i const dm4_128_1_2 = _mm_setr_epi8(-1, -1, -1, -1, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1);
    __m128i const dm4_128_2_0 = _mm_setr_epi8(-1, -1, -1, -1, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    __m128i const dm4_128_2_1 = _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 12, 13, 14, 15, -1, -1, -1, -1);
    __m128i const dm4_128_2_2 = _mm_setr_epi8(8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, 12, 13, 14, 15);
#endif
#if ZL_HAS_WASM_SIMD128
    v128_t const dm4_w_0_0 = wasm_i8x16_const(0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 6, 7);
    v128_t const dm4_w_0_1 = wasm_i8x16_const(-1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1);
    v128_t const dm4_w_0_2 = wasm_i8x16_const(-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1);
    v128_t const dm4_w_1_0 = wasm_i8x16_const(-1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, -1, -1, -1, -1);
    v128_t const dm4_w_1_1 = wasm_i8x16_const(4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11);
    v128_t const dm4_w_1_2 = wasm_i8x16_const(-1, -1, -1, -1, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1);
    v128_t const dm4_w_2_0 = wasm_i8x16_const(-1, -1, -1, -1, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    v128_t const dm4_w_2_1 = wasm_i8x16_const(-1, -1, -1, -1, -1, -1, -1, -1, 12, 13, 14, 15, -1, -1, -1, -1);
    v128_t const dm4_w_2_2 = wasm_i8x16_const(8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, 12, 13, 14, 15);
#endif

#if ZL_HAS_AVX2
    for (; i + 8u <= nbElts; i += 8u) {
        __m256i const vx = _mm256_loadu_si256((const __m256i_u*)(x + i * 4u));
        __m256i const vy = _mm256_loadu_si256((const __m256i_u*)(y + i * 4u));
        __m256i const vz = _mm256_loadu_si256((const __m256i_u*)(z + i * 4u));
        __m256i const t0 = _mm256_or_si256(
                _mm256_or_si256(
                        _mm256_shuffle_epi8(vx, dm4_0_0),
                        _mm256_shuffle_epi8(vy, dm4_0_1)),
                _mm256_shuffle_epi8(vz, dm4_0_2));
        __m256i const t1 = _mm256_or_si256(
                _mm256_or_si256(
                        _mm256_shuffle_epi8(vx, dm4_1_0),
                        _mm256_shuffle_epi8(vy, dm4_1_1)),
                _mm256_shuffle_epi8(vz, dm4_1_2));
        __m256i const t2 = _mm256_or_si256(
                _mm256_or_si256(
                        _mm256_shuffle_epi8(vx, dm4_2_0),
                        _mm256_shuffle_epi8(vy, dm4_2_1)),
                _mm256_shuffle_epi8(vz, dm4_2_2));
        _mm256_storeu_si256(
                (__m256i_u*)(out + i * 12u + 0u),
                _mm256_permute2x128_si256(t0, t1, 0x20));
        _mm256_storeu_si256(
                (__m256i_u*)(out + i * 12u + 32u),
                _mm256_permute2x128_si256(t2, t0, 0x30));
        _mm256_storeu_si256(
                (__m256i_u*)(out + i * 12u + 64u),
                _mm256_permute2x128_si256(t1, t2, 0x31));
    }
#endif
#if ZL_HAS_SSSE3
    for (; i + 4u <= nbElts; i += 4u) {
        __m128i const vx = _mm_loadu_si128((const __m128i_u*)(x + i * 4u));
        __m128i const vy = _mm_loadu_si128((const __m128i_u*)(y + i * 4u));
        __m128i const vz = _mm_loadu_si128((const __m128i_u*)(z + i * 4u));
        __m128i const o0 = _mm_or_si128(
                _mm_or_si128(
                        _mm_shuffle_epi8(vx, dm4_128_0_0),
                        _mm_shuffle_epi8(vy, dm4_128_0_1)),
                _mm_shuffle_epi8(vz, dm4_128_0_2));
        __m128i const o1 = _mm_or_si128(
                _mm_or_si128(
                        _mm_shuffle_epi8(vx, dm4_128_1_0),
                        _mm_shuffle_epi8(vy, dm4_128_1_1)),
                _mm_shuffle_epi8(vz, dm4_128_1_2));
        __m128i const o2 = _mm_or_si128(
                _mm_or_si128(
                        _mm_shuffle_epi8(vx, dm4_128_2_0),
                        _mm_shuffle_epi8(vy, dm4_128_2_1)),
                _mm_shuffle_epi8(vz, dm4_128_2_2));
        _mm_storeu_si128((__m128i_u*)(out + i * 12u + 0u), o0);
        _mm_storeu_si128((__m128i_u*)(out + i * 12u + 16u), o1);
        _mm_storeu_si128((__m128i_u*)(out + i * 12u + 32u), o2);
    }
#elif ZL_HAS_WASM_SIMD128
    for (; i + 4u <= nbElts; i += 4u) {
        v128_t const vx = wasm_v128_load(x + i * 4u);
        v128_t const vy = wasm_v128_load(y + i * 4u);
        v128_t const vz = wasm_v128_load(z + i * 4u);
        v128_t const o0 = wasm_v128_or(
                wasm_v128_or(
                        wasm_i8x16_swizzle(vx, dm4_w_0_0),
                        wasm_i8x16_swizzle(vy, dm4_w_0_1)),
                wasm_i8x16_swizzle(vz, dm4_w_0_2));
        v128_t const o1 = wasm_v128_or(
                wasm_v128_or(
                        wasm_i8x16_swizzle(vx, dm4_w_1_0),
                        wasm_i8x16_swizzle(vy, dm4_w_1_1)),
                wasm_i8x16_swizzle(vz, dm4_w_1_2));
        v128_t const o2 = wasm_v128_or(
                wasm_v128_or(
                        wasm_i8x16_swizzle(vx, dm4_w_2_0),
                        wasm_i8x16_swizzle(vy, dm4_w_2_1)),
                wasm_i8x16_swizzle(vz, dm4_w_2_2));
        wasm_v128_store(out + i * 12u + 0u, o0);
        wasm_v128_store(out + i * 12u + 16u, o1);
        wasm_v128_store(out + i * 12u + 32u, o2);
    }
#endif
    for (; i < nbElts; i++) {
        ZL_write32(out + i * 12u + 0u, ZL_read32(x + i * 4u));
        ZL_write32(out + i * 12u + 4u, ZL_read32(y + i * 4u));
        ZL_write32(out + i * 12u + 8u, ZL_read32(z + i * 4u));
    }
}


static size_t sumArrayST(const size_t array[], size_t arraySize)
{
    size_t total = 0;
    if (arraySize) {
        assert(array != NULL);
    }
    for (size_t n = 0; n < arraySize; n++) {
        total += array[n];
    }
    return total;
}

// Variant dedicated to all @fieldSizes <= 8
// Note : in the future more variant could be created for other sizes,
// typically 4 and 16, maybe 2 and 1, possibly 32, etc.
static size_t rejoin_max8(
        void* restrict dst,
        size_t dstCapacity,
        const void* restrict srcs[],
        const size_t fieldSizes[],
        size_t nbFields,
        const size_t nbElts)
{
    if (dstCapacity) {
        assert(dst != NULL);
    }
    if (!nbElts) {
        return 0;
    }
    assert(srcs != NULL);
    assert(fieldSizes != NULL);
    assert(nbFields);
    size_t minSize = (size_t)(-1);
    for (size_t n = 0; n < nbFields; n++) {
        assert(srcs[n] != NULL);
        assert(fieldSizes[n] <= 8);
        if (fieldSizes[n] < minSize) {
            minSize = fieldSizes[n];
        }
    }
    assert(minSize > 0);
    size_t const nbSafeRounds_precalc[8] = { 8, 4, 3, 2, 2, 2, 2, 1 };
    size_t const nbSafeRounds            = nbSafeRounds_precalc[minSize - 1];

    size_t const structSize = sumArrayST(fieldSizes, nbFields);
    size_t const dstSize    = structSize * nbElts;
    assert(dstSize <= dstCapacity);

    size_t wPos     = 0;
    size_t structNb = 0;
    if (nbElts > nbSafeRounds) {
        for (; structNb < (nbElts - nbSafeRounds); structNb++) {
            for (size_t f = 0; f < nbFields; f++) {
                size_t const fs = fieldSizes[f];
                memcpy((char*)dst + wPos,
                       (const char*)srcs[f] + structNb * fs,
                       8);
                wPos += fs;
            }
        }
    }
    // Finalize, using exact field sizes
    for (; structNb < nbElts; structNb++) {
        for (size_t f = 0; f < nbFields; f++) {
            size_t const fs = fieldSizes[f];
            memcpy((char*)dst + wPos, (const char*)srcs[f] + structNb * fs, fs);
            wPos += fs;
        }
    }
    assert(wPos == dstSize);
    return dstSize;
}

/// Copies @p nbElts fields of @p fieldSize bytes from @p src to @p dst, one
/// every @p stride bytes. Common sizes are copied with constant-size memcpy().
static void copyField(
        char* restrict dst,
        size_t stride,
        const char* restrict src,
        size_t fieldSize,
        size_t nbElts)
{
#define ZS_COPY_FIELD(size)                                 \
    for (size_t e = 0; e < nbElts; e++) {                   \
        memcpy(dst + e * stride, src + e * (size), (size)); \
    }
    switch (fieldSize) {
        case 1:
            ZS_COPY_FIELD(1);
            break;
        case 2:
            ZS_COPY_FIELD(2);
            break;
        case 4:
            ZS_COPY_FIELD(4);
            break;
        case 8:
            ZS_COPY_FIELD(8);
            break;
        case 12:
            ZS_COPY_FIELD(12);
            break;
        case 16:
            ZS_COPY_FIELD(16);
            break;
        case 32:
            ZS_COPY_FIELD(32);
            break;
        case 64:
            ZS_COPY_FIELD(64);
            break;
        case 128:
            ZS_COPY_FIELD(128);
            break;
        default:
            ZS_COPY_FIELD(fieldSize);
            break;
    }
#undef ZS_COPY_FIELD
}

/// Rejoins the fields one by one, over blocks of structures small enough for
/// the block of @p dst to stay in cache between fields.
static void rejoin_byField(
        void* restrict dst,
        const void* restrict srcs[],
        const size_t fieldSizes[],
        size_t nbFields,
        size_t structSize,
        size_t nbElts)
{
    if (structSize == 0) {
        return;
    }
    size_t const blockElts = structSize < 16384 ? 16384 / structSize : 1;
    for (size_t start = 0; start < nbElts; start += blockElts) {
        size_t const n =
                nbElts - start < blockElts ? nbElts - start : blockElts;
        size_t offset = 0;
        for (size_t f = 0; f < nbFields; f++) {
            copyField(
                    (char*)dst + start * structSize + offset,
                    structSize,
                    (const char*)srcs[f] + start * fieldSizes[f],
                    fieldSizes[f],
                    n);
            offset += fieldSizes[f];
        }
    }
}

size_t ZS_dispatchArrayFixedSizeStruct_decode(
        void* restrict dst,
        size_t dstCapacity,
        const void* restrict srcs[],
        const size_t fieldSizes[],
        size_t nbFields,
        const size_t nbElts)
{
    if (dstCapacity) {
        assert(dst != NULL);
    }
    if (!nbElts) {
        return 0;
    }
    assert(srcs != NULL);
    assert(fieldSizes != NULL);
    assert(nbFields);
    size_t maxFieldSize = 0;
    size_t minFieldSize = (size_t)(-1);
    for (size_t n = 0; n < nbFields; n++) {
        assert(srcs[n] != NULL);
        if (fieldSizes[n] > maxFieldSize) {
            maxFieldSize = fieldSizes[n];
        }
        if (fieldSizes[n] < minFieldSize) {
            minFieldSize = fieldSizes[n];
        }
    }
    size_t const structSize = sumArrayST(fieldSizes, nbFields);
    size_t const dstSize    = structSize * nbElts;
    assert(dstSize <= dstCapacity);

    if (nbFields == 3 && fieldSizes[0] == fieldSizes[1]
        && fieldSizes[1] == fieldSizes[2]) {
        if (fieldSizes[0] == 2) {
            rejoin_3ch_2b(dst, srcs[0], srcs[1], srcs[2], nbElts);
            return dstSize;
        }
        if (fieldSizes[0] == 4) {
            rejoin_3ch_4b(dst, srcs[0], srcs[1], srcs[2], nbElts);
            return dstSize;
        }
    }

    // Shortcut to 8-bytes variant
    if (maxFieldSize <= 8 && minFieldSize >= 1) {
        return rejoin_max8(
                dst, dstCapacity, srcs, fieldSizes, nbFields, nbElts);
    }

    rejoin_byField(dst, srcs, fieldSizes, nbFields, structSize, nbElts);
    return dstSize;
}
