// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/hilbert3/decode_hilbert3_binding.h"

#include <stdint.h>

#include "openzl/codecs/hilbert3/hilbert3_table.h"
#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
#include "openzl/shared/mem.h"
#include "openzl/shared/portability.h"

/* Gather 5 3-bit groups at bit positions 0, 9, 18, 27, 36 into bits 0..14. */
static inline uint16_t hilbert3_gather15(uint64_t v)
{
    uint64_t const even = v & 0x0070001C0007ULL;
    uint64_t const odd  = v & 0x000038000E00ULL;
    uint32_t const eg   = (uint32_t)((even * 0x01001001ULL) >> 24) & 0x71C7u;
    uint32_t const og   = (uint32_t)((odd >> 6) | (odd >> 18)) & 0x0E38u;
    return (uint16_t)(eg | og);
}

/* Gather 4 3-bit groups at bit positions 0, 9, 18, 27 into bits 0..11. */
static inline uint32_t hilbert3_gather12(uint64_t v)
{
    return ((uint32_t)v & 7u) | ((uint32_t)(v >> 6) & 0x38u)
            | ((uint32_t)(v >> 12) & 0x1C0u) | ((uint32_t)(v >> 18) & 0xE00u);
}

static inline void writeXYZ(uint8_t* dst, uint32_t m, uint64_t A)
{
    uint16_t const x = (uint16_t)(((m & 1u) << 15) | hilbert3_gather15(A));
    uint16_t const y =
            (uint16_t)((((m >> 1) & 1u) << 15) | hilbert3_gather15(A >> 3));
    uint16_t const z =
            (uint16_t)(((m >> 2) << 15) | hilbert3_gather15(A >> 6));
    ZL_writeLE16(dst, x);
    ZL_writeLE16(dst + 2, y);
    ZL_writeLE16(dst + 4, z);
}

#if ZL_HAS_AVX2
ZL_FORCE_INLINE __m256i hilbert3_pack8_u32(
        uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3,
        uint32_t a4, uint32_t a5, uint32_t a6, uint32_t a7)
{
    __m128i const lo = _mm_insert_epi64(
            _mm_cvtsi64_si128((int64_t)((uint64_t)a0 | ((uint64_t)a1 << 32))),
            (int64_t)((uint64_t)a2 | ((uint64_t)a3 << 32)),
            1);
    __m128i const hi = _mm_insert_epi64(
            _mm_cvtsi64_si128((int64_t)((uint64_t)a4 | ((uint64_t)a5 << 32))),
            (int64_t)((uint64_t)a6 | ((uint64_t)a7 << 32)),
            1);
    return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1);
}

/* Gather 3 3-bit groups at bit positions 0, 9, 18 into bits 0..8 per lane. */
ZL_FORCE_INLINE __m256i hilbert3_gather9_avx2(__m256i v27)
{
    __m256i const m = _mm256_and_si256(v27, _mm256_set1_epi32(0x001C0E07));
    __m256i const p = _mm256_mullo_epi32(m, _mm256_set1_epi32(0x00001041));
    return _mm256_and_si256(_mm256_srli_epi32(p, 12), _mm256_set1_epi32(0x1FF));
}

ZL_FORCE_INLINE void hilbert3_store8_u16_xyz(
        uint8_t* dst, __m256i vx32, __m256i vy32, __m256i vz32)
{
    __m128i const vx = _mm_packus_epi32(
            _mm256_castsi256_si128(vx32), _mm256_extracti128_si256(vx32, 1));
    __m128i const vy = _mm_packus_epi32(
            _mm256_castsi256_si128(vy32), _mm256_extracti128_si256(vy32, 1));
    __m128i const vz = _mm_packus_epi32(
            _mm256_castsi256_si128(vz32), _mm256_extracti128_si256(vz32, 1));
    __m128i const m00 = _mm_setr_epi8(0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5, -1, -1);
    __m128i const m01 = _mm_setr_epi8(-1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5);
    __m128i const m02 = _mm_setr_epi8(-1, -1, -1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1);
    __m128i const m10 = _mm_setr_epi8(-1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1, 10, 11);
    __m128i const m11 = _mm_setr_epi8(-1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1);
    __m128i const m12 = _mm_setr_epi8(4, 5, -1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1);
    __m128i const m20 = _mm_setr_epi8(-1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1, -1, -1);
    __m128i const m21 = _mm_setr_epi8(10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1);
    __m128i const m22 = _mm_setr_epi8(-1, -1, 10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15);
    _mm_storeu_si128((__m128i_u*)(void*)(dst + 0),
            _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(vx, m00), _mm_shuffle_epi8(vy, m01)), _mm_shuffle_epi8(vz, m02)));
    _mm_storeu_si128((__m128i_u*)(void*)(dst + 16),
            _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(vx, m10), _mm_shuffle_epi8(vy, m11)), _mm_shuffle_epi8(vz, m12)));
    _mm_storeu_si128((__m128i_u*)(void*)(dst + 32),
            _mm_or_si128(_mm_or_si128(_mm_shuffle_epi8(vx, m20), _mm_shuffle_epi8(vy, m21)), _mm_shuffle_epi8(vz, m22)));
}
#endif

#if ZL_HAS_AVX2
ZL_FORCE_INLINE void hilbert3_store8_i32_xyz(
        uint8_t* out, __m256i vx, __m256i vy, __m256i vz)
{
    __m256i const dm00 = _mm256_setr_epi8(0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 6, 7, 0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 6, 7);
    __m256i const dm01 = _mm256_setr_epi8(-1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1);
    __m256i const dm02 = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1);
    __m256i const dm10 = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, -1, -1, -1, -1);
    __m256i const dm11 = _mm256_setr_epi8(4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11);
    __m256i const dm12 = _mm256_setr_epi8(-1, -1, -1, -1, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1);
    __m256i const dm20 = _mm256_setr_epi8(-1, -1, -1, -1, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    __m256i const dm21 = _mm256_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 12, 13, 14, 15, -1, -1, -1, -1);
    __m256i const dm22 = _mm256_setr_epi8(8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, 12, 13, 14, 15, 8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, 12, 13, 14, 15);
    __m256i const t0 = _mm256_or_si256(_mm256_or_si256(_mm256_shuffle_epi8(vx, dm00), _mm256_shuffle_epi8(vy, dm01)), _mm256_shuffle_epi8(vz, dm02));
    __m256i const t1 = _mm256_or_si256(_mm256_or_si256(_mm256_shuffle_epi8(vx, dm10), _mm256_shuffle_epi8(vy, dm11)), _mm256_shuffle_epi8(vz, dm12));
    __m256i const t2 = _mm256_or_si256(_mm256_or_si256(_mm256_shuffle_epi8(vx, dm20), _mm256_shuffle_epi8(vy, dm21)), _mm256_shuffle_epi8(vz, dm22));
    _mm256_storeu_si256((__m256i_u*)(void*)(out + 0), _mm256_permute2x128_si256(t0, t1, 0x20));
    _mm256_storeu_si256((__m256i_u*)(void*)(out + 32), _mm256_permute2x128_si256(t2, t0, 0x30));
    _mm256_storeu_si256((__m256i_u*)(void*)(out + 64), _mm256_permute2x128_si256(t1, t2, 0x31));
}
#elif ZL_HAS_WASM_SIMD128
ZL_FORCE_INLINE v128_t hilbert3_gather9_wasm(v128_t v27)
{
    v128_t const m = wasm_v128_and(v27, wasm_i32x4_const_splat(0x001C0E07));
    v128_t const p = wasm_i32x4_mul(m, wasm_i32x4_const_splat(0x00001041));
    return wasm_v128_and(wasm_u32x4_shr(p, 12), wasm_i32x4_const_splat(0x1FF));
}

ZL_FORCE_INLINE void hilbert3_store8_u16_xyz_wasm(
        uint8_t* out,
        v128_t vx0, v128_t vx1,
        v128_t vy0, v128_t vy1,
        v128_t vz0, v128_t vz1)
{
    v128_t const vx  = wasm_u16x8_narrow_i32x4(vx0, vx1);
    v128_t const vy  = wasm_u16x8_narrow_i32x4(vy0, vy1);
    v128_t const vz  = wasm_u16x8_narrow_i32x4(vz0, vz1);
    v128_t const m00 = wasm_i8x16_const(0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5, -1, -1);
    v128_t const m01 = wasm_i8x16_const(-1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1, 4, 5);
    v128_t const m02 = wasm_i8x16_const(-1, -1, -1, -1, 0, 1, -1, -1, -1, -1, 2, 3, -1, -1, -1, -1);
    v128_t const m10 = wasm_i8x16_const(-1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1, 10, 11);
    v128_t const m11 = wasm_i8x16_const(-1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1, -1, -1);
    v128_t const m12 = wasm_i8x16_const(4, 5, -1, -1, -1, -1, 6, 7, -1, -1, -1, -1, 8, 9, -1, -1);
    v128_t const m20 = wasm_i8x16_const(-1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1, -1, -1);
    v128_t const m21 = wasm_i8x16_const(10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15, -1, -1);
    v128_t const m22 = wasm_i8x16_const(-1, -1, 10, 11, -1, -1, -1, -1, 12, 13, -1, -1, -1, -1, 14, 15);
    wasm_v128_store(out + 0, wasm_v128_or(wasm_v128_or(wasm_i8x16_swizzle(vx, m00), wasm_i8x16_swizzle(vy, m01)), wasm_i8x16_swizzle(vz, m02)));
    wasm_v128_store(out + 16, wasm_v128_or(wasm_v128_or(wasm_i8x16_swizzle(vx, m10), wasm_i8x16_swizzle(vy, m11)), wasm_i8x16_swizzle(vz, m12)));
    wasm_v128_store(out + 32, wasm_v128_or(wasm_v128_or(wasm_i8x16_swizzle(vx, m20), wasm_i8x16_swizzle(vy, m21)), wasm_i8x16_swizzle(vz, m22)));
}

ZL_FORCE_INLINE void hilbert3_store4_i32_xyz_wasm(
        uint8_t* out, v128_t vx, v128_t vy, v128_t vz)
{
    v128_t const m00 = wasm_i8x16_const(0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 6, 7);
    v128_t const m01 = wasm_i8x16_const(-1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1, -1, -1, -1, -1);
    v128_t const m02 = wasm_i8x16_const(-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, -1, -1, -1, -1);
    v128_t const m10 = wasm_i8x16_const(-1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, -1, -1, -1, -1);
    v128_t const m11 = wasm_i8x16_const(4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11);
    v128_t const m12 = wasm_i8x16_const(-1, -1, -1, -1, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1);
    v128_t const m20 = wasm_i8x16_const(-1, -1, -1, -1, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1);
    v128_t const m21 = wasm_i8x16_const(-1, -1, -1, -1, -1, -1, -1, -1, 12, 13, 14, 15, -1, -1, -1, -1);
    v128_t const m22 = wasm_i8x16_const(8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, 12, 13, 14, 15);
    wasm_v128_store(out + 0, wasm_v128_or(wasm_v128_or(wasm_i8x16_swizzle(vx, m00), wasm_i8x16_swizzle(vy, m01)), wasm_i8x16_swizzle(vz, m02)));
    wasm_v128_store(out + 16, wasm_v128_or(wasm_v128_or(wasm_i8x16_swizzle(vx, m10), wasm_i8x16_swizzle(vy, m11)), wasm_i8x16_swizzle(vz, m12)));
    wasm_v128_store(out + 32, wasm_v128_or(wasm_v128_or(wasm_i8x16_swizzle(vx, m20), wasm_i8x16_swizzle(vy, m21)), wasm_i8x16_swizzle(vz, m22)));
}
#endif

/* Decode the 15 low key bits of h below the top digit into 5 packed 9-bit
 * groups (x|y<<3|z<<6 per group, group L = levels 3L..3L+2). */
static inline uint64_t decodeLow(uint64_t h, uint32_t s)
{
    uint64_t A = 0;
    for (int L = 4; L >= 0; L--) {
        uint32_t const e = ZL_hilbert3_T3[s * 512 + ((h >> (9 * L)) & 511)];
        A |= (uint64_t)(e & 511) << (9 * L);
        s = e >> 9;
    }
    return A;
}

static void hilbert3Decode(uint8_t* dst, const uint64_t* gaps, size_t n)
{
    uint64_t k = 0;
    size_t i   = 0;
#if ZL_HAS_AVX2
    uint32_t top1Valid = 0;
    uint32_t top1S = 0, top1X = 0, top1Y = 0, top1Z = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t const g0 = gaps[i + 0], g1 = gaps[i + 1], g2 = gaps[i + 2], g3 = gaps[i + 3];
        uint64_t const g4 = gaps[i + 4], g5 = gaps[i + 5], g6 = gaps[i + 6], g7 = gaps[i + 7];
        __m256i vtx, vty, vtz, va;
        if (ZL_LIKELY(top1Valid && (((g0 | g1 | g2 | g3 | g4 | g5 | g6 | g7) >> 36) == 0ULL))) {
            uint64_t const b  = k & 0xFFFFFFFFFULL;
            uint64_t const l0 = b + g0, l1 = l0 + g1, l2 = l1 + g2, l3 = l2 + g3;
            uint64_t const l7 = l3 + g4 + g5 + g6 + g7;
            if (ZL_LIKELY((l7 >> 36) == 0ULL)) {
                k = (k & ~0xFFFFFFFFFULL) | l7;
#define H3_U16_LOW4_ONE(lk, e3o, ao)                                                        uint32_t const e3o     = ZL_hilbert3_T3[top1S | ((uint32_t)((lk) >> 27) & 511u)];   uint32_t const e2_##ao = ZL_hilbert3_T3[(e3o & 0xFE00u) | ((uint32_t)((lk) >> 18) & 511u)];     uint32_t const e1_##ao = ZL_hilbert3_T3[(e2_##ao & 0xFE00u) | ((uint32_t)((lk) >> 9) & 511u)];     uint32_t const e0_##ao = ZL_hilbert3_T3[(e1_##ao & 0xFE00u) | ((uint32_t)(lk) & 511u)];     uint32_t const ao      = ((e2_##ao & 511u) << 18) | ((e1_##ao & 511u) << 9) | (e0_##ao & 511u)
                __m128i ve3_lo, va_lo, ve3_hi, va_hi;
                {
                    H3_U16_LOW4_ONE(l0, e30, a0);
                    H3_U16_LOW4_ONE(l1, e31, a1);
                    H3_U16_LOW4_ONE(l2, e32, a2);
                    H3_U16_LOW4_ONE(l3, e33, a3);
                    ve3_lo = _mm_set_epi32((int)e33, (int)e32, (int)e31, (int)e30);
                    va_lo  = _mm_set_epi32((int)a3, (int)a2, (int)a1, (int)a0);
                }
                {
                    uint64_t const l4 = l3 + gaps[i + 4];
                    uint64_t const l5 = l4 + gaps[i + 5];
                    uint64_t const l6 = l5 + gaps[i + 6];
                    H3_U16_LOW4_ONE(l4, e34, a4);
                    H3_U16_LOW4_ONE(l5, e35, a5);
                    H3_U16_LOW4_ONE(l6, e36, a6);
                    H3_U16_LOW4_ONE(l7, e37, a7);
                    ve3_hi = _mm_set_epi32((int)e37, (int)e36, (int)e35, (int)e34);
                    va_hi  = _mm_set_epi32((int)a7, (int)a6, (int)a5, (int)a4);
                }
#undef H3_U16_LOW4_ONE
                __m256i const ve3 = _mm256_inserti128_si256(_mm256_castsi128_si256(ve3_lo), ve3_hi, 1);
                va  = _mm256_inserti128_si256(_mm256_castsi128_si256(va_lo), va_hi, 1);
                vtx = _mm256_or_si256(_mm256_set1_epi32((int)top1X), _mm256_slli_epi32(_mm256_and_si256(ve3, _mm256_set1_epi32(7)), 9));
                vty = _mm256_or_si256(_mm256_set1_epi32((int)top1Y), _mm256_slli_epi32(_mm256_and_si256(ve3, _mm256_set1_epi32(0x38)), 6));
                vtz = _mm256_or_si256(_mm256_set1_epi32((int)top1Z), _mm256_slli_epi32(_mm256_and_si256(ve3, _mm256_set1_epi32(0x1C0)), 3));
                hilbert3_store8_u16_xyz(
                        dst + 6 * i,
                        _mm256_or_si256(vtx, hilbert3_gather9_avx2(va)),
                        _mm256_or_si256(vty, hilbert3_gather9_avx2(_mm256_srli_epi32(va, 3))),
                        _mm256_or_si256(vtz, hilbert3_gather9_avx2(_mm256_srli_epi32(va, 6))));
                continue;
            }
        }
        {
            uint32_t e4s[8], e3s[8], ds[8], as[8];
            for (int j = 0; j < 8; j++) {
                k += gaps[i + (size_t)j];
                uint32_t const d  = (uint32_t)(k >> 45) & 7u;
                uint32_t const e4 = ZL_hilbert3_T3[((uint32_t)ZL_hilbert3_topS[d] << 9) | ((uint32_t)(k >> 36) & 511u)];
                uint32_t const e3 = ZL_hilbert3_T3[(e4 & 0xFE00u) | ((uint32_t)(k >> 27) & 511u)];
                uint32_t const e2 = ZL_hilbert3_T3[(e3 & 0xFE00u) | ((uint32_t)(k >> 18) & 511u)];
                uint32_t const e1 = ZL_hilbert3_T3[(e2 & 0xFE00u) | ((uint32_t)(k >> 9) & 511u)];
                uint32_t const e0 = ZL_hilbert3_T3[(e1 & 0xFE00u) | ((uint32_t)k & 511u)];
                ds[j]  = d;
                e4s[j] = e4;
                e3s[j] = e3;
                as[j]  = ((e2 & 511u) << 18) | ((e1 & 511u) << 9) | (e0 & 511u);
            }
            __m256i const lutM = _mm256_setr_epi8(0, 1, 3, 2, 6, 7, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 2, 6, 7, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0);
            __m256i const vm   = _mm256_shuffle_epi8(lutM, _mm256_loadu_si256((const __m256i_u*)(const void*)ds));
            __m256i const ve4  = _mm256_loadu_si256((const __m256i_u*)(const void*)e4s);
            __m256i const ve3  = _mm256_loadu_si256((const __m256i_u*)(const void*)e3s);
            va                 = _mm256_loadu_si256((const __m256i_u*)(const void*)as);
            __m256i const vx4  = _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(vm, _mm256_set1_epi32(1)), 15), _mm256_slli_epi32(_mm256_and_si256(ve4, _mm256_set1_epi32(7)), 12));
            __m256i const vy4  = _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(vm, _mm256_set1_epi32(2)), 14), _mm256_slli_epi32(_mm256_and_si256(ve4, _mm256_set1_epi32(0x38)), 9));
            __m256i const vz4  = _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(vm, _mm256_set1_epi32(4)), 13), _mm256_slli_epi32(_mm256_and_si256(ve4, _mm256_set1_epi32(0x1C0)), 6));
            vtx = _mm256_or_si256(vx4, _mm256_slli_epi32(_mm256_and_si256(ve3, _mm256_set1_epi32(7)), 9));
            vty = _mm256_or_si256(vy4, _mm256_slli_epi32(_mm256_and_si256(ve3, _mm256_set1_epi32(0x38)), 6));
            vtz = _mm256_or_si256(vz4, _mm256_slli_epi32(_mm256_and_si256(ve3, _mm256_set1_epi32(0x1C0)), 3));
            uint32_t const m7 = ZL_hilbert3_topM[ds[7]];
            top1Valid = 1;
            top1S = e4s[7] & 0xFE00u;
            top1X = ((m7 & 1u) << 15) | ((e4s[7] & 7u) << 12);
            top1Y = ((m7 & 2u) << 14) | ((e4s[7] & 0x38u) << 9);
            top1Z = ((m7 & 4u) << 13) | ((e4s[7] & 0x1C0u) << 6);
        }
        hilbert3_store8_u16_xyz(
                dst + 6 * i,
                _mm256_or_si256(vtx, hilbert3_gather9_avx2(va)),
                _mm256_or_si256(vty, hilbert3_gather9_avx2(_mm256_srli_epi32(va, 3))),
                _mm256_or_si256(vtz, hilbert3_gather9_avx2(_mm256_srli_epi32(va, 6))));
    }
#elif ZL_HAS_WASM_SIMD128
    uint64_t top2P = UINT64_MAX;
    uint64_t top1P = UINT64_MAX;
    uint32_t top2S = 0, top2X = 0, top2Y = 0, top2Z = 0;
    uint32_t top1S = 0, top1X = 0, top1Y = 0, top1Z = 0;
    for (; i + 8 <= n; i += 8) {
        uint64_t hs[8];
        for (int j = 0; j < 8; j++) {
            hs[j] = k += gaps[i + (size_t)j];
        }
        uint64_t const x07 = (hs[0] ^ hs[1]) | (hs[2] ^ hs[3]) | (hs[4] ^ hs[5])
                | (hs[6] ^ hs[7]) | (hs[0] ^ hs[2]) | (hs[4] ^ hs[6]) | (hs[0] ^ hs[4]);
        uint32_t ss[8], a[8];
        v128_t vtx0, vtx1, vty0, vty1, vtz0, vtz1;
        if (((x07 >> 27) == 0ULL) && ((hs[0] >> 27) == top2P)) {
            for (int j = 0; j < 8; j++) { ss[j] = top2S; }
            vtx0 = vtx1 = wasm_i32x4_splat((int32_t)top2X);
            vty0 = vty1 = wasm_i32x4_splat((int32_t)top2Y);
            vtz0 = vtz1 = wasm_i32x4_splat((int32_t)top2Z);
        } else if (ZL_LIKELY(((x07 >> 36) == 0ULL) && ((hs[0] >> 36) == top1P))) {
            uint32_t txs[8], tys[8], tzs[8];
            for (int j = 0; j < 8; j++) {
                uint32_t const e3 = ZL_hilbert3_T3[top1S | ((uint32_t)(hs[j] >> 27) & 511u)];
                ss[j]  = e3 & 0xFE00u;
                txs[j] = top1X | ((e3 & 7u) << 9);
                tys[j] = top1Y | ((e3 & 0x38u) << 6);
                tzs[j] = top1Z | ((e3 & 0x1C0u) << 3);
            }
            vtx0 = wasm_v128_load(txs); vtx1 = wasm_v128_load(txs + 4);
            vty0 = wasm_v128_load(tys); vty1 = wasm_v128_load(tys + 4);
            vtz0 = wasm_v128_load(tzs); vtz1 = wasm_v128_load(tzs + 4);
            top2P = hs[7] >> 27; top2S = ss[7];
            top2X = txs[7]; top2Y = tys[7]; top2Z = tzs[7];
        } else {
            uint32_t txs[8], tys[8], tzs[8];
            for (int j = 0; j < 8; j++) {
                uint32_t const d  = (uint32_t)(hs[j] >> 45) & 7u;
                uint32_t const m  = ZL_hilbert3_topM[d];
                uint32_t const e4 = ZL_hilbert3_T3[((uint32_t)ZL_hilbert3_topS[d] << 9) | ((uint32_t)(hs[j] >> 36) & 511u)];
                uint32_t const e3 = ZL_hilbert3_T3[(e4 & 0xFE00u) | ((uint32_t)(hs[j] >> 27) & 511u)];
                uint32_t const x4 = ((m & 1u) << 15) | ((e4 & 7u) << 12);
                uint32_t const y4 = ((m & 2u) << 14) | ((e4 & 0x38u) << 9);
                uint32_t const z4 = ((m & 4u) << 13) | ((e4 & 0x1C0u) << 6);
                ss[j]  = e3 & 0xFE00u;
                txs[j] = x4 | ((e3 & 7u) << 9);
                tys[j] = y4 | ((e3 & 0x38u) << 6);
                tzs[j] = z4 | ((e3 & 0x1C0u) << 3);
                if (j == 7) { top1P = hs[7] >> 36; top1S = e4 & 0xFE00u; top1X = x4; top1Y = y4; top1Z = z4; }
            }
            vtx0 = wasm_v128_load(txs); vtx1 = wasm_v128_load(txs + 4);
            vty0 = wasm_v128_load(tys); vty1 = wasm_v128_load(tys + 4);
            vtz0 = wasm_v128_load(tzs); vtz1 = wasm_v128_load(tzs + 4);
            top2P = hs[7] >> 27; top2S = ss[7];
            top2X = txs[7]; top2Y = tys[7]; top2Z = tzs[7];
        }
        for (int j = 0; j < 8; j++) {
            uint32_t const l  = (uint32_t)hs[j];
            uint32_t const e2 = ZL_hilbert3_T3[ss[j] | ((l >> 18) & 511u)];
            uint32_t const e1 = ZL_hilbert3_T3[(e2 & 0xFE00u) | ((l >> 9) & 511u)];
            uint32_t const e0 = ZL_hilbert3_T3[(e1 & 0xFE00u) | (l & 511u)];
            a[j] = ((e2 & 511u) << 18) | ((e1 & 511u) << 9) | (e0 & 511u);
        }
        v128_t const va0 = wasm_v128_load(a);
        v128_t const va1 = wasm_v128_load(a + 4);
        hilbert3_store8_u16_xyz_wasm(
                dst + 6 * i,
                wasm_v128_or(vtx0, hilbert3_gather9_wasm(va0)),
                wasm_v128_or(vtx1, hilbert3_gather9_wasm(va1)),
                wasm_v128_or(vty0, hilbert3_gather9_wasm(wasm_u32x4_shr(va0, 3))),
                wasm_v128_or(vty1, hilbert3_gather9_wasm(wasm_u32x4_shr(va1, 3))),
                wasm_v128_or(vtz0, hilbert3_gather9_wasm(wasm_u32x4_shr(va0, 6))),
                wasm_v128_or(vtz1, hilbert3_gather9_wasm(wasm_u32x4_shr(va1, 6))));
    }
#else
    uint64_t top2P = UINT64_MAX;
    uint64_t top1P = UINT64_MAX;
    uint32_t top2S = 0, top2M = 0, top1S = 0, top1M = 0;
    uint64_t top2A = 0, top1A = 0;
    for (; i + 4 <= n; i += 4) {
        uint64_t const h0 = k += gaps[i + 0];
        uint64_t const h1 = k += gaps[i + 1];
        uint64_t const h2 = k += gaps[i + 2];
        uint64_t const h3 = k += gaps[i + 3];
        uint64_t const x03 = (h0 ^ h1) | (h2 ^ h3) | (h0 ^ h2);
        uint32_t s0, s1, s2, s3, m0, m1, m2, m3;
        uint64_t A0, A1, A2, A3;
        if (((x03 >> 27) == 0ULL) && ((h0 >> 27) == top2P)) {
            s0 = s1 = s2 = s3 = top2S;
            m0 = m1 = m2 = m3 = top2M;
            A0 = A1 = A2 = A3 = top2A;
        } else if (ZL_LIKELY(((x03 >> 36) == 0ULL) && ((h0 >> 36) == top1P))) {
            uint32_t const e0 = ZL_hilbert3_T3[top1S | ((uint32_t)(h0 >> 27) & 511u)];
            uint32_t const e1 = ZL_hilbert3_T3[top1S | ((uint32_t)(h1 >> 27) & 511u)];
            uint32_t const e2 = ZL_hilbert3_T3[top1S | ((uint32_t)(h2 >> 27) & 511u)];
            uint32_t const e3 = ZL_hilbert3_T3[top1S | ((uint32_t)(h3 >> 27) & 511u)];
            s0 = e0 & 0xFE00u; s1 = e1 & 0xFE00u; s2 = e2 & 0xFE00u; s3 = e3 & 0xFE00u;
            m0 = m1 = m2 = m3 = top1M;
            A0 = top1A | ((uint64_t)(e0 & 511u) << 27);
            A1 = top1A | ((uint64_t)(e1 & 511u) << 27);
            A2 = top1A | ((uint64_t)(e2 & 511u) << 27);
            A3 = top1A | ((uint64_t)(e3 & 511u) << 27);
            top2P = h3 >> 27; top2S = s3; top2M = m3; top2A = A3;
        } else {
            uint64_t const hs[4] = { h0, h1, h2, h3 };
            uint32_t ss[4], ms[4];
            uint64_t As[4];
            for (int j = 0; j < 4; j++) {
                uint32_t const d  = (uint32_t)(hs[j] >> 45) & 7u;
                uint32_t const e4 = ZL_hilbert3_T3[((uint32_t)ZL_hilbert3_topS[d] << 9) | ((uint32_t)(hs[j] >> 36) & 511u)];
                uint32_t const e3 = ZL_hilbert3_T3[(e4 & 0xFE00u) | ((uint32_t)(hs[j] >> 27) & 511u)];
                ms[j] = ZL_hilbert3_topM[d];
                ss[j] = e3 & 0xFE00u;
                As[j] = ((uint64_t)(e4 & 511u) << 36) | ((uint64_t)(e3 & 511u) << 27);
                if (j == 3) { top1P = h3 >> 36; top1S = e4 & 0xFE00u; top1M = ms[3]; top1A = (uint64_t)(e4 & 511u) << 36; }
            }
            s0 = ss[0]; s1 = ss[1]; s2 = ss[2]; s3 = ss[3];
            m0 = ms[0]; m1 = ms[1]; m2 = ms[2]; m3 = ms[3];
            A0 = As[0]; A1 = As[1]; A2 = As[2]; A3 = As[3];
            top2P = h3 >> 27; top2S = s3; top2M = m3; top2A = A3;
        }
        for (int L = 2; L >= 0; L--) {
            uint32_t const sh = 9u * (uint32_t)L;
            uint32_t const e0 = ZL_hilbert3_T3[s0 | ((uint32_t)(h0 >> sh) & 511u)];
            uint32_t const e1 = ZL_hilbert3_T3[s1 | ((uint32_t)(h1 >> sh) & 511u)];
            uint32_t const e2 = ZL_hilbert3_T3[s2 | ((uint32_t)(h2 >> sh) & 511u)];
            uint32_t const e3 = ZL_hilbert3_T3[s3 | ((uint32_t)(h3 >> sh) & 511u)];
            A0 |= (uint64_t)(e0 & 511u) << sh; A1 |= (uint64_t)(e1 & 511u) << sh;
            A2 |= (uint64_t)(e2 & 511u) << sh; A3 |= (uint64_t)(e3 & 511u) << sh;
            s0 = e0 & 0xFE00u; s1 = e1 & 0xFE00u; s2 = e2 & 0xFE00u; s3 = e3 & 0xFE00u;
        }
        writeXYZ(dst + 6 * (i + 0), m0, A0);
        writeXYZ(dst + 6 * (i + 1), m1, A1);
        writeXYZ(dst + 6 * (i + 2), m2, A2);
        writeXYZ(dst + 6 * (i + 3), m3, A3);
    }
#endif
    for (; i < n; i++) {
        uint64_t const h = k += gaps[i];
        uint32_t const d = (uint32_t)(h >> 45) & 7u;
        writeXYZ(dst + 6 * i,
                 ZL_hilbert3_topM[d],
                 decodeLow(h, ZL_hilbert3_topS[d]));
    }
}

ZL_Report DI_hilbert3_u16(ZL_Decoder* dictx, const ZL_Input* ins[])
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dictx);
    const ZL_Input* const in = ins[0];
    ZL_ASSERT_EQ(ZL_Input_type(in), ZL_Type_numeric);
    ZL_ERR_IF_NE(ZL_Input_eltWidth(in), 8, corruption);
    size_t const n = ZL_Input_numElts(in);

    ZL_Output* const out = ZL_Decoder_create1OutStream(dictx, n, 6);
    ZL_ERR_IF_NULL(out, allocation);
    hilbert3Decode(
            (uint8_t*)ZL_Output_ptr(out), (const uint64_t*)ZL_Input_ptr(in), n);
    ZL_ERR_IF_ERR(ZL_Output_commit(out, n));
    return ZL_returnSuccess();
}

/* ---- hilbert3_i32 ---- */

typedef struct {
    uint32_t s, x, y, z;
} Hilbert3Top;

/* Digit L (0..9) of the low 90 key bits, 3 levels per digit. */
static uint32_t keyDigit(uint32_t hi, uint64_t lo, int L)
{
    if (L < 7) {
        return (uint32_t)(lo >> (9 * L)) & 511;
    }
    if (L == 7) {
        return (uint32_t)((lo >> 63) | ((uint64_t)hi << 1)) & 511;
    }
    return (hi >> (9 * L - 64)) & 511;
}

/* Walk the top 2 levels and digits 9..4 (key bits >= 36): the state and xyz
 * bits shared by every key with the same hi and lo >> 36. */
ZL_FORCE_NOINLINE Hilbert3Top walkTop(uint32_t hi, uint64_t lo)
{
    uint32_t const d = hi >> 26;
    uint32_t const m = ZL_hilbert3_top2M[d];
    uint32_t s       = ZL_hilbert3_top2S[d];
    uint32_t x = (m & 3u) << 30, y = ((m >> 2) & 3u) << 30, z = (m >> 4) << 30;
#define H3_TOP(L)                                                                do {                                                                             uint32_t const e_ = ZL_hilbert3_T3[s * 512u + keyDigit(hi, lo, L)];          s                 = e_ >> 9;                                                 x |= (e_ & 7u) << (3 * (L));                                                 y |= ((e_ >> 3) & 7u) << (3 * (L));                                          z |= ((e_ >> 6) & 7u) << (3 * (L));                                      } while (0)
    H3_TOP(9);
    H3_TOP(8);
    H3_TOP(7);
    H3_TOP(6);
    H3_TOP(5);
    H3_TOP(4);
#undef H3_TOP
    Hilbert3Top const r = { s, x, y, z };
    return r;
}

#if ZL_HAS_AVX2
/* Single-level Hilbert step tables, derived from ZL_hilbert3_T3:
 * hilbert3_L1[d][state] = next state | xyz << 4 for the 3-bit key digit d
 * (xyz = x | y << 1 | z << 2). Each table is repeated for both 128-bit lanes. */
static const uint8_t hilbert3_L1[8][32] = {
    { 4, 54, 87, 101, 8, 106, 59, 89, 0, 82, 99, 49, 0, 0, 0, 0,
      4, 54, 87, 101, 8, 106, 59, 89, 0, 82, 99, 49, 0, 0, 0, 0 },
    { 40, 27, 121, 74, 64, 35, 113, 18, 20, 71, 117, 38, 0, 0, 0, 0,
      40, 27, 121, 74, 64, 35, 113, 18, 20, 71, 117, 38, 0, 0, 0, 0 },
    { 104, 91, 57, 10, 80, 51, 97, 2, 52, 103, 85, 6, 0, 0, 0, 0,
      104, 91, 57, 10, 80, 51, 97, 2, 52, 103, 85, 6, 0, 0, 0, 0 },
    { 67, 114, 17, 32, 23, 118, 37, 68, 43, 122, 73, 24, 0, 0, 0, 0,
      67, 114, 17, 32, 23, 118, 37, 68, 43, 122, 73, 24, 0, 0, 0, 0 },
    { 83, 98, 1, 48, 55, 86, 5, 100, 107, 58, 9, 88, 0, 0, 0, 0,
      83, 98, 1, 48, 55, 86, 5, 100, 107, 58, 9, 88, 0, 0, 0, 0 },
    { 121, 74, 40, 27, 113, 18, 64, 35, 117, 38, 20, 71, 0, 0, 0, 0,
      121, 74, 40, 27, 113, 18, 64, 35, 117, 38, 20, 71, 0, 0, 0, 0 },
    { 57, 10, 104, 91, 97, 2, 80, 51, 85, 6, 52, 103, 0, 0, 0, 0,
      57, 10, 104, 91, 97, 2, 80, 51, 85, 6, 52, 103, 0, 0, 0, 0 },
    { 22, 36, 69, 119, 42, 72, 25, 123, 66, 16, 33, 115, 0, 0, 0, 0,
      22, 36, 69, 119, 42, 72, 25, 123, 66, 16, 33, 115, 0, 0, 0, 0 }
};

/* Byte vector whose per-byte MSB is bit `bit` of the matching byte of `plane`. */
ZL_FORCE_INLINE __m256i hilbert3_msb(__m256i plane, int bit)
{
    return bit == 7 ? plane : _mm256_slli_epi16(plane, 7 - bit);
}

ZL_FORCE_INLINE void hilbert3_store4_i32(
        uint8_t* out0, uint8_t* out1, __m256i r, __m256i c1, __m256i c2)
{
    /* r holds 2 points per lane as 16-bit words x, y, z, _, x, y, z, _. */
    __m256i const m1 = _mm256_setr_epi8(
            0, 1, -1, -1, 2, 3, -1, -1, 4, 5, -1, -1, 8, 9, -1, -1,
            0, 1, -1, -1, 2, 3, -1, -1, 4, 5, -1, -1, 8, 9, -1, -1);
    __m256i const m2 = _mm256_setr_epi8(
            10, 11, -1, -1, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
            10, 11, -1, -1, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1);
    __m256i const o1 = _mm256_or_si256(_mm256_shuffle_epi8(r, m1), c1);
    __m256i const o2 = _mm256_or_si256(_mm256_shuffle_epi8(r, m2), c2);
    _mm_storeu_si128((__m128i_u*)(void*)out0, _mm256_castsi256_si128(o1));
    _mm_storel_epi64((__m128i_u*)(void*)(out0 + 16), _mm256_castsi256_si128(o2));
    _mm_storeu_si128((__m128i_u*)(void*)out1, _mm256_extracti128_si256(o1, 1));
    _mm_storel_epi64((__m128i_u*)(void*)(out1 + 16), _mm256_extracti128_si256(o2, 1));
}

/* Decodes 32 keys that all share `top` (same key bits >= 36): g[0..31] are
 * plain gaps and low36 is the low 36 key bits before g[0]. The state machine
 * runs one Hilbert level (3 key bits) per step for all 32 keys at once in byte
 * lanes: pshufb maps the state through the 8 per-digit tables and a blend tree
 * picks the digit's result. Lane 0 holds keys 0..15, lane 1 keys 16..31.
 * Returns 0 without writing if a gap or the running key leaves the low 36
 * bits; otherwise writes 32 points and the new low 36 bits. */
ZL_FORCE_INLINE int hilbert3_i32_block32(
        uint8_t* out, const uint64_t* g, uint64_t low36, const Hilbert3Top* top,
        uint64_t* newLow36)
{
    __m256i p[8];
    __m256i orv = _mm256_setzero_si256();
    for (int q = 0; q < 8; q++) {
        p[q] = _mm256_inserti128_si256(
                _mm256_castsi128_si256(_mm_loadu_si128((const __m128i_u*)(const void*)(g + 2 * q))),
                _mm_loadu_si128((const __m128i_u*)(const void*)(g + 16 + 2 * q)), 1);
        orv = _mm256_or_si256(orv, p[q]);
    }
    if (ZL_UNLIKELY(!_mm256_testz_si256(orv, _mm256_set1_epi64x(~0xFFFFFFFFFLL)))) {
        return 0;
    }
    /* In-lane prefix sums; lane 1 gets the lane 0 total added afterwards. */
    __m256i carry = _mm256_setr_epi64x((long long)low36, (long long)low36, 0, 0);
    for (int q = 0; q < 8; q++) {
        p[q]  = _mm256_add_epi64(_mm256_add_epi64(p[q], _mm256_slli_si256(p[q], 8)), carry);
        carry = _mm256_unpackhi_epi64(p[q], p[q]);
    }
    uint64_t const last = (uint64_t)_mm_cvtsi128_si64(_mm256_castsi256_si128(carry))
            + (uint64_t)_mm_cvtsi128_si64(_mm256_extracti128_si256(carry, 1));
    if (ZL_UNLIKELY((last >> 36) != 0ULL)) {
        return 0;
    }
    *newLow36 = last;
    __m256i const half = _mm256_permute2x128_si256(carry, carry, 0x08);

    /* Byte planes b[k]: byte k of every key. */
    __m256i b[5];
    {
        __m256i const grp = _mm256_setr_epi8(
                0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
        __m256i w[4], hi[4];
        for (int q = 0; q < 4; q++) {
            __m256 const a0 = _mm256_castsi256_ps(_mm256_add_epi64(p[2 * q], half));
            __m256 const a1 = _mm256_castsi256_ps(_mm256_add_epi64(p[2 * q + 1], half));
            w[q]  = _mm256_shuffle_epi8(_mm256_castps_si256(_mm256_shuffle_ps(a0, a1, 0x88)), grp);
            hi[q] = _mm256_castps_si256(_mm256_shuffle_ps(a0, a1, 0xDD));
        }
        __m256i const t0 = _mm256_unpacklo_epi32(w[0], w[1]);
        __m256i const t1 = _mm256_unpackhi_epi32(w[0], w[1]);
        __m256i const t2 = _mm256_unpacklo_epi32(w[2], w[3]);
        __m256i const t3 = _mm256_unpackhi_epi32(w[2], w[3]);
        b[0] = _mm256_unpacklo_epi64(t0, t2);
        b[1] = _mm256_unpackhi_epi64(t0, t2);
        b[2] = _mm256_unpacklo_epi64(t1, t3);
        b[3] = _mm256_unpackhi_epi64(t1, t3);
        b[4] = _mm256_packus_epi16(_mm256_packus_epi32(hi[0], hi[1]), _mm256_packus_epi32(hi[2], hi[3]));
    }

    __m256i s = _mm256_set1_epi8((char)top->s);
    /* ax/ay/az[grp]: 4 levels each; x bits at 0..3, y at 1..4, z at 2..5. */
    __m256i ax[3], ay[3], az[3];
    for (int q = 0; q < 3; q++) {
        ax[q] = ay[q] = az[q] = _mm256_setzero_si256();
    }
    for (int L = 11; L >= 0; L--) {
        __m256i const m0 = hilbert3_msb(b[(3 * L) >> 3], (3 * L) & 7);
        __m256i const m1 = hilbert3_msb(b[(3 * L + 1) >> 3], (3 * L + 1) & 7);
        __m256i const m2 = hilbert3_msb(b[(3 * L + 2) >> 3], (3 * L + 2) & 7);
#define H3_L1(d) _mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i_u*)(const void*)hilbert3_L1[d]), s)
        __m256i const r01 = _mm256_blendv_epi8(H3_L1(0), H3_L1(1), m0);
        __m256i const r23 = _mm256_blendv_epi8(H3_L1(2), H3_L1(3), m0);
        __m256i const r45 = _mm256_blendv_epi8(H3_L1(4), H3_L1(5), m0);
        __m256i const r67 = _mm256_blendv_epi8(H3_L1(6), H3_L1(7), m0);
#undef H3_L1
        s = _mm256_blendv_epi8(
                _mm256_blendv_epi8(r01, r23, m1), _mm256_blendv_epi8(r45, r67, m1), m2);
        int const j       = L & 3;
        __m256i const t   = _mm256_srli_epi16(s, 4 - j);
        ax[L >> 2] = _mm256_or_si256(ax[L >> 2], _mm256_and_si256(t, _mm256_set1_epi8((char)(1 << j))));
        ay[L >> 2] = _mm256_or_si256(ay[L >> 2], _mm256_and_si256(t, _mm256_set1_epi8((char)(2 << j))));
        az[L >> 2] = _mm256_or_si256(az[L >> 2], _mm256_and_si256(t, _mm256_set1_epi8((char)(4 << j))));
    }
    __m256i const xlo = _mm256_or_si256(ax[0], _mm256_slli_epi16(ax[1], 4));
    __m256i const ylo = _mm256_or_si256(_mm256_srli_epi16(ay[0], 1), _mm256_slli_epi16(ay[1], 3));
    __m256i const zlo = _mm256_or_si256(_mm256_srli_epi16(az[0], 2), _mm256_slli_epi16(az[1], 2));
    __m256i const xhi = ax[2];
    __m256i const yhi = _mm256_srli_epi16(ay[2], 1);
    __m256i const zhi = _mm256_srli_epi16(az[2], 2);

    uint32_t const tx = top->x ^ 0x80000000u;
    uint32_t const ty = top->y ^ 0x80000000u;
    uint32_t const tz = top->z ^ 0x80000000u;
    __m256i const c1  = _mm256_setr_epi32((int)tx, (int)ty, (int)tz, (int)tx, (int)tx, (int)ty, (int)tz, (int)tx);
    __m256i const c2  = _mm256_setr_epi32((int)ty, (int)tz, 0, 0, (int)ty, (int)tz, 0, 0);
    for (int hh = 0; hh < 2; hh++) {
        /* 16-bit x, y, z of points 8*hh.. (lane 0) and 16+8*hh.. (lane 1) */
        __m256i const x16 = hh ? _mm256_unpackhi_epi8(xlo, xhi) : _mm256_unpacklo_epi8(xlo, xhi);
        __m256i const y16 = hh ? _mm256_unpackhi_epi8(ylo, yhi) : _mm256_unpacklo_epi8(ylo, yhi);
        __m256i const z16 = hh ? _mm256_unpackhi_epi8(zlo, zhi) : _mm256_unpacklo_epi8(zlo, zhi);
        __m256i const xy0 = _mm256_unpacklo_epi16(x16, y16);
        __m256i const xy1 = _mm256_unpackhi_epi16(x16, y16);
        __m256i const z0  = _mm256_unpacklo_epi16(z16, z16);
        __m256i const z1  = _mm256_unpackhi_epi16(z16, z16);
        uint8_t* const o0 = out + 12 * (8 * hh);
        uint8_t* const o1 = out + 12 * (16 + 8 * hh);
        hilbert3_store4_i32(o0 + 0, o1 + 0, _mm256_unpacklo_epi32(xy0, z0), c1, c2);
        hilbert3_store4_i32(o0 + 24, o1 + 24, _mm256_unpackhi_epi32(xy0, z0), c1, c2);
        hilbert3_store4_i32(o0 + 48, o1 + 48, _mm256_unpacklo_epi32(xy1, z1), c1, c2);
        hilbert3_store4_i32(o0 + 72, o1 + 72, _mm256_unpackhi_epi32(xy1, z1), c1, c2);
    }
    return 1;
}
#endif

static inline void writeXYZ32(uint8_t* dst, uint32_t x, uint32_t y, uint32_t z)
{
    ZL_writeLE32(dst, x ^ 0x80000000u);
    ZL_writeLE32(dst + 4, y ^ 0x80000000u);
    ZL_writeLE32(dst + 8, z ^ 0x80000000u);
}

#define H3_STEP_KEY(g, idx, hi, lo, topHi, topLo, top)               do {                                                                 uint64_t const v_ = (g)[idx];                                    if (ZL_UNLIKELY(v_ == UINT64_MAX)) {                                 (hi) = (uint32_t)(g)[(idx) + 1];                                 (lo) = (g)[(idx) + 2];                                           (idx) += 3;                                                  } else {                                                             uint64_t const nl_ = (lo) + v_;                                  (hi) += nl_ < (lo);                                              (lo) = nl_;                                                      (idx) += 1;                                                  }                                                                if (ZL_UNLIKELY((hi) != (topHi) || ((lo) >> 36) != (topLo))) {             (topHi) = (hi);                                                  (topLo) = (lo) >> 36;                                            (top)   = walkTop((hi), (lo));                               }                                                            } while (0)

static size_t hilbert3DecodeI32(uint8_t* dst, const uint64_t* g, size_t n)
{
    Hilbert3Top top = { 0, 0, 0, 0 };
    uint32_t topHi  = 0;
    uint64_t topLo  = UINT64_MAX;
    uint32_t hi     = 0;
    uint64_t lo     = 0;
    size_t np = 0, i = 0;

#if ZL_HAS_AVX2
    size_t skip32 = 0, fail32 = 0;
    while (i + 24 <= n) {
        if (skip32 > 0) {
            skip32--;
        } else if (i + 32 <= n && topLo != UINT64_MAX) {
            uint64_t newLow;
            if (ZL_LIKELY(hilbert3_i32_block32(dst + 12 * np, g + i, lo & 0xFFFFFFFFFULL, &top, &newLow))) {
                lo = (lo & ~0xFFFFFFFFFULL) | newLow;
                i += 32;
                np += 32;
                fail32 = 0;
                continue;
            }
            /* Let the 8-key path below handle the block boundary; back off
             * when 32-key blocks keep failing. */
            fail32 = fail32 < 6 ? fail32 + 1 : 6;
            skip32 = ((size_t)1 << fail32) >> 2;
        }
        uint64_t const g0 = g[i + 0], g1 = g[i + 1], g2 = g[i + 2], g3 = g[i + 3];
        uint64_t const g4 = g[i + 4], g5 = g[i + 5], g6 = g[i + 6], g7 = g[i + 7];
        if (ZL_LIKELY(topLo != UINT64_MAX && (((g0 | g1 | g2 | g3 | g4 | g5 | g6 | g7) >> 36) == 0ULL))) {
            uint64_t const b  = lo & 0xFFFFFFFFFULL;
            uint64_t const l0 = b + g0, l1 = l0 + g1, l2 = l1 + g2, l3 = l2 + g3;
            uint64_t const l4 = l3 + g4, l5 = l4 + g5, l6 = l5 + g6, l7 = l6 + g7;
            if (ZL_LIKELY((l7 >> 36) == 0ULL)) {
                lo = (lo & ~0xFFFFFFFFFULL) | l7;
                i += 8;
                uint32_t const ts = top.s << 9;
#define H3_I32_LOW4(lk, e3o, ao)                                                                uint32_t const e3o     = ZL_hilbert3_T3[ts | ((uint32_t)((lk) >> 27) & 511u)];              uint32_t const e2_##ao = ZL_hilbert3_T3[(e3o & 0xFE00u) | ((uint32_t)((lk) >> 18) & 511u)];     uint32_t const e1_##ao = ZL_hilbert3_T3[(e2_##ao & 0xFE00u) | ((uint32_t)((lk) >> 9) & 511u)];     uint32_t const e0_##ao = ZL_hilbert3_T3[(e1_##ao & 0xFE00u) | ((uint32_t)(lk) & 511u)];     uint32_t const ao      = ((e2_##ao & 511u) << 18) | ((e1_##ao & 511u) << 9) | (e0_##ao & 511u)
                H3_I32_LOW4(l0, e30, a0); H3_I32_LOW4(l1, e31, a1); H3_I32_LOW4(l2, e32, a2); H3_I32_LOW4(l3, e33, a3);
                H3_I32_LOW4(l4, e34, a4); H3_I32_LOW4(l5, e35, a5); H3_I32_LOW4(l6, e36, a6); H3_I32_LOW4(l7, e37, a7);
#undef H3_I32_LOW4
                __m256i const ve3 = hilbert3_pack8_u32(e30, e31, e32, e33, e34, e35, e36, e37);
                __m256i const va  = hilbert3_pack8_u32(a0, a1, a2, a3, a4, a5, a6, a7);
                hilbert3_store8_i32_xyz(
                        dst + 12 * np,
                        _mm256_or_si256(_mm256_set1_epi32((int)(top.x ^ 0x80000000u)), _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(ve3, _mm256_set1_epi32(7)), 9), hilbert3_gather9_avx2(va))),
                        _mm256_or_si256(_mm256_set1_epi32((int)(top.y ^ 0x80000000u)), _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(ve3, _mm256_set1_epi32(0x38)), 6), hilbert3_gather9_avx2(_mm256_srli_epi32(va, 3)))),
                        _mm256_or_si256(_mm256_set1_epi32((int)(top.z ^ 0x80000000u)), _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(ve3, _mm256_set1_epi32(0x1C0)), 3), hilbert3_gather9_avx2(_mm256_srli_epi32(va, 6)))));
                np += 8;
                continue;
            }
        }
        uint32_t txa[8], tya[8], tza[8], e3a[8], aa[8];
        for (int j = 0; j < 8; j++) {
            H3_STEP_KEY(g, i, hi, lo, topHi, topLo, top);
            txa[j] = top.x ^ 0x80000000u;
            tya[j] = top.y ^ 0x80000000u;
            tza[j] = top.z ^ 0x80000000u;
            uint32_t const l  = (uint32_t)lo;
            uint32_t const e3 = ZL_hilbert3_T3[(top.s << 9) | ((uint32_t)(lo >> 27) & 511u)];
            uint32_t const e2 = ZL_hilbert3_T3[(e3 & 0xFE00u) | ((l >> 18) & 511u)];
            uint32_t const e1 = ZL_hilbert3_T3[(e2 & 0xFE00u) | ((l >> 9) & 511u)];
            uint32_t const e0 = ZL_hilbert3_T3[(e1 & 0xFE00u) | (l & 511u)];
            e3a[j] = e3;
            aa[j]  = ((e2 & 511u) << 18) | ((e1 & 511u) << 9) | (e0 & 511u);
        }
        __m256i const ve3 = hilbert3_pack8_u32(e3a[0], e3a[1], e3a[2], e3a[3], e3a[4], e3a[5], e3a[6], e3a[7]);
        __m256i const va  = hilbert3_pack8_u32(aa[0], aa[1], aa[2], aa[3], aa[4], aa[5], aa[6], aa[7]);
        hilbert3_store8_i32_xyz(
                dst + 12 * np,
                _mm256_or_si256(_mm256_loadu_si256((const __m256i_u*)(const void*)txa), _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(ve3, _mm256_set1_epi32(7)), 9), hilbert3_gather9_avx2(va))),
                _mm256_or_si256(_mm256_loadu_si256((const __m256i_u*)(const void*)tya), _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(ve3, _mm256_set1_epi32(0x38)), 6), hilbert3_gather9_avx2(_mm256_srli_epi32(va, 3)))),
                _mm256_or_si256(_mm256_loadu_si256((const __m256i_u*)(const void*)tza), _mm256_or_si256(_mm256_slli_epi32(_mm256_and_si256(ve3, _mm256_set1_epi32(0x1C0)), 3), hilbert3_gather9_avx2(_mm256_srli_epi32(va, 6)))));
        np += 8;
    }
#elif ZL_HAS_WASM_SIMD128
    while (i + 12 <= n) {
        uint32_t txa[4], tya[4], tza[4], e3a[4], aa[4];
        v128_t vtx, vty, vtz;
        uint64_t const g0 = g[i + 0], g1 = g[i + 1], g2 = g[i + 2], g3 = g[i + 3];
        if (ZL_LIKELY(topLo != UINT64_MAX && (((g0 | g1 | g2 | g3) >> 36) == 0ULL) && ((((lo & 0xFFFFFFFFFULL) + g0 + g1 + g2 + g3) >> 36) == 0ULL))) {
            uint64_t ls[4];
            ls[0] = (lo & 0xFFFFFFFFFULL) + g0;
            ls[1] = ls[0] + g1;
            ls[2] = ls[1] + g2;
            ls[3] = ls[2] + g3;
            lo    = (lo & ~0xFFFFFFFFFULL) | ls[3];
            i += 4;
            for (int j = 0; j < 4; j++) {
                uint32_t const l  = (uint32_t)ls[j];
                uint32_t const e3 = ZL_hilbert3_T3[(top.s << 9) | ((uint32_t)(ls[j] >> 27) & 511u)];
                uint32_t const e2 = ZL_hilbert3_T3[(e3 & 0xFE00u) | ((l >> 18) & 511u)];
                uint32_t const e1 = ZL_hilbert3_T3[(e2 & 0xFE00u) | ((l >> 9) & 511u)];
                uint32_t const e0 = ZL_hilbert3_T3[(e1 & 0xFE00u) | (l & 511u)];
                e3a[j] = e3;
                aa[j]  = ((e2 & 511u) << 18) | ((e1 & 511u) << 9) | (e0 & 511u);
            }
            vtx = wasm_i32x4_splat((int32_t)(top.x ^ 0x80000000u));
            vty = wasm_i32x4_splat((int32_t)(top.y ^ 0x80000000u));
            vtz = wasm_i32x4_splat((int32_t)(top.z ^ 0x80000000u));
        } else {
            for (int j = 0; j < 4; j++) {
                H3_STEP_KEY(g, i, hi, lo, topHi, topLo, top);
                txa[j] = top.x ^ 0x80000000u; tya[j] = top.y ^ 0x80000000u; tza[j] = top.z ^ 0x80000000u;
                uint32_t const l  = (uint32_t)lo;
                uint32_t const e3 = ZL_hilbert3_T3[(top.s << 9) | ((uint32_t)(lo >> 27) & 511u)];
                uint32_t const e2 = ZL_hilbert3_T3[(e3 & 0xFE00u) | ((l >> 18) & 511u)];
                uint32_t const e1 = ZL_hilbert3_T3[(e2 & 0xFE00u) | ((l >> 9) & 511u)];
                uint32_t const e0 = ZL_hilbert3_T3[(e1 & 0xFE00u) | (l & 511u)];
                e3a[j] = e3;
                aa[j]  = ((e2 & 511u) << 18) | ((e1 & 511u) << 9) | (e0 & 511u);
            }
            vtx = wasm_v128_load(txa); vty = wasm_v128_load(tya); vtz = wasm_v128_load(tza);
        }
        v128_t const ve3 = wasm_v128_load(e3a);
        v128_t const va  = wasm_v128_load(aa);
        hilbert3_store4_i32_xyz_wasm(
                dst + 12 * np,
                wasm_v128_or(vtx, wasm_v128_or(wasm_i32x4_shl(wasm_v128_and(ve3, wasm_i32x4_const_splat(7)), 9), hilbert3_gather9_wasm(va))),
                wasm_v128_or(vty, wasm_v128_or(wasm_i32x4_shl(wasm_v128_and(ve3, wasm_i32x4_const_splat(0x38)), 6), hilbert3_gather9_wasm(wasm_u32x4_shr(va, 3)))),
                wasm_v128_or(vtz, wasm_v128_or(wasm_i32x4_shl(wasm_v128_and(ve3, wasm_i32x4_const_splat(0x1C0)), 3), hilbert3_gather9_wasm(wasm_u32x4_shr(va, 6)))));
        np += 4;
    }
#else
    while (i + 12 <= n) {
        uint32_t s[4], tx[4], ty[4], tz[4];
        uint64_t l[4], A[4] = { 0, 0, 0, 0 };
        for (int j = 0; j < 4; j++) {
            H3_STEP_KEY(g, i, hi, lo, topHi, topLo, top);
            s[j] = top.s << 9; tx[j] = top.x; ty[j] = top.y; tz[j] = top.z; l[j] = lo;
        }
        for (int L = 3; L >= 0; L--) {
            for (int j = 0; j < 4; j++) {
                uint32_t const e = ZL_hilbert3_T3[s[j] | ((uint32_t)(l[j] >> (9 * L)) & 511u)];
                s[j] = e & 0xFE00u;
                A[j] |= (uint64_t)(e & 511u) << (9 * L);
            }
        }
        for (int j = 0; j < 4; j++) {
            writeXYZ32(dst + 12 * (np + (size_t)j),
                       tx[j] | hilbert3_gather12(A[j]),
                       ty[j] | hilbert3_gather12(A[j] >> 3),
                       tz[j] | hilbert3_gather12(A[j] >> 6));
        }
        np += 4;
    }
#endif
    while (i < n) {
        if (g[i] == UINT64_MAX && i + 3 > n) {
            return SIZE_MAX;
        }
        H3_STEP_KEY(g, i, hi, lo, topHi, topLo, top);
        uint32_t s = top.s << 9;
        uint64_t A = 0;
        for (int L = 3; L >= 0; L--) {
            uint32_t const e = ZL_hilbert3_T3[s | ((uint32_t)(lo >> (9 * L)) & 511u)];
            s = e & 0xFE00u;
            A |= (uint64_t)(e & 511u) << (9 * L);
        }
        writeXYZ32(dst + 12 * np,
                   top.x | hilbert3_gather12(A),
                   top.y | hilbert3_gather12(A >> 3),
                   top.z | hilbert3_gather12(A >> 6));
        np++;
    }
#undef H3_STEP_KEY
    return np;
}

ZL_Report DI_hilbert3_i32(ZL_Decoder* dictx, const ZL_Input* ins[])
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dictx);
    const ZL_Input* const in = ins[0];
    ZL_ASSERT_EQ(ZL_Input_type(in), ZL_Type_numeric);
    ZL_ERR_IF_NE(ZL_Input_eltWidth(in), 8, corruption);
    size_t const n = ZL_Input_numElts(in);

    ZL_Output* const out = ZL_Decoder_create1OutStream(dictx, n, 12);
    ZL_ERR_IF_NULL(out, allocation);
    size_t const np = hilbert3DecodeI32(
            (uint8_t*)ZL_Output_ptr(out), (const uint64_t*)ZL_Input_ptr(in), n);
    ZL_ERR_IF_EQ(np, SIZE_MAX, corruption, "hilbert3_i32: truncated escape");
    ZL_ERR_IF_ERR(ZL_Output_commit(out, np));
    return ZL_returnSuccess();
}
