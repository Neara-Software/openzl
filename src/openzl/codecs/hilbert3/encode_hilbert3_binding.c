// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/hilbert3/encode_hilbert3_binding.h"

#include <stdint.h>

#define ZL_HILBERT3_ENCODE_ONLY
#include "openzl/codecs/hilbert3/hilbert3_table.h"
#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
#include "openzl/shared/mem.h"
#include "openzl/shared/portability.h"

/* Scatter 15 bits (5 groups of 3 bits) into bit positions 0..2, 9..11, 18..20,
 * 27..29, 36..38. */
static inline uint64_t hilbert3_scatter15(uint32_t v)
{
    uint64_t const even = (uint64_t)(v & 0x71C7u);
    uint64_t const odd  = (uint64_t)(v & 0x0E38u);
    uint64_t const es   = (even * 0x01001001ULL) & 0x0070001C0007ULL;
    uint64_t const os   = ((odd << 6) | (odd << 18)) & 0x000038000E00ULL;
    return es | os;
}

static inline uint64_t hilbert3_packA15(uint32_t x, uint32_t y, uint32_t z)
{
    return hilbert3_scatter15(x & 0x7FFFu)
            | (hilbert3_scatter15(y & 0x7FFFu) << 3)
            | (hilbert3_scatter15(z & 0x7FFFu) << 6);
}

#if ZL_HAS_AVX2
ZL_FORCE_INLINE __m256i hilbert3_scatter9_one_avx2(__m256i v)
{
    return _mm256_or_si256(
            _mm256_and_si256(v, _mm256_set1_epi32(7)),
            _mm256_or_si256(
                    _mm256_slli_epi32(_mm256_and_si256(v, _mm256_set1_epi32(0x38)), 6),
                    _mm256_slli_epi32(_mm256_and_si256(v, _mm256_set1_epi32(0x1C0)), 12)));
}

ZL_FORCE_INLINE __m256i hilbert3_scatter9_xyz_avx2(__m256i vx, __m256i vy, __m256i vz)
{
    return _mm256_or_si256(
            hilbert3_scatter9_one_avx2(vx),
            _mm256_or_si256(
                    _mm256_slli_epi32(hilbert3_scatter9_one_avx2(vy), 3),
                    _mm256_slli_epi32(hilbert3_scatter9_one_avx2(vz), 6)));
}

ZL_FORCE_INLINE __m256i hilbert3_scatter6_one_avx2(__m256i v)
{
    return _mm256_or_si256(
            _mm256_and_si256(v, _mm256_set1_epi32(7)),
            _mm256_slli_epi32(_mm256_and_si256(v, _mm256_set1_epi32(0x38)), 6));
}

ZL_FORCE_INLINE __m256i hilbert3_scatter6_xyz_avx2(__m256i vx, __m256i vy, __m256i vz)
{
    return _mm256_or_si256(
            hilbert3_scatter6_one_avx2(vx),
            _mm256_or_si256(
                    _mm256_slli_epi32(hilbert3_scatter6_one_avx2(vy), 3),
                    _mm256_slli_epi32(hilbert3_scatter6_one_avx2(vz), 6)));
}

ZL_FORCE_INLINE void hilbert3_load8_u16_xyz(
        const uint8_t* src, __m256i* vx, __m256i* vy, __m256i* vz)
{
    __m128i const a0  = _mm_loadu_si128((const __m128i_u*)(const void*)(src + 0));
    __m128i const a1  = _mm_loadu_si128((const __m128i_u*)(const void*)(src + 16));
    __m128i const a2  = _mm_loadu_si128((const __m128i_u*)(const void*)(src + 32));
    __m128i const x16 = _mm_or_si128(
            _mm_or_si128(
                    _mm_shuffle_epi8(a0, _mm_setr_epi8(0, 1, 6, 7, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)),
                    _mm_shuffle_epi8(a1, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 2, 3, 8, 9, 14, 15, -1, -1, -1, -1))),
            _mm_shuffle_epi8(a2, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 10, 11)));
    __m128i const y16 = _mm_or_si128(
            _mm_or_si128(
                    _mm_shuffle_epi8(a0, _mm_setr_epi8(2, 3, 8, 9, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)),
                    _mm_shuffle_epi8(a1, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, 4, 5, 10, 11, -1, -1, -1, -1, -1, -1))),
            _mm_shuffle_epi8(a2, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 6, 7, 12, 13)));
    __m128i const z16 = _mm_or_si128(
            _mm_or_si128(
                    _mm_shuffle_epi8(a0, _mm_setr_epi8(4, 5, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)),
                    _mm_shuffle_epi8(a1, _mm_setr_epi8(-1, -1, -1, -1, 0, 1, 6, 7, 12, 13, -1, -1, -1, -1, -1, -1))),
            _mm_shuffle_epi8(a2, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3, 8, 9, 14, 15)));
    *vx = _mm256_cvtepu16_epi32(x16);
    *vy = _mm256_cvtepu16_epi32(y16);
    *vz = _mm256_cvtepu16_epi32(z16);
}

ZL_FORCE_INLINE void hilbert3_deint4_i32(
        const uint8_t* s, __m128i* ox, __m128i* oy, __m128i* oz)
{
    __m128i const b0 = _mm_loadu_si128((const __m128i_u*)(const void*)(s + 0));
    __m128i const b1 = _mm_loadu_si128((const __m128i_u*)(const void*)(s + 16));
    __m128i const b2 = _mm_loadu_si128((const __m128i_u*)(const void*)(s + 32));
    *ox = _mm_or_si128(
            _mm_shuffle_epi8(b0, _mm_setr_epi8(0, 1, 2, 3, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1)),
            _mm_or_si128(
                    _mm_shuffle_epi8(b1, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, -1, -1, -1, -1)),
                    _mm_shuffle_epi8(b2, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 6, 7))));
    *oy = _mm_or_si128(
            _mm_shuffle_epi8(b0, _mm_setr_epi8(4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)),
            _mm_or_si128(
                    _mm_shuffle_epi8(b1, _mm_setr_epi8(-1, -1, -1, -1, 0, 1, 2, 3, 12, 13, 14, 15, -1, -1, -1, -1)),
                    _mm_shuffle_epi8(b2, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11))));
    *oz = _mm_or_si128(
            _mm_shuffle_epi8(b0, _mm_setr_epi8(8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)),
            _mm_or_si128(
                    _mm_shuffle_epi8(b1, _mm_setr_epi8(-1, -1, -1, -1, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1)),
                    _mm_shuffle_epi8(b2, _mm_setr_epi8(-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 12, 13, 14, 15))));
}

ZL_FORCE_INLINE void hilbert3_load8_i32_xyz_flipped(
        const uint8_t* src, __m256i* vx, __m256i* vy, __m256i* vz)
{
    __m128i x0, y0, z0, x1, y1, z1;
    hilbert3_deint4_i32(src, &x0, &y0, &z0);
    hilbert3_deint4_i32(src + 48, &x1, &y1, &z1);
    __m256i const sign = _mm256_set1_epi32((int)0x80000000u);
    *vx = _mm256_xor_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(x0), x1, 1), sign);
    *vy = _mm256_xor_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(y0), y1, 1), sign);
    *vz = _mm256_xor_si256(_mm256_inserti128_si256(_mm256_castsi128_si256(z0), z1, 1), sign);
}
#elif ZL_HAS_WASM_SIMD128
ZL_FORCE_INLINE v128_t hilbert3_scatter9_one_wasm(v128_t v)
{
    return wasm_v128_or(
            wasm_v128_and(v, wasm_i32x4_const_splat(7)),
            wasm_v128_or(
                    wasm_i32x4_shl(wasm_v128_and(v, wasm_i32x4_const_splat(0x38)), 6),
                    wasm_i32x4_shl(wasm_v128_and(v, wasm_i32x4_const_splat(0x1C0)), 12)));
}

ZL_FORCE_INLINE v128_t hilbert3_scatter9_xyz_wasm(v128_t vx, v128_t vy, v128_t vz)
{
    return wasm_v128_or(
            hilbert3_scatter9_one_wasm(vx),
            wasm_v128_or(
                    wasm_i32x4_shl(hilbert3_scatter9_one_wasm(vy), 3),
                    wasm_i32x4_shl(hilbert3_scatter9_one_wasm(vz), 6)));
}

ZL_FORCE_INLINE v128_t hilbert3_scatter6_one_wasm(v128_t v)
{
    return wasm_v128_or(
            wasm_v128_and(v, wasm_i32x4_const_splat(7)),
            wasm_i32x4_shl(wasm_v128_and(v, wasm_i32x4_const_splat(0x38)), 6));
}

ZL_FORCE_INLINE v128_t hilbert3_scatter6_xyz_wasm(v128_t vx, v128_t vy, v128_t vz)
{
    return wasm_v128_or(
            hilbert3_scatter6_one_wasm(vx),
            wasm_v128_or(
                    wasm_i32x4_shl(hilbert3_scatter6_one_wasm(vy), 3),
                    wasm_i32x4_shl(hilbert3_scatter6_one_wasm(vz), 6)));
}

ZL_FORCE_INLINE void hilbert3_load8_u16_xyz_wasm(
        const uint8_t* src,
        v128_t* vx0, v128_t* vx1,
        v128_t* vy0, v128_t* vy1,
        v128_t* vz0, v128_t* vz1)
{
    v128_t const a0  = wasm_v128_load(src + 0);
    v128_t const a1  = wasm_v128_load(src + 16);
    v128_t const a2  = wasm_v128_load(src + 32);
    v128_t const x16 = wasm_v128_or(
            wasm_v128_or(
                    wasm_i8x16_swizzle(a0, wasm_i8x16_const(0, 1, 6, 7, 12, 13, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)),
                    wasm_i8x16_swizzle(a1, wasm_i8x16_const(-1, -1, -1, -1, -1, -1, 2, 3, 8, 9, 14, 15, -1, -1, -1, -1))),
            wasm_i8x16_swizzle(a2, wasm_i8x16_const(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 10, 11)));
    v128_t const y16 = wasm_v128_or(
            wasm_v128_or(
                    wasm_i8x16_swizzle(a0, wasm_i8x16_const(2, 3, 8, 9, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)),
                    wasm_i8x16_swizzle(a1, wasm_i8x16_const(-1, -1, -1, -1, -1, -1, 4, 5, 10, 11, -1, -1, -1, -1, -1, -1))),
            wasm_i8x16_swizzle(a2, wasm_i8x16_const(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 6, 7, 12, 13)));
    v128_t const z16 = wasm_v128_or(
            wasm_v128_or(
                    wasm_i8x16_swizzle(a0, wasm_i8x16_const(4, 5, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)),
                    wasm_i8x16_swizzle(a1, wasm_i8x16_const(-1, -1, -1, -1, 0, 1, 6, 7, 12, 13, -1, -1, -1, -1, -1, -1))),
            wasm_i8x16_swizzle(a2, wasm_i8x16_const(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 2, 3, 8, 9, 14, 15)));
    *vx0 = wasm_u32x4_extend_low_u16x8(x16);  *vx1 = wasm_u32x4_extend_high_u16x8(x16);
    *vy0 = wasm_u32x4_extend_low_u16x8(y16);  *vy1 = wasm_u32x4_extend_high_u16x8(y16);
    *vz0 = wasm_u32x4_extend_low_u16x8(z16);  *vz1 = wasm_u32x4_extend_high_u16x8(z16);
}

ZL_FORCE_INLINE void hilbert3_load4_i32_xyz_flipped_wasm(
        const uint8_t* s, v128_t* ox, v128_t* oy, v128_t* oz)
{
    v128_t const b0   = wasm_v128_load(s + 0);
    v128_t const b1   = wasm_v128_load(s + 16);
    v128_t const b2   = wasm_v128_load(s + 32);
    v128_t const sign = wasm_i32x4_const_splat((int32_t)0x80000000u);
    *ox = wasm_v128_xor(sign, wasm_v128_or(
            wasm_i8x16_swizzle(b0, wasm_i8x16_const(0, 1, 2, 3, 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1)),
            wasm_v128_or(
                    wasm_i8x16_swizzle(b1, wasm_i8x16_const(-1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11, -1, -1, -1, -1)),
                    wasm_i8x16_swizzle(b2, wasm_i8x16_const(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 4, 5, 6, 7)))));
    *oy = wasm_v128_xor(sign, wasm_v128_or(
            wasm_i8x16_swizzle(b0, wasm_i8x16_const(4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)),
            wasm_v128_or(
                    wasm_i8x16_swizzle(b1, wasm_i8x16_const(-1, -1, -1, -1, 0, 1, 2, 3, 12, 13, 14, 15, -1, -1, -1, -1)),
                    wasm_i8x16_swizzle(b2, wasm_i8x16_const(-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 8, 9, 10, 11)))));
    *oz = wasm_v128_xor(sign, wasm_v128_or(
            wasm_i8x16_swizzle(b0, wasm_i8x16_const(8, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1)),
            wasm_v128_or(
                    wasm_i8x16_swizzle(b1, wasm_i8x16_const(-1, -1, -1, -1, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1)),
                    wasm_i8x16_swizzle(b2, wasm_i8x16_const(-1, -1, -1, -1, -1, -1, -1, -1, 0, 1, 2, 3, 12, 13, 14, 15)))));
}
#endif

static inline uint64_t hilbert3_encode_u16_one(
        uint32_t x,
        uint32_t y,
        uint32_t z)
{
    uint32_t const m = ((x >> 15) & 1u) | (((y >> 15) & 1u) << 1)
            | (((z >> 15) & 1u) << 2);
    uint32_t const d = ZL_hilbert3_topD[m];
    uint32_t s       = ZL_hilbert3_topS[d];
    uint64_t const A = hilbert3_packA15(x, y, z);
    uint64_t h       = (uint64_t)d << 45;
    for (int L = 4; L >= 0; L--) {
        uint32_t const e = ZL_hilbert3_E3[s * 512u + ((A >> (9 * L)) & 511u)];
        h |= (uint64_t)(e & 511u) << (9 * L);
        s = e >> 9;
    }
    return h;
}

ZL_Report
EI_hilbert3_u16(ZL_Encoder* eictx, const ZL_Input* ins[], size_t nbIns)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(eictx);
    ZL_ASSERT_EQ(nbIns, 1);
    const ZL_Input* const in = ins[0];
    ZL_ASSERT_EQ(ZL_Input_type(in), ZL_Type_struct);
    ZL_ERR_IF_NE(
            ZL_Input_eltWidth(in),
            6,
            node_invalid_input,
            "hilbert3_u16 requires 6-byte records");
    size_t const n         = ZL_Input_numElts(in);
    const uint8_t* const src = (const uint8_t*)ZL_Input_ptr(in);

    ZL_Output* const out = ZL_Encoder_createTypedStream(eictx, 0, n, 8);
    ZL_ERR_IF_NULL(out, allocation);
    uint64_t* const dst = (uint64_t*)ZL_Output_ptr(out);

    uint64_t prev = 0;
    size_t i      = 0;
#if ZL_HAS_AVX2
    uint64_t top2H = 0, top1H = 0;
    uint32_t top2Key = UINT32_MAX, top1Key = UINT32_MAX, top2S = 0, top1S = 0;
    for (; i + 8u <= n; i += 8u) {
        __m256i vx, vy, vz;
        hilbert3_load8_u16_xyz(src + 6 * i, &vx, &vy, &vz);
        uint32_t a27[8];
        _mm256_storeu_si256((__m256i_u*)(void*)a27, hilbert3_scatter9_xyz_avx2(vx, vy, vz));
        __m256i const vtop7 = _mm256_or_si256(
                _mm256_srli_epi32(vx, 9),
                _mm256_or_si256(
                        _mm256_slli_epi32(_mm256_srli_epi32(vy, 9), 8),
                        _mm256_slli_epi32(_mm256_srli_epi32(vz, 9), 16)));
        __m256i const diff7 = _mm256_xor_si256(vtop7, _mm256_set1_epi32((int)top2Key));
        uint64_t kh[8];
        uint32_t ss[8];
        if (_mm256_testz_si256(diff7, diff7)) {
            for (int j = 0; j < 8; j++) { kh[j] = top2H; ss[j] = top2S; }
        } else {
            __m256i const diff4 = _mm256_and_si256(
                    _mm256_xor_si256(vtop7, _mm256_set1_epi32((int)top1Key)),
                    _mm256_set1_epi32((int)0xF8F8F8u));
            uint32_t a18[8], top7a[8];
            _mm256_storeu_si256((__m256i_u*)(void*)a18, hilbert3_scatter6_xyz_avx2(_mm256_srli_epi32(vx, 9), _mm256_srli_epi32(vy, 9), _mm256_srli_epi32(vz, 9)));
            _mm256_storeu_si256((__m256i_u*)(void*)top7a, vtop7);
            if (ZL_LIKELY(_mm256_testz_si256(diff4, diff4))) {
                for (int j = 0; j < 8; j++) {
                    uint32_t const e3 = ZL_hilbert3_E3[top1S | (a18[j] & 511u)];
                    kh[j] = top1H | ((uint64_t)(e3 & 511u) << 27);
                    ss[j] = e3 & 0xFE00u;
                }
            } else {
                for (int j = 0; j < 8; j++) {
                    uint32_t const t  = top7a[j];
                    uint32_t const m  = ((t >> 6) & 1u) | (((t >> 14) & 1u) << 1) | (((t >> 22) & 1u) << 2);
                    uint32_t const d  = ZL_hilbert3_topD[m];
                    uint32_t const e4 = ZL_hilbert3_E3[((uint32_t)ZL_hilbert3_topS[d] << 9) | ((a18[j] >> 9) & 511u)];
                    uint32_t const e3 = ZL_hilbert3_E3[(e4 & 0xFE00u) | (a18[j] & 511u)];
                    uint64_t const h1 = ((uint64_t)d << 45) | ((uint64_t)(e4 & 511u) << 36);
                    kh[j] = h1 | ((uint64_t)(e3 & 511u) << 27);
                    ss[j] = e3 & 0xFE00u;
                    if (j == 7) { top1Key = t & 0xF8F8F8u; top1S = e4 & 0xFE00u; top1H = h1; }
                }
            }
            top2Key = top7a[7]; top2S = ss[7]; top2H = kh[7];
        }
        for (int j = 0; j < 8; j++) {
            uint32_t const l  = a27[j];
            uint32_t const e2 = ZL_hilbert3_E3[ss[j] | ((l >> 18) & 511u)];
            uint32_t const e1 = ZL_hilbert3_E3[(e2 & 0xFE00u) | ((l >> 9) & 511u)];
            uint32_t const e0 = ZL_hilbert3_E3[(e1 & 0xFE00u) | (l & 511u)];
            uint64_t const k  = kh[j] | ((uint64_t)(e2 & 511u) << 18) | ((uint64_t)(e1 & 511u) << 9) | (e0 & 511u);
            dst[i + (size_t)j] = k - prev;
            prev = k;
        }
    }
#elif ZL_HAS_WASM_SIMD128
    uint64_t top2H = 0, top1H = 0;
    uint32_t top2Key = UINT32_MAX, top1Key = UINT32_MAX, top2S = 0, top1S = 0;
    for (; i + 8u <= n; i += 8u) {
        v128_t vx0, vx1, vy0, vy1, vz0, vz1;
        hilbert3_load8_u16_xyz_wasm(src + 6 * i, &vx0, &vx1, &vy0, &vy1, &vz0, &vz1);
        uint32_t a27[8];
        wasm_v128_store(a27, hilbert3_scatter9_xyz_wasm(vx0, vy0, vz0));
        wasm_v128_store(a27 + 4, hilbert3_scatter9_xyz_wasm(vx1, vy1, vz1));
        v128_t const vt0 = wasm_v128_or(wasm_u32x4_shr(vx0, 9), wasm_v128_or(wasm_i32x4_shl(wasm_u32x4_shr(vy0, 9), 8), wasm_i32x4_shl(wasm_u32x4_shr(vz0, 9), 16)));
        v128_t const vt1 = wasm_v128_or(wasm_u32x4_shr(vx1, 9), wasm_v128_or(wasm_i32x4_shl(wasm_u32x4_shr(vy1, 9), 8), wasm_i32x4_shl(wasm_u32x4_shr(vz1, 9), 16)));
        v128_t const d7  = wasm_v128_or(wasm_v128_xor(vt0, wasm_i32x4_splat((int32_t)top2Key)), wasm_v128_xor(vt1, wasm_i32x4_splat((int32_t)top2Key)));
        uint64_t kh[8];
        uint32_t ss[8];
        if (!wasm_v128_any_true(d7)) {
            for (int j = 0; j < 8; j++) { kh[j] = top2H; ss[j] = top2S; }
        } else {
            uint32_t a18[8], top7a[8];
            wasm_v128_store(a18, hilbert3_scatter6_xyz_wasm(wasm_u32x4_shr(vx0, 9), wasm_u32x4_shr(vy0, 9), wasm_u32x4_shr(vz0, 9)));
            wasm_v128_store(a18 + 4, hilbert3_scatter6_xyz_wasm(wasm_u32x4_shr(vx1, 9), wasm_u32x4_shr(vy1, 9), wasm_u32x4_shr(vz1, 9)));
            wasm_v128_store(top7a, vt0);
            wasm_v128_store(top7a + 4, vt1);
            v128_t const d4 = wasm_v128_and(
                    wasm_v128_or(wasm_v128_xor(vt0, wasm_i32x4_splat((int32_t)top1Key)), wasm_v128_xor(vt1, wasm_i32x4_splat((int32_t)top1Key))),
                    wasm_i32x4_const_splat((int32_t)0xF8F8F8u));
            if (ZL_LIKELY(!wasm_v128_any_true(d4))) {
                for (int j = 0; j < 8; j++) {
                    uint32_t const e3 = ZL_hilbert3_E3[top1S | (a18[j] & 511u)];
                    kh[j] = top1H | ((uint64_t)(e3 & 511u) << 27);
                    ss[j] = e3 & 0xFE00u;
                }
            } else {
                for (int j = 0; j < 8; j++) {
                    uint32_t const t  = top7a[j];
                    uint32_t const m  = ((t >> 6) & 1u) | (((t >> 14) & 1u) << 1) | (((t >> 22) & 1u) << 2);
                    uint32_t const d  = ZL_hilbert3_topD[m];
                    uint32_t const e4 = ZL_hilbert3_E3[((uint32_t)ZL_hilbert3_topS[d] << 9) | ((a18[j] >> 9) & 511u)];
                    uint32_t const e3 = ZL_hilbert3_E3[(e4 & 0xFE00u) | (a18[j] & 511u)];
                    uint64_t const h1 = ((uint64_t)d << 45) | ((uint64_t)(e4 & 511u) << 36);
                    kh[j] = h1 | ((uint64_t)(e3 & 511u) << 27);
                    ss[j] = e3 & 0xFE00u;
                    if (j == 7) { top1Key = t & 0xF8F8F8u; top1S = e4 & 0xFE00u; top1H = h1; }
                }
            }
            top2Key = top7a[7]; top2S = ss[7]; top2H = kh[7];
        }
        for (int j = 0; j < 8; j++) {
            uint32_t const l  = a27[j];
            uint32_t const e2 = ZL_hilbert3_E3[ss[j] | ((l >> 18) & 511u)];
            uint32_t const e1 = ZL_hilbert3_E3[(e2 & 0xFE00u) | ((l >> 9) & 511u)];
            uint32_t const e0 = ZL_hilbert3_E3[(e1 & 0xFE00u) | (l & 511u)];
            uint64_t const k  = kh[j] | ((uint64_t)(e2 & 511u) << 18) | ((uint64_t)(e1 & 511u) << 9) | (e0 & 511u);
            dst[i + (size_t)j] = k - prev;
            prev = k;
        }
    }
#else
    for (; i + 4u <= n; i += 4u) {
        const uint8_t* const p0 = src + 6 * (i + 0u);
        const uint8_t* const p1 = src + 6 * (i + 1u);
        const uint8_t* const p2 = src + 6 * (i + 2u);
        const uint8_t* const p3 = src + 6 * (i + 3u);
        uint32_t const x0 = ZL_readLE16(p0), y0 = ZL_readLE16(p0 + 2),
                       z0 = ZL_readLE16(p0 + 4);
        uint32_t const x1 = ZL_readLE16(p1), y1 = ZL_readLE16(p1 + 2),
                       z1 = ZL_readLE16(p1 + 4);
        uint32_t const x2 = ZL_readLE16(p2), y2 = ZL_readLE16(p2 + 2),
                       z2 = ZL_readLE16(p2 + 4);
        uint32_t const x3 = ZL_readLE16(p3), y3 = ZL_readLE16(p3 + 2),
                       z3 = ZL_readLE16(p3 + 4);

        uint32_t const d0 = ZL_hilbert3_topD
                [((x0 >> 15) & 1u) | (((y0 >> 15) & 1u) << 1)
                 | (((z0 >> 15) & 1u) << 2)];
        uint32_t const d1 = ZL_hilbert3_topD
                [((x1 >> 15) & 1u) | (((y1 >> 15) & 1u) << 1)
                 | (((z1 >> 15) & 1u) << 2)];
        uint32_t const d2 = ZL_hilbert3_topD
                [((x2 >> 15) & 1u) | (((y2 >> 15) & 1u) << 1)
                 | (((z2 >> 15) & 1u) << 2)];
        uint32_t const d3 = ZL_hilbert3_topD
                [((x3 >> 15) & 1u) | (((y3 >> 15) & 1u) << 1)
                 | (((z3 >> 15) & 1u) << 2)];

        uint32_t s0 = ZL_hilbert3_topS[d0];
        uint32_t s1 = ZL_hilbert3_topS[d1];
        uint32_t s2 = ZL_hilbert3_topS[d2];
        uint32_t s3 = ZL_hilbert3_topS[d3];

        uint64_t const A0 = hilbert3_packA15(x0, y0, z0);
        uint64_t const A1 = hilbert3_packA15(x1, y1, z1);
        uint64_t const A2 = hilbert3_packA15(x2, y2, z2);
        uint64_t const A3 = hilbert3_packA15(x3, y3, z3);

        uint64_t k0 = (uint64_t)d0 << 45;
        uint64_t k1 = (uint64_t)d1 << 45;
        uint64_t k2 = (uint64_t)d2 << 45;
        uint64_t k3 = (uint64_t)d3 << 45;

        for (int L = 4; L >= 0; L--) {
            uint32_t const sh = 9u * (uint32_t)L;
            uint32_t const e0 =
                    ZL_hilbert3_E3[s0 * 512u + ((A0 >> sh) & 511u)];
            uint32_t const e1 =
                    ZL_hilbert3_E3[s1 * 512u + ((A1 >> sh) & 511u)];
            uint32_t const e2 =
                    ZL_hilbert3_E3[s2 * 512u + ((A2 >> sh) & 511u)];
            uint32_t const e3 =
                    ZL_hilbert3_E3[s3 * 512u + ((A3 >> sh) & 511u)];
            k0 |= (uint64_t)(e0 & 511u) << sh;
            k1 |= (uint64_t)(e1 & 511u) << sh;
            k2 |= (uint64_t)(e2 & 511u) << sh;
            k3 |= (uint64_t)(e3 & 511u) << sh;
            s0 = e0 >> 9;
            s1 = e1 >> 9;
            s2 = e2 >> 9;
            s3 = e3 >> 9;
        }
        dst[i + 0u] = k0 - prev;
        dst[i + 1u] = k1 - k0;
        dst[i + 2u] = k2 - k1;
        dst[i + 3u] = k3 - k2;
        prev        = k3;
    }
#endif
    for (; i < n; i++) {
        const uint8_t* const p = src + 6 * i;
        uint64_t const k       = hilbert3_encode_u16_one(
                ZL_readLE16(p), ZL_readLE16(p + 2), ZL_readLE16(p + 4));
        dst[i] = k - prev;
        prev   = k;
    }
    ZL_ERR_IF_ERR(ZL_Output_commit(out, n));
    return ZL_returnSuccess();
}

/* 32-level 3D Hilbert key via 3-level LUT with top-15-bit prefix cache. */
typedef struct {
    uint64_t prevTopKey;
    uint32_t topS;
    uint32_t topHi;
    uint64_t topLoHi;
} Hilbert3EncTop32;

ZL_FORCE_NOINLINE void hilbert3WalkEncTop32(
        Hilbert3EncTop32* t,
        uint32_t x,
        uint32_t y,
        uint32_t z)
{
    uint32_t const m =
            ((x >> 30) & 3u) | (((y >> 30) & 3u) << 2) | (((z >> 30) & 3u) << 4);
    uint32_t const d = ZL_hilbert3_top2D[m];
    uint32_t s       = ZL_hilbert3_top2S[d];
    uint32_t hi      = d << 26;
    uint64_t const A_hi =
            hilbert3_packA15(x >> 15, y >> 15, z >> 15);
    uint32_t const e9 = ZL_hilbert3_E3[s * 512u + ((A_hi >> 36) & 511u)];
    s                 = e9 >> 9;
    hi |= (e9 & 511u) << 17;
    uint32_t const e8 = ZL_hilbert3_E3[s * 512u + ((A_hi >> 27) & 511u)];
    s                 = e8 >> 9;
    hi |= (e8 & 511u) << 8;
    uint32_t const e7 = ZL_hilbert3_E3[s * 512u + ((A_hi >> 18) & 511u)];
    s                 = e7 >> 9;
    hi |= (e7 & 511u) >> 1;
    uint64_t lo       = (uint64_t)(e7 & 1u) << 63;
    uint32_t const e6 = ZL_hilbert3_E3[s * 512u + ((A_hi >> 9) & 511u)];
    s                 = e6 >> 9;
    lo |= (uint64_t)(e6 & 511u) << 54;
    uint32_t const e5 = ZL_hilbert3_E3[s * 512u + (A_hi & 511u)];
    s                 = e5 >> 9;
    lo |= (uint64_t)(e5 & 511u) << 45;

    t->topS    = s;
    t->topHi   = hi;
    t->topLoHi = lo;
}

/* Writes the gap stream for n records to dst (room for 3 * n words) and
 * returns the number of words written. */
static size_t hilbert3EncodeI32(uint64_t* dst, const uint8_t* src, size_t n)
{
    size_t o               = 0;
    size_t i               = 0;
    uint32_t phi           = 0;
    uint64_t plo           = 0;
    Hilbert3EncTop32 cache = { .prevTopKey = UINT64_MAX };
#if ZL_HAS_AVX2
    uint32_t cx12 = UINT32_MAX, cy12 = UINT32_MAX, cz12 = UINT32_MAX, cS4 = 0, cHi = 0;
    uint64_t cLo4 = 0;
    for (; i + 8u <= n; i += 8u) {
        __m256i vx, vy, vz;
        hilbert3_load8_i32_xyz_flipped(src + 12 * i, &vx, &vy, &vz);
        __m256i const dx   = _mm256_xor_si256(_mm256_srli_epi32(vx, 12), _mm256_set1_epi32((int)cx12));
        __m256i const dy   = _mm256_xor_si256(_mm256_srli_epi32(vy, 12), _mm256_set1_epi32((int)cy12));
        __m256i const dz   = _mm256_xor_si256(_mm256_srli_epi32(vz, 12), _mm256_set1_epi32((int)cz12));
        __m256i const dall = _mm256_or_si256(dx, _mm256_or_si256(dy, dz));
        if (ZL_LIKELY(i > 0 && cHi == phi && _mm256_testz_si256(dall, dall))) {
            uint32_t a27[8], d3a[8];
            _mm256_storeu_si256((__m256i_u*)(void*)a27, hilbert3_scatter9_xyz_avx2(vx, vy, vz));
            __m256i const vd3 = _mm256_or_si256(
                    _mm256_and_si256(_mm256_srli_epi32(vx, 9), _mm256_set1_epi32(7)),
                    _mm256_or_si256(
                            _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(vy, 9), _mm256_set1_epi32(7)), 3),
                            _mm256_slli_epi32(_mm256_and_si256(_mm256_srli_epi32(vz, 9), _mm256_set1_epi32(7)), 6)));
            _mm256_storeu_si256((__m256i_u*)(void*)d3a, vd3);
            for (int j = 0; j < 8; j++) {
                uint32_t const l  = a27[j];
                uint32_t const e3 = ZL_hilbert3_E3[cS4 | d3a[j]];
                uint32_t const e2 = ZL_hilbert3_E3[(e3 & 0xFE00u) | ((l >> 18) & 511u)];
                uint32_t const e1 = ZL_hilbert3_E3[(e2 & 0xFE00u) | ((l >> 9) & 511u)];
                uint32_t const e0 = ZL_hilbert3_E3[(e1 & 0xFE00u) | (l & 511u)];
                uint64_t const lo = cLo4 | ((uint64_t)(e3 & 511u) << 27) | ((uint64_t)(e2 & 511u) << 18) | ((uint64_t)(e1 & 511u) << 9) | (e0 & 511u);
                if (ZL_UNLIKELY(lo < plo || lo - plo == UINT64_MAX)) {
                    dst[o++] = UINT64_MAX; dst[o++] = cHi; dst[o++] = lo;
                } else {
                    dst[o++] = lo - plo;
                }
                plo = lo;
            }
            continue;
        }
        uint32_t xa[8], ya[8], za[8], a27[8], a18[8];
        _mm256_storeu_si256((__m256i_u*)(void*)xa, vx);
        _mm256_storeu_si256((__m256i_u*)(void*)ya, vy);
        _mm256_storeu_si256((__m256i_u*)(void*)za, vz);
        _mm256_storeu_si256((__m256i_u*)(void*)a27, hilbert3_scatter9_xyz_avx2(vx, vy, vz));
        _mm256_storeu_si256((__m256i_u*)(void*)a18, hilbert3_scatter6_xyz_avx2(_mm256_srli_epi32(vx, 9), _mm256_srli_epi32(vy, 9), _mm256_srli_epi32(vz, 9)));
        for (int j = 0; j < 8; j++) {
            uint32_t const x      = xa[j], y = ya[j], z = za[j];
            uint64_t const topKey = ((uint64_t)(x >> 15)) | ((uint64_t)(y >> 15) << 17) | ((uint64_t)(z >> 15) << 34);
            if (ZL_UNLIKELY(topKey != cache.prevTopKey)) {
                cache.prevTopKey = topKey;
                hilbert3WalkEncTop32(&cache, x, y, z);
            }
            uint32_t const e4  = ZL_hilbert3_E3[(cache.topS << 9) | ((a18[j] >> 9) & 511u)];
            uint32_t const e3  = ZL_hilbert3_E3[(e4 & 0xFE00u) | (a18[j] & 511u)];
            uint32_t const l   = a27[j];
            uint32_t const e2  = ZL_hilbert3_E3[(e3 & 0xFE00u) | ((l >> 18) & 511u)];
            uint32_t const e1  = ZL_hilbert3_E3[(e2 & 0xFE00u) | ((l >> 9) & 511u)];
            uint32_t const e0  = ZL_hilbert3_E3[(e1 & 0xFE00u) | (l & 511u)];
            uint32_t const hi  = cache.topHi;
            uint64_t const lo4 = cache.topLoHi | ((uint64_t)(e4 & 511u) << 36);
            uint64_t const lo  = lo4 | ((uint64_t)(e3 & 511u) << 27) | ((uint64_t)(e2 & 511u) << 18) | ((uint64_t)(e1 & 511u) << 9) | (e0 & 511u);
            uint64_t const dlo = lo - plo;
            uint32_t const dhi = hi - phi - (lo < plo);
            if ((i == 0 && j == 0) || hi < phi || (hi == phi && lo < plo) || dhi != 0 || dlo == UINT64_MAX) {
                dst[o++] = UINT64_MAX; dst[o++] = hi; dst[o++] = lo;
            } else {
                dst[o++] = dlo;
            }
            phi = hi; plo = lo;
            if (j == 7) { cx12 = x >> 12; cy12 = y >> 12; cz12 = z >> 12; cS4 = e4 & 0xFE00u; cHi = hi; cLo4 = lo4; }
        }
    }
#elif ZL_HAS_WASM_SIMD128
    uint32_t cx12 = UINT32_MAX, cy12 = UINT32_MAX, cz12 = UINT32_MAX, cS4 = 0, cHi = 0;
    uint64_t cLo4 = 0;
    for (; i + 4u <= n; i += 4u) {
        v128_t vx, vy, vz;
        hilbert3_load4_i32_xyz_flipped_wasm(src + 12 * i, &vx, &vy, &vz);
        v128_t const dx   = wasm_v128_xor(wasm_u32x4_shr(vx, 12), wasm_i32x4_splat((int32_t)cx12));
        v128_t const dy   = wasm_v128_xor(wasm_u32x4_shr(vy, 12), wasm_i32x4_splat((int32_t)cy12));
        v128_t const dz   = wasm_v128_xor(wasm_u32x4_shr(vz, 12), wasm_i32x4_splat((int32_t)cz12));
        v128_t const dall = wasm_v128_or(dx, wasm_v128_or(dy, dz));
        if (ZL_LIKELY(i > 0 && cHi == phi && !wasm_v128_any_true(dall))) {
            uint32_t a27[4], d3a[4];
            wasm_v128_store(a27, hilbert3_scatter9_xyz_wasm(vx, vy, vz));
            v128_t const vd3 = wasm_v128_or(
                    wasm_v128_and(wasm_u32x4_shr(vx, 9), wasm_i32x4_const_splat(7)),
                    wasm_v128_or(
                            wasm_i32x4_shl(wasm_v128_and(wasm_u32x4_shr(vy, 9), wasm_i32x4_const_splat(7)), 3),
                            wasm_i32x4_shl(wasm_v128_and(wasm_u32x4_shr(vz, 9), wasm_i32x4_const_splat(7)), 6)));
            wasm_v128_store(d3a, vd3);
            for (int j = 0; j < 4; j++) {
                uint32_t const l  = a27[j];
                uint32_t const e3 = ZL_hilbert3_E3[cS4 | d3a[j]];
                uint32_t const e2 = ZL_hilbert3_E3[(e3 & 0xFE00u) | ((l >> 18) & 511u)];
                uint32_t const e1 = ZL_hilbert3_E3[(e2 & 0xFE00u) | ((l >> 9) & 511u)];
                uint32_t const e0 = ZL_hilbert3_E3[(e1 & 0xFE00u) | (l & 511u)];
                uint64_t const lo = cLo4 | ((uint64_t)(e3 & 511u) << 27) | ((uint64_t)(e2 & 511u) << 18) | ((uint64_t)(e1 & 511u) << 9) | (e0 & 511u);
                if (ZL_UNLIKELY(lo < plo || lo - plo == UINT64_MAX)) {
                    dst[o++] = UINT64_MAX; dst[o++] = cHi; dst[o++] = lo;
                } else {
                    dst[o++] = lo - plo;
                }
                plo = lo;
            }
            continue;
        }
        uint32_t xa[4], ya[4], za[4], a27[4], a18[4];
        wasm_v128_store(xa, vx); wasm_v128_store(ya, vy); wasm_v128_store(za, vz);
        wasm_v128_store(a27, hilbert3_scatter9_xyz_wasm(vx, vy, vz));
        wasm_v128_store(a18, hilbert3_scatter6_xyz_wasm(wasm_u32x4_shr(vx, 9), wasm_u32x4_shr(vy, 9), wasm_u32x4_shr(vz, 9)));
        for (int j = 0; j < 4; j++) {
            uint32_t const x      = xa[j], y = ya[j], z = za[j];
            uint64_t const topKey = ((uint64_t)(x >> 15)) | ((uint64_t)(y >> 15) << 17) | ((uint64_t)(z >> 15) << 34);
            if (ZL_UNLIKELY(topKey != cache.prevTopKey)) {
                cache.prevTopKey = topKey;
                hilbert3WalkEncTop32(&cache, x, y, z);
            }
            uint32_t const e4  = ZL_hilbert3_E3[(cache.topS << 9) | ((a18[j] >> 9) & 511u)];
            uint32_t const e3  = ZL_hilbert3_E3[(e4 & 0xFE00u) | (a18[j] & 511u)];
            uint32_t const l   = a27[j];
            uint32_t const e2  = ZL_hilbert3_E3[(e3 & 0xFE00u) | ((l >> 18) & 511u)];
            uint32_t const e1  = ZL_hilbert3_E3[(e2 & 0xFE00u) | ((l >> 9) & 511u)];
            uint32_t const e0  = ZL_hilbert3_E3[(e1 & 0xFE00u) | (l & 511u)];
            uint32_t const hi  = cache.topHi;
            uint64_t const lo4 = cache.topLoHi | ((uint64_t)(e4 & 511u) << 36);
            uint64_t const lo  = lo4 | ((uint64_t)(e3 & 511u) << 27) | ((uint64_t)(e2 & 511u) << 18) | ((uint64_t)(e1 & 511u) << 9) | (e0 & 511u);
            uint64_t const dlo = lo - plo;
            uint32_t const dhi = hi - phi - (lo < plo);
            if ((i == 0 && j == 0) || hi < phi || (hi == phi && lo < plo) || dhi != 0 || dlo == UINT64_MAX) {
                dst[o++] = UINT64_MAX; dst[o++] = hi; dst[o++] = lo;
            } else {
                dst[o++] = dlo;
            }
            phi = hi; plo = lo;
            if (j == 3) { cx12 = x >> 12; cy12 = y >> 12; cz12 = z >> 12; cS4 = e4 & 0xFE00u; cHi = hi; cLo4 = lo4; }
        }
    }
#endif
    for (; i < n; i++) {
        const uint8_t* const p = src + 12 * i;
        uint32_t const x       = ZL_readLE32(p) ^ 0x80000000u;
        uint32_t const y       = ZL_readLE32(p + 4) ^ 0x80000000u;
        uint32_t const z       = ZL_readLE32(p + 8) ^ 0x80000000u;
        uint64_t const topKey  = ((uint64_t)(x >> 15))
                | ((uint64_t)(y >> 15) << 17) | ((uint64_t)(z >> 15) << 34);
        if (ZL_UNLIKELY(topKey != cache.prevTopKey)) {
            cache.prevTopKey = topKey;
            hilbert3WalkEncTop32(&cache, x, y, z);
        }
        uint32_t s          = cache.topS;
        uint32_t const hi   = cache.topHi;
        uint64_t lo         = cache.topLoHi;
        uint64_t const A_lo = hilbert3_packA15(x, y, z);
        for (int L = 4; L >= 0; L--) {
            uint32_t const e =
                    ZL_hilbert3_E3[s * 512u + ((A_lo >> (9 * L)) & 511u)];
            lo |= (uint64_t)(e & 511u) << (9 * L);
            s = e >> 9;
        }
        uint64_t const dlo = lo - plo;
        uint32_t const dhi = hi - phi - (lo < plo);
        int const down     = hi < phi || (hi == phi && lo < plo);
        if (i == 0 || down || dhi != 0 || dlo == UINT64_MAX) {
            dst[o++] = UINT64_MAX;
            dst[o++] = hi;
            dst[o++] = lo;
        } else {
            dst[o++] = dlo;
        }
        phi = hi;
        plo = lo;
    }
    return o;
}

ZL_Report
EI_hilbert3_i32(ZL_Encoder* eictx, const ZL_Input* ins[], size_t nbIns)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(eictx);
    ZL_ASSERT_EQ(nbIns, 1);
    const ZL_Input* const in = ins[0];
    ZL_ASSERT_EQ(ZL_Input_type(in), ZL_Type_struct);
    ZL_ERR_IF_NE(
            ZL_Input_eltWidth(in),
            12,
            node_invalid_input,
            "hilbert3_i32 requires 12-byte records");
    size_t const n = ZL_Input_numElts(in);

    ZL_Output* const out = ZL_Encoder_createTypedStream(eictx, 0, 3 * n, 8);
    ZL_ERR_IF_NULL(out, allocation);
    size_t const nbWords = hilbert3EncodeI32(
            (uint64_t*)ZL_Output_ptr(out), (const uint8_t*)ZL_Input_ptr(in), n);
    ZL_ERR_IF_ERR(ZL_Output_commit(out, nbWords));
    return ZL_returnSuccess();
}
