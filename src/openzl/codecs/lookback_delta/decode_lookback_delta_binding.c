// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/lookback_delta/decode_lookback_delta_binding.h"

#include <stdint.h>
#include <string.h>

#include "openzl/codecs/delta/decode_delta_kernel.h"
#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
#include "openzl/shared/portability.h"

enum {
    LBD_MODE_RAW           = 0,
    LBD_MODE_CONSECUTIVE   = 1,
    LBD_MODE_LOOKBACK      = 2,
    LBD_MODE_CONSECUTIVE_2 = 3,
    LBD_MODE_CONSECUTIVE_3 = 4,
};

#if ZL_HAS_AVX2 || ZL_HAS_WASM_SIMD128
static inline int lbd_validate_lb(const uint16_t* lb, size_t nLb)
{
    size_t const checkN = nLb < 65535u ? nLb : 65535u;
    size_t k            = 0;
#if ZL_HAS_AVX2
    __m256i idx        = _mm256_setr_epi16(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m256i const step = _mm256_set1_epi16(16);
    for (; k + 16u <= checkN; k += 16u) {
        __m256i const v    = _mm256_loadu_si256((__m256i_u const*)(lb + k));
        __m256i const diff = _mm256_subs_epu16(v, idx);
        if (ZL_UNLIKELY(!_mm256_testz_si256(diff, diff))) {
            return 0;
        }
        idx = _mm256_add_epi16(idx, step);
    }
#elif ZL_HAS_WASM_SIMD128
    v128_t idx        = wasm_u16x8_const(0, 1, 2, 3, 4, 5, 6, 7);
    v128_t const step = wasm_u16x8_splat(8u);
    for (; k + 8u <= checkN; k += 8u) {
        v128_t const v   = wasm_v128_load(lb + k);
        v128_t const bad = wasm_u16x8_gt(v, idx);
        if (ZL_UNLIKELY(wasm_v128_any_true(bad))) {
            return 0;
        }
        idx = wasm_i16x8_add(idx, step);
    }
#endif
    for (; k < checkN; k++) {
        if (ZL_UNLIKELY((size_t)lb[k] > k)) {
            return 0;
        }
    }
    return 1;
}
#endif

#define LBD_DEC_LB_STEP4(T, dst, res, lb, i)                  \
    do {                                                      \
        T const r0      = (res)[(i) + 0u];                    \
        T const r1      = (res)[(i) + 1u];                    \
        T const r2      = (res)[(i) + 2u];                    \
        T const r3      = (res)[(i) + 3u];                    \
        size_t const l0 = (lb)[(i) - 1u];                     \
        size_t const l1 = (lb)[(i) + 0u];                     \
        size_t const l2 = (lb)[(i) + 1u];                     \
        size_t const l3 = (lb)[(i) + 2u];                     \
        (dst)[(i) + 0u] = (T)((dst)[(i) - 1u - l0] + r0);     \
        (dst)[(i) + 1u] = (T)((dst)[(i) + 0u - l1] + r1);     \
        (dst)[(i) + 2u] = (T)((dst)[(i) + 1u - l2] + r2);     \
        (dst)[(i) + 3u] = (T)((dst)[(i) + 2u - l3] + r3);     \
    } while (0)

#define LBD_DEC_LB_SCALAR_TAIL(T, dst, res, lb, i, n)                         \
    do {                                                                      \
        if ((i) < (n)) {                                                      \
            while ((n) - (i) >= 4u) {                                         \
                LBD_DEC_LB_STEP4(T, dst, res, lb, i);                         \
                (i) += 4u;                                                    \
            }                                                                 \
            size_t const lbd_rem = (n) - (i);                                 \
            for (size_t lbd_r = 0; lbd_r < lbd_rem; lbd_r++, (i)++) {         \
                (dst)[(i)] = (T)((dst)[(i) - (size_t)(lb)[(i) - 1u] - 1u] + (res)[(i)]); \
            }                                                                 \
        }                                                                     \
    } while (0)

#if ZL_HAS_AVX2

static inline __m256i lbd_psum8_avx2(__m256i v)
{
    v = _mm256_add_epi8(v, _mm256_slli_si256(v, 1));
    v = _mm256_add_epi8(v, _mm256_slli_si256(v, 2));
    v = _mm256_add_epi8(v, _mm256_slli_si256(v, 4));
    v = _mm256_add_epi8(v, _mm256_slli_si256(v, 8));
    __m256i const cross = _mm256_set_m128i(
            _mm_set1_epi8(15), _mm_set1_epi8((char)0x80));
    return _mm256_add_epi8(
            v, _mm256_shuffle_epi8(_mm256_permute4x64_epi64(v, 0x55), cross));
}

static inline __m256i lbd_bcast_last8_avx2(__m256i v)
{
    return _mm256_shuffle_epi8(
            _mm256_permute4x64_epi64(v, 0xFF), _mm256_set1_epi8(15));
}

static inline __m256i lbd_psum16_avx2(__m256i v)
{
    v = _mm256_add_epi16(v, _mm256_slli_si256(v, 2));
    v = _mm256_add_epi16(v, _mm256_slli_si256(v, 4));
    v = _mm256_add_epi16(v, _mm256_slli_si256(v, 8));
    __m256i const cross = _mm256_set_m128i(
            _mm_set1_epi16(0x0F0E), _mm_set1_epi8((char)0x80));
    return _mm256_add_epi16(
            v, _mm256_shuffle_epi8(_mm256_permute4x64_epi64(v, 0x55), cross));
}

static inline __m256i lbd_bcast_last16_avx2(__m256i v)
{
    return _mm256_shuffle_epi8(
            _mm256_permute4x64_epi64(v, 0xFF), _mm256_set1_epi16(0x0F0E));
}

static inline __m256i lbd_psum32_avx2(__m256i v)
{
    v = _mm256_add_epi32(v, _mm256_slli_si256(v, 4));
    v = _mm256_add_epi32(v, _mm256_slli_si256(v, 8));
    __m256i const shuf = _mm256_shuffle_epi32(v, 0xFF);
    return _mm256_add_epi32(v, _mm256_permute2x128_si256(shuf, shuf, 0x08));
}

static inline __m256i lbd_bcast_last32_avx2(__m256i v)
{
    return _mm256_permutevar8x32_epi32(v, _mm256_set1_epi32(7));
}

static inline __m256i lbd_psum64_avx2(__m256i v)
{
    v = _mm256_add_epi64(v, _mm256_slli_si256(v, 8));
    __m256i const shuf = _mm256_unpackhi_epi64(v, v);
    return _mm256_add_epi64(v, _mm256_permute2x128_si256(shuf, shuf, 0x08));
}

/* Resolves arbitrary causal lookbacks inside a 16-lane u8 or 8-lane u16 block
 * via log2(lanes) rounds of pshufb pointer-jumping. Lanes whose ancestor lies
 * in `prev` receive `prev[q + 16]` up front with bit 7 set in `shuf` so pshufb
 * zeroes them in subsequent intra-block rounds; intra-block lanes preserve
 * bit 7 via `shuf = pshufb(shuf, shuf) | (shuf & 0x80)` once resolved. */
static inline __m128i lbd_dag16_u8_avx2(
        uint8_t* __restrict dst,
        const uint8_t* __restrict res,
        size_t i,
        __m256i vlb16,
        __m128i prev)
{
    __m256i const k_minus_1 = _mm256_setr_epi16(
            -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);
    __m128i const c16   = _mm_set1_epi8(16);
    __m128i const msb   = _mm_set1_epi8((char)0x80);
    __m256i const q16   = _mm256_sub_epi16(k_minus_1, vlb16);
    __m128i const intra = _mm_packs_epi16(
            _mm256_castsi256_si128(q16),
            _mm256_extracti128_si256(q16, 1));
    __m128i const ext_shuf = _mm_or_si128(
            _mm_add_epi8(intra, c16), _mm_andnot_si128(intra, msb));
    __m128i const vres = _mm_loadu_si128((__m128i_u const*)(res + i));
    __m128i acc  = _mm_add_epi8(vres, _mm_shuffle_epi8(prev, ext_shuf));
    __m128i shuf = intra;

    acc  = _mm_add_epi8(acc, _mm_shuffle_epi8(acc, shuf));
    shuf = _mm_or_si128(_mm_shuffle_epi8(shuf, shuf), _mm_and_si128(shuf, msb));
    acc  = _mm_add_epi8(acc, _mm_shuffle_epi8(acc, shuf));
    shuf = _mm_or_si128(_mm_shuffle_epi8(shuf, shuf), _mm_and_si128(shuf, msb));
    acc  = _mm_add_epi8(acc, _mm_shuffle_epi8(acc, shuf));
    shuf = _mm_or_si128(_mm_shuffle_epi8(shuf, shuf), _mm_and_si128(shuf, msb));
    acc  = _mm_add_epi8(acc, _mm_shuffle_epi8(acc, shuf));

    _mm_storeu_si128((__m128i_u*)(dst + i), acc);
    return acc;
}

static inline __m128i lbd_dag8_u16_avx2(
        uint16_t* __restrict dst,
        const uint16_t* __restrict res,
        size_t i,
        __m128i vlb8,
        __m128i prev)
{
    __m128i const k_minus_1 = _mm_setr_epi16(-1, 0, 1, 2, 3, 4, 5, 6);
    __m128i const mask_lo   = _mm_set1_epi16(0x00FF);
    __m128i const add_hi    = _mm_set1_epi16(0x0100);
    __m128i const c16       = _mm_set1_epi8(16);
    __m128i const msb       = _mm_set1_epi8((char)0x80);

    __m128i const q  = _mm_sub_epi16(k_minus_1, vlb8);
    __m128i const b2 = _mm_and_si128(_mm_slli_epi16(q, 1), mask_lo);
    __m128i const intra = _mm_add_epi8(
            _mm_or_si128(b2, _mm_slli_epi16(b2, 8)), add_hi);
    __m128i const ext_shuf = _mm_or_si128(
            _mm_add_epi8(intra, c16), _mm_andnot_si128(intra, msb));

    __m128i const vres = _mm_loadu_si128((__m128i_u const*)(res + i));
    __m128i acc  = _mm_add_epi16(vres, _mm_shuffle_epi8(prev, ext_shuf));
    __m128i shuf = intra;

    acc  = _mm_add_epi16(acc, _mm_shuffle_epi8(acc, shuf));
    shuf = _mm_or_si128(_mm_shuffle_epi8(shuf, shuf), _mm_and_si128(shuf, msb));
    acc  = _mm_add_epi16(acc, _mm_shuffle_epi8(acc, shuf));
    shuf = _mm_or_si128(_mm_shuffle_epi8(shuf, shuf), _mm_and_si128(shuf, msb));
    acc  = _mm_add_epi16(acc, _mm_shuffle_epi8(acc, shuf));

    _mm_storeu_si128((__m128i_u*)(dst + i), acc);
    return acc;
}

static void lbd_dec_lb_1_avx2(
        uint8_t* __restrict dst,
        const uint8_t* __restrict res,
        const uint16_t* __restrict lb,
        size_t n)
{
    dst[0]      = res[0];
    size_t i    = 1;
    size_t warm = n < 17u ? n : 17u;
    for (; i < warm; i++) {
        dst[i] = (uint8_t)(dst[i - (size_t)lb[i - 1u] - 1u] + res[i]);
    }
    if (i >= n) {
        return;
    }

    __m256i const k_plus_15 = _mm256_setr_epi16(
            15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30);
    __m128i prev    = _mm_loadu_si128((__m128i_u const*)(dst + i - 16u));
    int prev_valid  = 1;

    while (i + 32u <= n) {
        __m256i const vlb0 = _mm256_loadu_si256((__m256i_u const*)(lb + i - 1u));
        uint16_t const lb0 = lb[i - 1u];
        __m256i const vc   = _mm256_set1_epi16((int16_t)lb0);
        __m256i const d0   = _mm256_xor_si256(vlb0, vc);
        if (ZL_UNLIKELY(_mm256_testz_si256(d0, d0))) {
            __m256i const vlb1 = _mm256_loadu_si256((__m256i_u const*)(lb + i + 15u));
            __m256i const d1   = _mm256_xor_si256(vlb1, vc);
            if (_mm256_testz_si256(d1, d1)) {
                size_t const dist = (size_t)lb0 + 1u;
                if (lb0 == 0u) {
                    __m256i carry = _mm256_set1_epi8((char)dst[i - 1u]);
                    for (;;) {
                        __m256i const vr = _mm256_loadu_si256((__m256i_u const*)(res + i));
                        __m256i const ps = lbd_psum8_avx2(vr);
                        __m256i const vv = _mm256_add_epi8(carry, ps);
                        _mm256_storeu_si256((__m256i_u*)(dst + i), vv);
                        prev  = _mm256_extracti128_si256(vv, 1);
                        carry = _mm256_add_epi8(carry, lbd_bcast_last8_avx2(ps));
                        i += 32u;
                        if (i + 32u > n) {
                            break;
                        }
                        __m256i const a = _mm256_loadu_si256((__m256i_u const*)(lb + i - 1u));
                        __m256i const b = _mm256_loadu_si256((__m256i_u const*)(lb + i + 15u));
                        __m256i const u = _mm256_or_si256(a, b);
                        if (!_mm256_testz_si256(u, u)) {
                            break;
                        }
                    }
                    prev_valid = 1;
                    continue;
                } else if (dist >= 32u) {
                    for (;;) {
                        __m256i const vr = _mm256_loadu_si256((__m256i_u const*)(res + i));
                        __m256i const vs = _mm256_loadu_si256((__m256i_u const*)(dst + i - dist));
                        __m256i const vv = _mm256_add_epi8(vr, vs);
                        _mm256_storeu_si256((__m256i_u*)(dst + i), vv);
                        prev = _mm256_extracti128_si256(vv, 1);
                        i += 32u;
                        if (i + 32u > n) {
                            break;
                        }
                        __m256i const a = _mm256_xor_si256(
                                _mm256_loadu_si256((__m256i_u const*)(lb + i - 1u)), vc);
                        __m256i const b = _mm256_xor_si256(
                                _mm256_loadu_si256((__m256i_u const*)(lb + i + 15u)), vc);
                        __m256i const u = _mm256_or_si256(a, b);
                        if (!_mm256_testz_si256(u, u)) {
                            break;
                        }
                    }
                    prev_valid = 1;
                    continue;
                } else if (dist >= 16u) {
                    __m128i const vr0 = _mm_loadu_si128((__m128i_u const*)(res + i));
                    __m128i const vs0 = _mm_loadu_si128((__m128i_u const*)(dst + i - dist));
                    _mm_storeu_si128((__m128i_u*)(dst + i), _mm_add_epi8(vr0, vs0));
                    __m128i const vr1 = _mm_loadu_si128((__m128i_u const*)(res + i + 16u));
                    __m128i const vs1 = _mm_loadu_si128((__m128i_u const*)(dst + i + 16u - dist));
                    prev = _mm_add_epi8(vr1, vs1);
                    _mm_storeu_si128((__m128i_u*)(dst + i + 16u), prev);
                    prev_valid = 1;
                    i += 32u;
                    continue;
                }
            }
        }
        __m256i const over = _mm256_subs_epu16(vlb0, k_plus_15);
        if (ZL_LIKELY(_mm256_testz_si256(over, over))) {
            if (ZL_UNLIKELY(!prev_valid)) {
                prev       = _mm_loadu_si128((__m128i_u const*)(dst + i - 16u));
                prev_valid = 1;
            }
            prev = lbd_dag16_u8_avx2(dst, res, i, vlb0, prev);
            i += 16u;
            continue;
        }
        /* Long non-uniform lookbacks: stay in 4x-unrolled scalar without
         * touching XMM loads over partial byte stores in the store buffer. */
        prev_valid = 0;
        for (;;) {
            LBD_DEC_LB_STEP4(uint8_t, dst, res, lb, i);
            LBD_DEC_LB_STEP4(uint8_t, dst, res, lb, i + 4u);
            LBD_DEC_LB_STEP4(uint8_t, dst, res, lb, i + 8u);
            LBD_DEC_LB_STEP4(uint8_t, dst, res, lb, i + 12u);
            i += 16u;
            if (i + 32u > n) {
                break;
            }
            __m256i const nxt_lb = _mm256_loadu_si256((__m256i_u const*)(lb + i - 1u));
            __m256i const nxt_ov = _mm256_subs_epu16(nxt_lb, k_plus_15);
            if (_mm256_testz_si256(nxt_ov, nxt_ov)) {
                break;
            }
            __m256i const nxt_df = _mm256_xor_si256(nxt_lb, _mm256_set1_epi16((int16_t)lb[i - 1u]));
            if (_mm256_testz_si256(nxt_df, nxt_df)) {
                break;
            }
        }
    }
    if (i + 16u <= n) {
        __m256i const vlb0 = _mm256_loadu_si256((__m256i_u const*)(lb + i - 1u));
        __m256i const over = _mm256_subs_epu16(vlb0, k_plus_15);
        if (_mm256_testz_si256(over, over)) {
            if (!prev_valid) {
                prev = _mm_loadu_si128((__m128i_u const*)(dst + i - 16u));
            }
            (void)lbd_dag16_u8_avx2(dst, res, i, vlb0, prev);
            i += 16u;
        }
    }
    LBD_DEC_LB_SCALAR_TAIL(uint8_t, dst, res, lb, i, n);
}

static void lbd_dec_lb_2_avx2(
        uint16_t* __restrict dst,
        const uint16_t* __restrict res,
        const uint16_t* __restrict lb,
        size_t n)
{
    dst[0]      = res[0];
    size_t i    = 1;
    size_t warm = n < 9u ? n : 9u;
    for (; i < warm; i++) {
        dst[i] = (uint16_t)(dst[i - (size_t)lb[i - 1u] - 1u] + res[i]);
    }
    if (i >= n) {
        return;
    }

    __m128i const k_plus_7 = _mm_setr_epi16(7, 8, 9, 10, 11, 12, 13, 14);
    __m128i prev           = _mm_loadu_si128((__m128i_u const*)(dst + i - 8u));
    int prev_valid         = 1;

    while (i + 16u <= n) {
        __m256i const vlb16 = _mm256_loadu_si256((__m256i_u const*)(lb + i - 1u));
        uint16_t const lb0  = lb[i - 1u];
        __m256i const vc    = _mm256_set1_epi16((int16_t)lb0);
        __m256i const diff  = _mm256_xor_si256(vlb16, vc);
        if (ZL_UNLIKELY(_mm256_testz_si256(diff, diff))) {
            size_t const dist = (size_t)lb0 + 1u;
            if (lb0 == 0u) {
                __m256i carry = _mm256_set1_epi16((int16_t)dst[i - 1u]);
                for (;;) {
                    __m256i const vr = _mm256_loadu_si256((__m256i_u const*)(res + i));
                    __m256i const ps = lbd_psum16_avx2(vr);
                    __m256i const vv = _mm256_add_epi16(carry, ps);
                    _mm256_storeu_si256((__m256i_u*)(dst + i), vv);
                    prev  = _mm256_extracti128_si256(vv, 1);
                    carry = _mm256_add_epi16(carry, lbd_bcast_last16_avx2(ps));
                    i += 16u;
                    if (i + 16u > n) {
                        break;
                    }
                    __m256i const nxt = _mm256_loadu_si256((__m256i_u const*)(lb + i - 1u));
                    if (!_mm256_testz_si256(nxt, nxt)) {
                        break;
                    }
                }
                prev_valid = 1;
                continue;
            } else if (dist >= 16u) {
                for (;;) {
                    __m256i const vr = _mm256_loadu_si256((__m256i_u const*)(res + i));
                    __m256i const vs = _mm256_loadu_si256((__m256i_u const*)(dst + i - dist));
                    __m256i const vv = _mm256_add_epi16(vr, vs);
                    _mm256_storeu_si256((__m256i_u*)(dst + i), vv);
                    prev = _mm256_extracti128_si256(vv, 1);
                    i += 16u;
                    if (i + 16u > n) {
                        break;
                    }
                    __m256i const nxt  = _mm256_loadu_si256((__m256i_u const*)(lb + i - 1u));
                    __m256i const ndif = _mm256_xor_si256(nxt, vc);
                    if (!_mm256_testz_si256(ndif, ndif)) {
                        break;
                    }
                }
                prev_valid = 1;
                continue;
            } else if (dist >= 8u) {
                __m128i const vr0 = _mm_loadu_si128((__m128i_u const*)(res + i));
                __m128i const vs0 = _mm_loadu_si128((__m128i_u const*)(dst + i - dist));
                _mm_storeu_si128((__m128i_u*)(dst + i), _mm_add_epi16(vr0, vs0));
                __m128i const vr1 = _mm_loadu_si128((__m128i_u const*)(res + i + 8u));
                __m128i const vs1 = _mm_loadu_si128((__m128i_u const*)(dst + i + 8u - dist));
                prev = _mm_add_epi16(vr1, vs1);
                _mm_storeu_si128((__m128i_u*)(dst + i + 8u), prev);
                prev_valid = 1;
                i += 16u;
                continue;
            }
        }
        __m128i const lo8 = _mm256_castsi256_si128(vlb16);
        __m128i const hi8 = _mm256_extracti128_si256(vlb16, 1);
        __m128i const ov0 = _mm_subs_epu16(lo8, k_plus_7);
        if (ZL_LIKELY(_mm_testz_si128(ov0, ov0))) {
            if (ZL_UNLIKELY(!prev_valid)) {
                prev       = _mm_loadu_si128((__m128i_u const*)(dst + i - 8u));
                prev_valid = 1;
            }
            prev = lbd_dag8_u16_avx2(dst, res, i, lo8, prev);
            __m128i const ov1 = _mm_subs_epu16(hi8, k_plus_7);
            if (ZL_LIKELY(_mm_testz_si128(ov1, ov1))) {
                prev = lbd_dag8_u16_avx2(dst, res, i + 8u, hi8, prev);
            } else {
                LBD_DEC_LB_STEP4(uint16_t, dst, res, lb, i + 8u);
                LBD_DEC_LB_STEP4(uint16_t, dst, res, lb, i + 12u);
                prev_valid = 0;
            }
            i += 16u;
            continue;
        }
        LBD_DEC_LB_STEP4(uint16_t, dst, res, lb, i);
        LBD_DEC_LB_STEP4(uint16_t, dst, res, lb, i + 4u);
        __m128i const ov1 = _mm_subs_epu16(hi8, k_plus_7);
        if (_mm_testz_si128(ov1, ov1)) {
            prev       = _mm_loadu_si128((__m128i_u const*)(dst + i));
            prev       = lbd_dag8_u16_avx2(dst, res, i + 8u, hi8, prev);
            prev_valid = 1;
        } else {
            LBD_DEC_LB_STEP4(uint16_t, dst, res, lb, i + 8u);
            LBD_DEC_LB_STEP4(uint16_t, dst, res, lb, i + 12u);
            prev_valid = 0;
        }
        i += 16u;
    }
    if (i + 8u <= n) {
        __m128i const vlb8 = _mm_loadu_si128((__m128i_u const*)(lb + i - 1u));
        __m128i const ov   = _mm_subs_epu16(vlb8, k_plus_7);
        if (_mm_testz_si128(ov, ov)) {
            if (!prev_valid) {
                prev = _mm_loadu_si128((__m128i_u const*)(dst + i - 8u));
            }
            (void)lbd_dag8_u16_avx2(dst, res, i, vlb8, prev);
            i += 8u;
        }
    }
    LBD_DEC_LB_SCALAR_TAIL(uint16_t, dst, res, lb, i, n);
}

static void lbd_dec_lb_4_avx2(
        uint32_t* __restrict dst,
        const uint32_t* __restrict res,
        const uint16_t* __restrict lb,
        size_t n)
{
    dst[0]   = res[0];
    size_t i = 1;
    while (i + 16u <= n) {
        __m256i const vlb  = _mm256_loadu_si256((__m256i_u const*)(lb + i - 1u));
        uint16_t const lb0 = lb[i - 1u];
        __m256i const vc   = _mm256_set1_epi16((int16_t)lb0);
        __m256i const diff = _mm256_xor_si256(vlb, vc);
        if (ZL_UNLIKELY(_mm256_testz_si256(diff, diff))) {
            size_t const dist = (size_t)lb0 + 1u;
            if (lb0 == 0u) {
                __m256i carry = _mm256_set1_epi32((int32_t)dst[i - 1u]);
                for (;;) {
                    __m256i const vr0 = _mm256_loadu_si256((__m256i_u const*)(res + i));
                    __m256i const ps0 = lbd_psum32_avx2(vr0);
                    _mm256_storeu_si256((__m256i_u*)(dst + i), _mm256_add_epi32(carry, ps0));
                    carry = _mm256_add_epi32(carry, lbd_bcast_last32_avx2(ps0));

                    __m256i const vr1 = _mm256_loadu_si256((__m256i_u const*)(res + i + 8u));
                    __m256i const ps1 = lbd_psum32_avx2(vr1);
                    _mm256_storeu_si256((__m256i_u*)(dst + i + 8u), _mm256_add_epi32(carry, ps1));
                    carry = _mm256_add_epi32(carry, lbd_bcast_last32_avx2(ps1));

                    i += 16u;
                    if (i + 16u > n) {
                        break;
                    }
                    __m256i const nxt = _mm256_loadu_si256((__m256i_u const*)(lb + i - 1u));
                    if (!_mm256_testz_si256(nxt, nxt)) {
                        break;
                    }
                }
                continue;
            } else if (dist >= 8u) {
                for (;;) {
                    __m256i const vr0 = _mm256_loadu_si256((__m256i_u const*)(res + i));
                    __m256i const vs0 = _mm256_loadu_si256((__m256i_u const*)(dst + i - dist));
                    _mm256_storeu_si256((__m256i_u*)(dst + i), _mm256_add_epi32(vr0, vs0));

                    __m256i const vr1 = _mm256_loadu_si256((__m256i_u const*)(res + i + 8u));
                    __m256i const vs1 = _mm256_loadu_si256((__m256i_u const*)(dst + i + 8u - dist));
                    _mm256_storeu_si256((__m256i_u*)(dst + i + 8u), _mm256_add_epi32(vr1, vs1));

                    i += 16u;
                    if (i + 16u > n) {
                        break;
                    }
                    __m256i const nxt  = _mm256_loadu_si256((__m256i_u const*)(lb + i - 1u));
                    __m256i const ndif = _mm256_xor_si256(nxt, vc);
                    if (!_mm256_testz_si256(ndif, ndif)) {
                        break;
                    }
                }
                continue;
            } else if (dist >= 4u) {
                for (size_t b = 0; b < 4u; b++, i += 4u) {
                    __m128i const vr = _mm_loadu_si128((__m128i_u const*)(res + i));
                    __m128i const vs = _mm_loadu_si128((__m128i_u const*)(dst + i - dist));
                    _mm_storeu_si128((__m128i_u*)(dst + i), _mm_add_epi32(vr, vs));
                }
                continue;
            }
        }
        size_t const end = i + 16u;
        LBD_DEC_LB_SCALAR_TAIL(uint32_t, dst, res, lb, i, end);
    }
    LBD_DEC_LB_SCALAR_TAIL(uint32_t, dst, res, lb, i, n);
}

static void lbd_dec_lb_8_avx2(
        uint64_t* __restrict dst,
        const uint64_t* __restrict res,
        const uint16_t* __restrict lb,
        size_t n)
{
    dst[0]   = res[0];
    size_t i = 1;
    while (i + 16u <= n) {
        __m256i const vlb  = _mm256_loadu_si256((__m256i_u const*)(lb + i - 1u));
        uint16_t const lb0 = lb[i - 1u];
        __m256i const vc   = _mm256_set1_epi16((int16_t)lb0);
        __m256i const diff = _mm256_xor_si256(vlb, vc);
        if (ZL_UNLIKELY(_mm256_testz_si256(diff, diff))) {
            size_t const dist = (size_t)lb0 + 1u;
            if (lb0 == 0u) {
                __m256i carry = _mm256_set1_epi64x((int64_t)dst[i - 1u]);
                for (;;) {
                    for (size_t b = 0; b < 4u; b++, i += 4u) {
                        __m256i const vr = _mm256_loadu_si256((__m256i_u const*)(res + i));
                        __m256i const ps = lbd_psum64_avx2(vr);
                        _mm256_storeu_si256((__m256i_u*)(dst + i), _mm256_add_epi64(carry, ps));
                        carry = _mm256_add_epi64(carry, _mm256_permute4x64_epi64(ps, 0xFF));
                    }
                    if (i + 16u > n) {
                        break;
                    }
                    __m256i const nxt = _mm256_loadu_si256((__m256i_u const*)(lb + i - 1u));
                    if (!_mm256_testz_si256(nxt, nxt)) {
                        break;
                    }
                }
                continue;
            } else if (dist >= 4u) {
                for (;;) {
                    for (size_t b = 0; b < 4u; b++, i += 4u) {
                        __m256i const vr = _mm256_loadu_si256((__m256i_u const*)(res + i));
                        __m256i const vs = _mm256_loadu_si256((__m256i_u const*)(dst + i - dist));
                        _mm256_storeu_si256((__m256i_u*)(dst + i), _mm256_add_epi64(vr, vs));
                    }
                    if (i + 16u > n) {
                        break;
                    }
                    __m256i const nxt  = _mm256_loadu_si256((__m256i_u const*)(lb + i - 1u));
                    __m256i const ndif = _mm256_xor_si256(nxt, vc);
                    if (!_mm256_testz_si256(ndif, ndif)) {
                        break;
                    }
                }
                continue;
            } else {
                for (size_t b = 0; b < 8u; b++, i += 2u) {
                    __m128i const vr = _mm_loadu_si128((__m128i_u const*)(res + i));
                    __m128i const vs = _mm_loadu_si128((__m128i_u const*)(dst + i - dist));
                    _mm_storeu_si128((__m128i_u*)(dst + i), _mm_add_epi64(vr, vs));
                }
                continue;
            }
        }
        size_t const end = i + 16u;
        LBD_DEC_LB_SCALAR_TAIL(uint64_t, dst, res, lb, i, end);
    }
    LBD_DEC_LB_SCALAR_TAIL(uint64_t, dst, res, lb, i, n);
}

#elif ZL_HAS_WASM_SIMD128

static inline v128_t lbd_psum8_wasm(v128_t v)
{
    v128_t const z = wasm_u8x16_splat(0);
    v = wasm_i8x16_add(
            v, wasm_i8x16_shuffle(z, v, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23));
    v = wasm_i8x16_add(
            v, wasm_i8x16_shuffle(z, v, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27));
    v = wasm_i8x16_add(
            v, wasm_i8x16_shuffle(z, v, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29));
    return wasm_i8x16_add(
            v, wasm_i8x16_shuffle(z, v, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30));
}

static inline v128_t lbd_psum16_wasm(v128_t v)
{
    v128_t const z = wasm_u16x8_splat(0);
    v = wasm_i16x8_add(
            v, wasm_i8x16_shuffle(z, v, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23));
    v = wasm_i16x8_add(
            v, wasm_i8x16_shuffle(z, v, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27));
    return wasm_i16x8_add(
            v, wasm_i8x16_shuffle(z, v, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29));
}

static inline v128_t lbd_psum32_wasm(v128_t v)
{
    v128_t const z = wasm_u32x4_splat(0);
    v = wasm_i32x4_add(
            v, wasm_i8x16_shuffle(z, v, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23));
    return wasm_i32x4_add(
            v, wasm_i8x16_shuffle(z, v, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27));
}

static inline v128_t lbd_psum64_wasm(v128_t v)
{
    v128_t const z = wasm_u64x2_splat(0);
    return wasm_i64x2_add(v, wasm_i64x2_shuffle(z, v, 0, 2));
}

/* wasm_i8x16_swizzle zeroes any lane with unsigned index >= 16, so `intra + 16`
 * directly selects `prev[0..15]` for `intra in [-16..-1]` and zeroes `intra >= 0`.
 * `swizzle(shuf, shuf)` outputs 0 on OOB lanes, so `(shuf & 0x80)` must still be
 * OR'd back in each round to keep resolved lanes >= 16. */
static void lbd_dec_lb_1_wasm(
        uint8_t* __restrict dst,
        const uint8_t* __restrict res,
        const uint16_t* __restrict lb,
        size_t n)
{
    dst[0]      = res[0];
    size_t i    = 1;
    size_t warm = n < 17u ? n : 17u;
    for (; i < warm; i++) {
        dst[i] = (uint8_t)(dst[i - (size_t)lb[i - 1u] - 1u] + res[i]);
    }
    if (i >= n) {
        return;
    }

    v128_t prev            = wasm_v128_load(dst + i - 16u);
    int prev_valid         = 1;
    v128_t const k_lo_15   = wasm_u16x8_const(15, 16, 17, 18, 19, 20, 21, 22);
    v128_t const k_hi_15   = wasm_u16x8_const(23, 24, 25, 26, 27, 28, 29, 30);
    v128_t const km1_lo    = wasm_i16x8_const(-1, 0, 1, 2, 3, 4, 5, 6);
    v128_t const km1_hi    = wasm_i16x8_const(7, 8, 9, 10, 11, 12, 13, 14);
    v128_t const c16       = wasm_u8x16_splat(16u);
    v128_t const msb       = wasm_u8x16_splat(0x80u);

    while (i + 16u <= n) {
        v128_t const vlb_lo = wasm_v128_load(lb + i - 1u);
        v128_t const vlb_hi = wasm_v128_load(lb + i + 7u);
        uint16_t const lb0  = lb[i - 1u];
        v128_t const vc     = wasm_u16x8_splat(lb0);
        v128_t const diff   = wasm_v128_or(
                wasm_v128_xor(vlb_lo, vc), wasm_v128_xor(vlb_hi, vc));
        if (ZL_UNLIKELY(!wasm_v128_any_true(diff))) {
            size_t const dist = (size_t)lb0 + 1u;
            if (lb0 == 0u) {
                v128_t carry = wasm_u8x16_splat(dst[i - 1u]);
                for (;;) {
                    v128_t const vr = wasm_v128_load(res + i);
                    v128_t const ps = lbd_psum8_wasm(vr);
                    prev            = wasm_i8x16_add(carry, ps);
                    wasm_v128_store(dst + i, prev);
                    carry = wasm_i8x16_add(
                            carry,
                            wasm_i8x16_shuffle(
                                    ps, ps, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15));
                    i += 16u;
                    if (i + 16u > n) {
                        break;
                    }
                    v128_t const u = wasm_v128_or(
                            wasm_v128_load(lb + i - 1u), wasm_v128_load(lb + i + 7u));
                    if (wasm_v128_any_true(u)) {
                        break;
                    }
                }
                prev_valid = 1;
                continue;
            } else if (dist >= 16u) {
                for (;;) {
                    v128_t const vr = wasm_v128_load(res + i);
                    v128_t const vs = wasm_v128_load(dst + i - dist);
                    prev            = wasm_i8x16_add(vr, vs);
                    wasm_v128_store(dst + i, prev);
                    i += 16u;
                    if (i + 16u > n) {
                        break;
                    }
                    v128_t const u = wasm_v128_or(
                            wasm_v128_xor(wasm_v128_load(lb + i - 1u), vc),
                            wasm_v128_xor(wasm_v128_load(lb + i + 7u), vc));
                    if (wasm_v128_any_true(u)) {
                        break;
                    }
                }
                prev_valid = 1;
                continue;
            }
        }

        v128_t const over = wasm_v128_or(
                wasm_u16x8_gt(vlb_lo, k_lo_15), wasm_u16x8_gt(vlb_hi, k_hi_15));
        if (ZL_LIKELY(!wasm_v128_any_true(over))) {
            if (ZL_UNLIKELY(!prev_valid)) {
                prev = wasm_v128_load(dst + i - 16u);
            }
            v128_t const q_lo  = wasm_i16x8_sub(km1_lo, vlb_lo);
            v128_t const q_hi  = wasm_i16x8_sub(km1_hi, vlb_hi);
            v128_t const intra = wasm_i8x16_narrow_i16x8(q_lo, q_hi);
            v128_t const vres  = wasm_v128_load(res + i);
            v128_t acc  = wasm_i8x16_add(vres, wasm_i8x16_swizzle(prev, wasm_i8x16_add(intra, c16)));
            v128_t shuf = intra;

            acc  = wasm_i8x16_add(acc, wasm_i8x16_swizzle(acc, shuf));
            shuf = wasm_v128_or(wasm_i8x16_swizzle(shuf, shuf), wasm_v128_and(shuf, msb));
            acc  = wasm_i8x16_add(acc, wasm_i8x16_swizzle(acc, shuf));
            shuf = wasm_v128_or(wasm_i8x16_swizzle(shuf, shuf), wasm_v128_and(shuf, msb));
            acc  = wasm_i8x16_add(acc, wasm_i8x16_swizzle(acc, shuf));
            shuf = wasm_v128_or(wasm_i8x16_swizzle(shuf, shuf), wasm_v128_and(shuf, msb));
            acc  = wasm_i8x16_add(acc, wasm_i8x16_swizzle(acc, shuf));

            wasm_v128_store(dst + i, acc);
            prev       = acc;
            prev_valid = 1;
            i += 16u;
        } else {
            LBD_DEC_LB_STEP4(uint8_t, dst, res, lb, i);
            LBD_DEC_LB_STEP4(uint8_t, dst, res, lb, i + 4u);
            LBD_DEC_LB_STEP4(uint8_t, dst, res, lb, i + 8u);
            LBD_DEC_LB_STEP4(uint8_t, dst, res, lb, i + 12u);
            prev_valid = 0;
            i += 16u;
        }
    }
    LBD_DEC_LB_SCALAR_TAIL(uint8_t, dst, res, lb, i, n);
}

static void lbd_dec_lb_2_wasm(
        uint16_t* __restrict dst,
        const uint16_t* __restrict res,
        const uint16_t* __restrict lb,
        size_t n)
{
    dst[0]      = res[0];
    size_t i    = 1;
    size_t warm = n < 9u ? n : 9u;
    for (; i < warm; i++) {
        dst[i] = (uint16_t)(dst[i - (size_t)lb[i - 1u] - 1u] + res[i]);
    }
    if (i >= n) {
        return;
    }

    v128_t prev            = wasm_v128_load(dst + i - 8u);
    int prev_valid         = 1;
    v128_t const k_plus_7  = wasm_u16x8_const(7, 8, 9, 10, 11, 12, 13, 14);
    v128_t const k_minus_1 = wasm_i16x8_const(-1, 0, 1, 2, 3, 4, 5, 6);
    v128_t const mask_lo   = wasm_u16x8_splat(0x00FFu);
    v128_t const add_hi    = wasm_u16x8_splat(0x0100u);
    v128_t const c16       = wasm_u8x16_splat(16u);
    v128_t const msb       = wasm_u8x16_splat(0x80u);

    while (i + 8u <= n) {
        v128_t const vlb   = wasm_v128_load(lb + i - 1u);
        uint16_t const lb0 = lb[i - 1u];
        v128_t const vc    = wasm_u16x8_splat(lb0);
        v128_t const diff  = wasm_v128_xor(vlb, vc);
        if (ZL_UNLIKELY(!wasm_v128_any_true(diff))) {
            size_t const dist = (size_t)lb0 + 1u;
            if (lb0 == 0u) {
                v128_t carry = wasm_u16x8_splat(dst[i - 1u]);
                for (;;) {
                    v128_t const vr = wasm_v128_load(res + i);
                    v128_t const ps = lbd_psum16_wasm(vr);
                    prev            = wasm_i16x8_add(carry, ps);
                    wasm_v128_store(dst + i, prev);
                    carry = wasm_i16x8_add(
                            carry, wasm_i16x8_shuffle(ps, ps, 7, 7, 7, 7, 7, 7, 7, 7));
                    i += 8u;
                    if (i + 8u > n) {
                        break;
                    }
                    if (wasm_v128_any_true(wasm_v128_load(lb + i - 1u))) {
                        break;
                    }
                }
                prev_valid = 1;
                continue;
            } else if (dist >= 8u) {
                for (;;) {
                    v128_t const vr = wasm_v128_load(res + i);
                    v128_t const vs = wasm_v128_load(dst + i - dist);
                    prev            = wasm_i16x8_add(vr, vs);
                    wasm_v128_store(dst + i, prev);
                    i += 8u;
                    if (i + 8u > n) {
                        break;
                    }
                    if (wasm_v128_any_true(wasm_v128_xor(wasm_v128_load(lb + i - 1u), vc))) {
                        break;
                    }
                }
                prev_valid = 1;
                continue;
            }
        }

        if (ZL_LIKELY(!wasm_v128_any_true(wasm_u16x8_gt(vlb, k_plus_7)))) {
            if (ZL_UNLIKELY(!prev_valid)) {
                prev = wasm_v128_load(dst + i - 8u);
            }
            v128_t const q     = wasm_i16x8_sub(k_minus_1, vlb);
            v128_t const b2    = wasm_v128_and(wasm_i16x8_shl(q, 1), mask_lo);
            v128_t const intra = wasm_i8x16_add(
                    wasm_v128_or(b2, wasm_i16x8_shl(b2, 8)), add_hi);
            v128_t const vres  = wasm_v128_load(res + i);
            v128_t acc  = wasm_i16x8_add(vres, wasm_i8x16_swizzle(prev, wasm_i8x16_add(intra, c16)));
            v128_t shuf = intra;

            acc  = wasm_i16x8_add(acc, wasm_i8x16_swizzle(acc, shuf));
            shuf = wasm_v128_or(wasm_i8x16_swizzle(shuf, shuf), wasm_v128_and(shuf, msb));
            acc  = wasm_i16x8_add(acc, wasm_i8x16_swizzle(acc, shuf));
            shuf = wasm_v128_or(wasm_i8x16_swizzle(shuf, shuf), wasm_v128_and(shuf, msb));
            acc  = wasm_i16x8_add(acc, wasm_i8x16_swizzle(acc, shuf));

            wasm_v128_store(dst + i, acc);
            prev       = acc;
            prev_valid = 1;
            i += 8u;
        } else {
            LBD_DEC_LB_STEP4(uint16_t, dst, res, lb, i);
            LBD_DEC_LB_STEP4(uint16_t, dst, res, lb, i + 4u);
            prev_valid = 0;
            i += 8u;
        }
    }
    LBD_DEC_LB_SCALAR_TAIL(uint16_t, dst, res, lb, i, n);
}

static void lbd_dec_lb_4_wasm(
        uint32_t* __restrict dst,
        const uint32_t* __restrict res,
        const uint16_t* __restrict lb,
        size_t n)
{
    dst[0]   = res[0];
    size_t i = 1;
    while (i + 8u <= n) {
        v128_t const vlb   = wasm_v128_load(lb + i - 1u);
        uint16_t const lb0 = lb[i - 1u];
        v128_t const vc    = wasm_u16x8_splat(lb0);
        if (ZL_UNLIKELY(!wasm_v128_any_true(wasm_v128_xor(vlb, vc)))) {
            size_t const dist = (size_t)lb0 + 1u;
            if (lb0 == 0u) {
                v128_t carry = wasm_u32x4_splat(dst[i - 1u]);
                for (;;) {
                    v128_t const ps0 = lbd_psum32_wasm(wasm_v128_load(res + i));
                    wasm_v128_store(dst + i, wasm_i32x4_add(carry, ps0));
                    carry = wasm_i32x4_add(carry, wasm_i32x4_shuffle(ps0, ps0, 3, 3, 3, 3));

                    v128_t const ps1 = lbd_psum32_wasm(wasm_v128_load(res + i + 4u));
                    wasm_v128_store(dst + i + 4u, wasm_i32x4_add(carry, ps1));
                    carry = wasm_i32x4_add(carry, wasm_i32x4_shuffle(ps1, ps1, 3, 3, 3, 3));

                    i += 8u;
                    if (i + 8u > n || wasm_v128_any_true(wasm_v128_load(lb + i - 1u))) {
                        break;
                    }
                }
                continue;
            } else if (dist >= 4u) {
                for (;;) {
                    wasm_v128_store(
                            dst + i,
                            wasm_i32x4_add(wasm_v128_load(res + i), wasm_v128_load(dst + i - dist)));
                    wasm_v128_store(
                            dst + i + 4u,
                            wasm_i32x4_add(
                                    wasm_v128_load(res + i + 4u),
                                    wasm_v128_load(dst + i + 4u - dist)));
                    i += 8u;
                    if (i + 8u > n || wasm_v128_any_true(wasm_v128_xor(wasm_v128_load(lb + i - 1u), vc))) {
                        break;
                    }
                }
                continue;
            }
        }
        size_t const end = i + 8u;
        LBD_DEC_LB_SCALAR_TAIL(uint32_t, dst, res, lb, i, end);
    }
    LBD_DEC_LB_SCALAR_TAIL(uint32_t, dst, res, lb, i, n);
}

static void lbd_dec_lb_8_wasm(
        uint64_t* __restrict dst,
        const uint64_t* __restrict res,
        const uint16_t* __restrict lb,
        size_t n)
{
    dst[0]   = res[0];
    size_t i = 1;
    while (i + 8u <= n) {
        v128_t const vlb   = wasm_v128_load(lb + i - 1u);
        uint16_t const lb0 = lb[i - 1u];
        v128_t const vc    = wasm_u16x8_splat(lb0);
        if (ZL_UNLIKELY(!wasm_v128_any_true(wasm_v128_xor(vlb, vc)))) {
            size_t const dist = (size_t)lb0 + 1u;
            if (lb0 == 0u) {
                v128_t carry = wasm_u64x2_splat(dst[i - 1u]);
                for (;;) {
                    for (size_t b = 0; b < 4u; b++, i += 2u) {
                        v128_t const ps = lbd_psum64_wasm(wasm_v128_load(res + i));
                        wasm_v128_store(dst + i, wasm_i64x2_add(carry, ps));
                        carry = wasm_i64x2_add(carry, wasm_i64x2_shuffle(ps, ps, 1, 1));
                    }
                    if (i + 8u > n || wasm_v128_any_true(wasm_v128_load(lb + i - 1u))) {
                        break;
                    }
                }
                continue;
            } else {
                for (;;) {
                    for (size_t b = 0; b < 4u; b++, i += 2u) {
                        wasm_v128_store(
                                dst + i,
                                wasm_i64x2_add(
                                        wasm_v128_load(res + i), wasm_v128_load(dst + i - dist)));
                    }
                    if (i + 8u > n || wasm_v128_any_true(wasm_v128_xor(wasm_v128_load(lb + i - 1u), vc))) {
                        break;
                    }
                }
                continue;
            }
        }
        size_t const end = i + 8u;
        LBD_DEC_LB_SCALAR_TAIL(uint64_t, dst, res, lb, i, end);
    }
    LBD_DEC_LB_SCALAR_TAIL(uint64_t, dst, res, lb, i, n);
}

#endif

#if ZL_HAS_AVX2
#    define LBD_DEFINE_LB_FUNC(W, T)                           \
        static int lbd_dec_lb_##W(                             \
                T* __restrict dst,                             \
                const T* __restrict res,                       \
                const uint16_t* __restrict lb,                 \
                size_t n)                                      \
        {                                                      \
            if (ZL_UNLIKELY(!lbd_validate_lb(lb, n - 1u))) {   \
                return 0;                                      \
            }                                                  \
            lbd_dec_lb_##W##_avx2(dst, res, lb, n);            \
            return 1;                                          \
        }
#elif ZL_HAS_WASM_SIMD128
#    define LBD_DEFINE_LB_FUNC(W, T)                           \
        static int lbd_dec_lb_##W(                             \
                T* __restrict dst,                             \
                const T* __restrict res,                       \
                const uint16_t* __restrict lb,                 \
                size_t n)                                      \
        {                                                      \
            if (ZL_UNLIKELY(!lbd_validate_lb(lb, n - 1u))) {   \
                return 0;                                      \
            }                                                  \
            lbd_dec_lb_##W##_wasm(dst, res, lb, n);            \
            return 1;                                          \
        }
#else
#    define LBD_DEFINE_LB_FUNC(W, T)                                \
        static int lbd_dec_lb_##W(                                  \
                T* __restrict dst,                                  \
                const T* __restrict res,                            \
                const uint16_t* __restrict lb,                      \
                size_t n)                                           \
        {                                                           \
            dst[0]               = res[0];                          \
            size_t const checked = n < 65536u ? n : 65536u;         \
            size_t i             = 1;                               \
            for (; i < checked; i++) {                              \
                size_t const d = (size_t)lb[i - 1u] + 1u;           \
                if (ZL_UNLIKELY(d > i)) {                           \
                    return 0;                                       \
                }                                                   \
                dst[i] = (T)(dst[i - d] + res[i]);                  \
            }                                                       \
            LBD_DEC_LB_SCALAR_TAIL(T, dst, res, lb, i, n);          \
            return 1;                                               \
        }
#endif

LBD_DEFINE_LB_FUNC(1, uint8_t)
LBD_DEFINE_LB_FUNC(2, uint16_t)
LBD_DEFINE_LB_FUNC(4, uint32_t)
LBD_DEFINE_LB_FUNC(8, uint64_t)

/* ZS_deltaDecode reads deltas[k - 1] before writing dst[k], so deltas == dst + 1
 * is an exact in-place prefix sum. For narrow elements (1B/2B), chaining
 * vectorized ZS_deltaDecode passes outperforms scalar recurrence; for 4B/8B,
 * a 4x-unrolled register recurrence runs at ~1 cycle/elt in a single pass. */
#define LBD_DEFINE_CONSEC_WIDE(W, T)                                     \
    static void lbd_dec_consec2_##W(                                     \
            T* __restrict dst, const T* __restrict res, size_t n)        \
    {                                                                    \
        T const s0 = res[0];                                             \
        T d1       = res[1];                                             \
        T s1       = (T)(s0 + d1);                                       \
        dst[0]     = s0;                                                 \
        dst[1]     = s1;                                                 \
        size_t i   = 2;                                                  \
        for (; i + 4u <= n; i += 4u) {                                   \
            d1 = (T)(d1 + res[i + 0u]); s1 = (T)(s1 + d1); dst[i + 0u] = s1; \
            d1 = (T)(d1 + res[i + 1u]); s1 = (T)(s1 + d1); dst[i + 1u] = s1; \
            d1 = (T)(d1 + res[i + 2u]); s1 = (T)(s1 + d1); dst[i + 2u] = s1; \
            d1 = (T)(d1 + res[i + 3u]); s1 = (T)(s1 + d1); dst[i + 3u] = s1; \
        }                                                                \
        for (; i < n; i++) {                                             \
            d1     = (T)(d1 + res[i]);                                   \
            s1     = (T)(s1 + d1);                                       \
            dst[i] = s1;                                                 \
        }                                                                \
    }                                                                    \
    static void lbd_dec_consec3_##W(                                     \
            T* __restrict dst, const T* __restrict res, size_t n)        \
    {                                                                    \
        T const s0 = res[0];                                             \
        T d1       = res[1];                                             \
        T d2       = res[2];                                             \
        T const s1 = (T)(s0 + d1);                                       \
        d1         = (T)(d1 + d2);                                       \
        T s2       = (T)(s1 + d1);                                       \
        dst[0]     = s0;                                                 \
        dst[1]     = s1;                                                 \
        dst[2]     = s2;                                                 \
        size_t i   = 3;                                                  \
        for (; i + 4u <= n; i += 4u) {                                   \
            d2 = (T)(d2 + res[i + 0u]); d1 = (T)(d1 + d2); s2 = (T)(s2 + d1); dst[i + 0u] = s2; \
            d2 = (T)(d2 + res[i + 1u]); d1 = (T)(d1 + d2); s2 = (T)(s2 + d1); dst[i + 1u] = s2; \
            d2 = (T)(d2 + res[i + 2u]); d1 = (T)(d1 + d2); s2 = (T)(s2 + d1); dst[i + 2u] = s2; \
            d2 = (T)(d2 + res[i + 3u]); d1 = (T)(d1 + d2); s2 = (T)(s2 + d1); dst[i + 3u] = s2; \
        }                                                                \
        for (; i < n; i++) {                                             \
            d2     = (T)(d2 + res[i]);                                   \
            d1     = (T)(d1 + d2);                                       \
            s2     = (T)(s2 + d1);                                       \
            dst[i] = s2;                                                 \
        }                                                                \
    }

#if ZL_HAS_SSSE3 || ZL_HAS_AVX2 || ZL_HAS_NEON || ZL_HAS_WASM_SIMD128
static void lbd_dec_consec2_1(
        uint8_t* __restrict dst,
        const uint8_t* __restrict res,
        size_t n)
{
    ZS_deltaDecode8(dst + 1, res[1], res + 2, n - 1u);
    ZS_deltaDecode8(dst, res[0], dst + 1, n);
}

static void lbd_dec_consec3_1(
        uint8_t* __restrict dst,
        const uint8_t* __restrict res,
        size_t n)
{
    ZS_deltaDecode8(dst + 2, res[2], res + 3, n - 2u);
    ZS_deltaDecode8(dst + 1, res[1], dst + 2, n - 1u);
    ZS_deltaDecode8(dst, res[0], dst + 1, n);
}

static void lbd_dec_consec2_2(
        uint16_t* __restrict dst,
        const uint16_t* __restrict res,
        size_t n)
{
    ZS_deltaDecode16(dst + 1, res[1], res + 2, n - 1u);
    ZS_deltaDecode16(dst, res[0], dst + 1, n);
}

static void lbd_dec_consec3_2(
        uint16_t* __restrict dst,
        const uint16_t* __restrict res,
        size_t n)
{
    ZS_deltaDecode16(dst + 2, res[2], res + 3, n - 2u);
    ZS_deltaDecode16(dst + 1, res[1], dst + 2, n - 1u);
    ZS_deltaDecode16(dst, res[0], dst + 1, n);
}
#else
LBD_DEFINE_CONSEC_WIDE(1, uint8_t)
LBD_DEFINE_CONSEC_WIDE(2, uint16_t)
#endif

LBD_DEFINE_CONSEC_WIDE(4, uint32_t)
LBD_DEFINE_CONSEC_WIDE(8, uint64_t)

ZL_Report DI_lookback_delta(ZL_Decoder* dictx, const ZL_Input* ins[])
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dictx);
    const ZL_Input* const resIn = ins[0];
    const ZL_Input* const lbIn  = ins[1];
    ZL_ASSERT_EQ(ZL_Input_type(resIn), ZL_Type_numeric);
    ZL_ASSERT_EQ(ZL_Input_type(lbIn), ZL_Type_numeric);

    size_t const eltWidth = ZL_Input_eltWidth(resIn);
    size_t const n        = ZL_Input_numElts(resIn);
    ZL_ERR_IF_NOT(
            eltWidth == 1 || eltWidth == 2 || eltWidth == 4 || eltWidth == 8,
            corruption);
    ZL_ERR_IF_NE(ZL_Input_eltWidth(lbIn), 2, corruption);

    ZL_RBuffer const hdr = ZL_Decoder_getCodecHeader(dictx);
    ZL_ERR_IF_NE(hdr.size, 1, corruption);
    uint8_t const mode = *(const uint8_t*)hdr.start;
    ZL_ERR_IF_GT(mode, LBD_MODE_CONSECUTIVE_3, corruption);

    ZL_Output* const out = ZL_Decoder_create1OutStream(dictx, n, eltWidth);
    ZL_ERR_IF_NULL(out, allocation);

    if (n == 0) {
        ZL_ERR_IF_NE(ZL_Input_numElts(lbIn), 0, corruption);
        ZL_ERR_IF_GE(mode, LBD_MODE_LOOKBACK, corruption);
        ZL_ERR_IF_ERR(ZL_Output_commit(out, 0));
        return ZL_returnValue(1);
    }

    if (mode == LBD_MODE_RAW) {
        ZL_ERR_IF_NE(ZL_Input_numElts(lbIn), 0, corruption);
        memcpy(ZL_Output_ptr(out), ZL_Input_ptr(resIn), n * eltWidth);
    } else if (mode == LBD_MODE_CONSECUTIVE) {
        ZL_ERR_IF_NE(ZL_Input_numElts(lbIn), 0, corruption);
        const uint8_t* const resPtr = (const uint8_t*)ZL_Input_ptr(resIn);
        ZS_deltaDecode(
                ZL_Output_ptr(out),
                resPtr,
                resPtr + eltWidth,
                n,
                eltWidth);
    } else if (mode == LBD_MODE_CONSECUTIVE_2) {
        ZL_ERR_IF_LT(n, 2, corruption);
        ZL_ERR_IF_NE(ZL_Input_numElts(lbIn), 0, corruption);
        switch (eltWidth) {
            case 1:
                lbd_dec_consec2_1(
                        (uint8_t*)ZL_Output_ptr(out),
                        (const uint8_t*)ZL_Input_ptr(resIn),
                        n);
                break;
            case 2:
                lbd_dec_consec2_2(
                        (uint16_t*)ZL_Output_ptr(out),
                        (const uint16_t*)ZL_Input_ptr(resIn),
                        n);
                break;
            case 4:
                lbd_dec_consec2_4(
                        (uint32_t*)ZL_Output_ptr(out),
                        (const uint32_t*)ZL_Input_ptr(resIn),
                        n);
                break;
            case 8:
                lbd_dec_consec2_8(
                        (uint64_t*)ZL_Output_ptr(out),
                        (const uint64_t*)ZL_Input_ptr(resIn),
                        n);
                break;
        }
    } else if (mode == LBD_MODE_CONSECUTIVE_3) {
        ZL_ERR_IF_LT(n, 3, corruption);
        ZL_ERR_IF_NE(ZL_Input_numElts(lbIn), 0, corruption);
        switch (eltWidth) {
            case 1:
                lbd_dec_consec3_1(
                        (uint8_t*)ZL_Output_ptr(out),
                        (const uint8_t*)ZL_Input_ptr(resIn),
                        n);
                break;
            case 2:
                lbd_dec_consec3_2(
                        (uint16_t*)ZL_Output_ptr(out),
                        (const uint16_t*)ZL_Input_ptr(resIn),
                        n);
                break;
            case 4:
                lbd_dec_consec3_4(
                        (uint32_t*)ZL_Output_ptr(out),
                        (const uint32_t*)ZL_Input_ptr(resIn),
                        n);
                break;
            case 8:
                lbd_dec_consec3_8(
                        (uint64_t*)ZL_Output_ptr(out),
                        (const uint64_t*)ZL_Input_ptr(resIn),
                        n);
                break;
        }
    } else {
        ZL_ERR_IF_NE(ZL_Input_numElts(lbIn), n - 1, corruption);
        const uint16_t* const lbPtr = (const uint16_t*)ZL_Input_ptr(lbIn);
        int ok                      = 0;
        switch (eltWidth) {
            case 1:
                ok = lbd_dec_lb_1(
                        (uint8_t*)ZL_Output_ptr(out),
                        (const uint8_t*)ZL_Input_ptr(resIn),
                        lbPtr,
                        n);
                break;
            case 2:
                ok = lbd_dec_lb_2(
                        (uint16_t*)ZL_Output_ptr(out),
                        (const uint16_t*)ZL_Input_ptr(resIn),
                        lbPtr,
                        n);
                break;
            case 4:
                ok = lbd_dec_lb_4(
                        (uint32_t*)ZL_Output_ptr(out),
                        (const uint32_t*)ZL_Input_ptr(resIn),
                        lbPtr,
                        n);
                break;
            case 8:
                ok = lbd_dec_lb_8(
                        (uint64_t*)ZL_Output_ptr(out),
                        (const uint64_t*)ZL_Input_ptr(resIn),
                        lbPtr,
                        n);
                break;
        }
        ZL_ERR_IF_NOT(ok, corruption, "lookback_delta: out-of-range lookback");
    }

    ZL_ERR_IF_ERR(ZL_Output_commit(out, n));
    return ZL_returnValue(1);
}
