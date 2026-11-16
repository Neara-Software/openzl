// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/lookback_delta/decode_lookback_delta_binding.h"

#include <stdint.h>
#include <string.h>

#include "openzl/codecs/delta/decode_delta_kernel.h"
#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
#include "openzl/shared/mem.h"
#include "openzl/shared/portability.h"

enum {
    LBD_MODE_RAW         = 0,
    LBD_MODE_CONSECUTIVE = 1,
    LBD_MODE_LOOKBACK    = 2,
};

static inline int lbd_validate_lb(const uint16_t* lb, size_t nLb)
{
    size_t const checkN = nLb < 65535u ? nLb : 65535u;
    size_t k            = 0;
#if ZL_HAS_AVX2
    __m256i idx        = _mm256_setr_epi16(
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
    __m256i const step = _mm256_set1_epi16(16);
    for (; k + 16u <= checkN; k += 16u) {
        __m256i const v    = _mm256_loadu_si256((const __m256i_u*)(lb + k));
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

static inline size_t lbd_scan_const_run(
        const uint16_t* lb,
        size_t k,
        size_t nLb,
        uint16_t lb0)
{
    size_t runEnd = k + 4u;
#if ZL_HAS_AVX2
    __m256i const target256 = _mm256_set1_epi16((short)lb0);
    while (runEnd + 16u <= nLb) {
        __m256i const v   = _mm256_loadu_si256((const __m256i_u*)(lb + runEnd));
        __m256i const cmp = _mm256_cmpeq_epi16(v, target256);
        if (_mm256_movemask_epi8(cmp) != -1) {
            break;
        }
        runEnd += 16u;
    }
#elif ZL_HAS_WASM_SIMD128
    v128_t const target128 = wasm_u16x8_splat(lb0);
    while (runEnd + 8u <= nLb) {
        v128_t const v   = wasm_v128_load(lb + runEnd);
        v128_t const cmp = wasm_i16x8_eq(v, target128);
        if (!wasm_i16x8_all_true(cmp)) {
            break;
        }
        runEnd += 8u;
    }
#endif
    uint64_t const target64 = (uint64_t)lb0 * 0x0001000100010001ULL;
    while (runEnd + 4u <= nLb && ZL_read64(lb + runEnd) == target64) {
        runEnd += 4u;
    }
    while (runEnd < nLb && lb[runEnd] == lb0) {
        runEnd++;
    }
    return runEnd;
}

#if ZL_HAS_AVX2
#    define LBD_SIMD256_ADD(W, T, AVX2_ADD, LANES_256)                                 while (runRem >= (LANES_256)) {                                                    __m256i const prev =                                                                   _mm256_loadu_si256((const __m256i_u*)(dst + i - d));                   __m256i const r =                                                                      _mm256_loadu_si256((const __m256i_u*)(res + i));                       _mm256_storeu_si256(                                                                   (__m256i_u*)(dst + i), AVX2_ADD(prev, r));                             i += (LANES_256);                                                              runRem -= (LANES_256);                                                     }
#    define LBD_SIMD128_X86_ADD(W, T, SSE2_ADD, LANES_128)                             while (runRem >= (LANES_128)) {                                                    __m128i const prev =                                                                   _mm_loadu_si128((const __m128i_u*)(dst + i - d));                      __m128i const r =                                                                      _mm_loadu_si128((const __m128i_u*)(res + i));                          _mm_storeu_si128((__m128i_u*)(dst + i), SSE2_ADD(prev, r));                    i += (LANES_128);                                                              runRem -= (LANES_128);                                                     }
#endif

#if ZL_HAS_WASM_SIMD128
#    define LBD_SIMD128_WASM_ADD(W, T, WASM_ADD, LANES_128)                            while (runRem >= (LANES_128)) {                                                    v128_t const prev = wasm_v128_load(dst + i - d);                               v128_t const r    = wasm_v128_load(res + i);                                   wasm_v128_store(dst + i, WASM_ADD(prev, r));                                   i += (LANES_128);                                                              runRem -= (LANES_128);                                                     }
#endif

#if ZL_HAS_AVX2
#    define LBD_RUN_BODY(W, T, DELTA_FN, AVX2_ADD, SSE2_ADD, WASM_ADD, L256, L128)         if (d >= (L256)) {                                                                     LBD_SIMD256_ADD(W, T, AVX2_ADD, L256)                                          } else if (d >= (L128)) {                                                              LBD_SIMD128_X86_ADD(W, T, SSE2_ADD, L128)                                      }
#elif ZL_HAS_WASM_SIMD128
#    define LBD_RUN_BODY(W, T, DELTA_FN, AVX2_ADD, SSE2_ADD, WASM_ADD, L256, L128)         if (d >= (L128)) {                                                                     LBD_SIMD128_WASM_ADD(W, T, WASM_ADD, L128)                                     }
#else
#    define LBD_RUN_BODY(W, T, DELTA_FN, AVX2_ADD, SSE2_ADD, WASM_ADD, L256, L128)
#endif

#define LBD_DEFINE_DECODE(                                                             W, T, DELTA_FN, AVX2_ADD, SSE2_ADD, WASM_ADD, L256, L128)                  static int lbd_dec_lb_##W(                                                             T* dst, const T* res, const uint16_t* lb, size_t n)                    {                                                                                  size_t const nLb = n - 1u;                                                     if (ZL_UNLIKELY(!lbd_validate_lb(lb, nLb))) {                                      return 0;                                                                  }                                                                              dst[0]   = res[0];                                                             size_t i = 1;                                                                  while (i < n) {                                                                    size_t const k    = i - 1u;                                                    uint16_t const lb0 = lb[k];                                                    uint64_t const target64 =                                                              (uint64_t)lb0 * 0x0001000100010001ULL;                                 if (k + 4u <= nLb && ZL_read64(lb + k) == target64) {                              size_t const runEnd = lbd_scan_const_run(lb, k, nLb, lb0);                     size_t runRem       = runEnd - k;                                              size_t const d      = (size_t)lb0 + 1u;                                        if (d == 1u) {                                                                     DELTA_FN(dst + i - 1u, dst[i - 1u], res + i, runRem + 1u);                     i += runRem;                                                               } else {                                                                           LBD_RUN_BODY(                                                                          W,                                                                             T,                                                                             DELTA_FN,                                                                      AVX2_ADD,                                                                      SSE2_ADD,                                                                      WASM_ADD,                                                                      L256,                                                                          L128)                                                                  for (; runRem > 0u; runRem--, i++) {                                               dst[i] = (T)(dst[i - d] + res[i]);                                         }                                                                          }                                                                          } else {                                                                           size_t const d = (size_t)lb0 + 1u;                                             dst[i]         = (T)(dst[i - d] + res[i]);                                     i++;                                                                       }                                                                          }                                                                              return 1;                                                                  }

LBD_DEFINE_DECODE(
        1,
        uint8_t,
        ZS_deltaDecode8,
        _mm256_add_epi8,
        _mm_add_epi8,
        wasm_i8x16_add,
        32u,
        16u)
LBD_DEFINE_DECODE(
        2,
        uint16_t,
        ZS_deltaDecode16,
        _mm256_add_epi16,
        _mm_add_epi16,
        wasm_i16x8_add,
        16u,
        8u)
LBD_DEFINE_DECODE(
        4,
        uint32_t,
        ZS_deltaDecode32,
        _mm256_add_epi32,
        _mm_add_epi32,
        wasm_i32x4_add,
        8u,
        4u)
LBD_DEFINE_DECODE(
        8,
        uint64_t,
        ZS_deltaDecode64,
        _mm256_add_epi64,
        _mm_add_epi64,
        wasm_i64x2_add,
        4u,
        2u)

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
    ZL_ERR_IF_GT(mode, LBD_MODE_LOOKBACK, corruption);

    ZL_Output* const out = ZL_Decoder_create1OutStream(dictx, n, eltWidth);
    ZL_ERR_IF_NULL(out, allocation);

    if (n == 0) {
        ZL_ERR_IF_NE(ZL_Input_numElts(lbIn), 0, corruption);
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
