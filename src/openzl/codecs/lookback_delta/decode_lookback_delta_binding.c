// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/lookback_delta/decode_lookback_delta_binding.h"

#include <stdint.h>
#include <string.h>

#include "openzl/codecs/delta/decode_delta_kernel.h"
#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
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

#define LBD_DEFINE_DECODE(W, T)                                                \
    static int lbd_dec_lb_##W(                                                 \
            T* dst, const T* res, const uint16_t* lb, size_t n)                \
    {                                                                          \
        if (ZL_UNLIKELY(!lbd_validate_lb(lb, n - 1u))) {                       \
            return 0;                                                          \
        }                                                                      \
        dst[0] = res[0];                                                       \
        for (size_t i = 1; i < n; i++) {                                       \
            dst[i] = (T)(dst[i - (size_t)lb[i - 1u] - 1u] + res[i]);           \
        }                                                                      \
        return 1;                                                              \
    }

LBD_DEFINE_DECODE(1, uint8_t)
LBD_DEFINE_DECODE(2, uint16_t)
LBD_DEFINE_DECODE(4, uint32_t)
LBD_DEFINE_DECODE(8, uint64_t)

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
