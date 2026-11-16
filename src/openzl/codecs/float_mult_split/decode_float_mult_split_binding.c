// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/float_mult_split/decode_float_mult_split_binding.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
#include "openzl/shared/mem.h"

static inline uint32_t fms_f32_to_ordered(float x)
{
    uint32_t b;
    memcpy(&b, &x, sizeof(b));
    uint32_t const mask = (uint32_t)(-(int32_t)(b >> 31)) | 0x80000000u;
    return b ^ mask;
}

static inline float fms_ordered_to_f32(uint32_t u)
{
    uint32_t const mask = (uint32_t)(-(int32_t)((u >> 31) ^ 1u)) | 0x80000000u;
    uint32_t const b    = u ^ mask;
    float x;
    memcpy(&x, &b, sizeof(x));
    return x;
}

static inline uint64_t fms_f64_to_ordered(double x)
{
    uint64_t b;
    memcpy(&b, &x, sizeof(b));
    uint64_t const mask = (uint64_t)(-(int64_t)(b >> 63)) | 0x8000000000000000ULL;
    return b ^ mask;
}

static inline double fms_ordered_to_f64(uint64_t u)
{
    uint64_t const mask = (uint64_t)(-(int64_t)((u >> 63) ^ 1ULL)) | 0x8000000000000000ULL;
    uint64_t const b    = u ^ mask;
    double x;
    memcpy(&x, &b, sizeof(x));
    return x;
}

static inline float fms_i32_bits_to_int_float(uint32_t u)
{
    uint32_t const sign    = u >> 31;
    uint32_t const absInt  = u ^ (uint32_t)(-(int32_t)sign);
    uint32_t const gpi     = 1u << 24;
    uint32_t const gpiBits = 0x4b800000u;
    uint32_t absBits;
    if (absInt < gpi) {
        float const absF = (float)absInt;
        memcpy(&absBits, &absF, sizeof(absBits));
    } else {
        absBits = gpiBits + (absInt - gpi);
    }
    uint32_t const b = absBits | (sign << 31);
    float x;
    memcpy(&x, &b, sizeof(x));
    return x;
}

static inline double fms_i64_bits_to_int_float(uint64_t u)
{
    uint64_t const sign    = u >> 63;
    uint64_t const absInt  = u ^ (uint64_t)(-(int64_t)sign);
    uint64_t const gpi     = 1ULL << 53;
    uint64_t const gpiBits = 0x4340000000000000ULL;
    uint64_t absBits;
    if (absInt < gpi) {
        double const absD = (double)absInt;
        memcpy(&absBits, &absD, sizeof(absBits));
    } else {
        absBits = gpiBits + (absInt - gpi);
    }
    uint64_t const b = absBits | (sign << 63);
    double x;
    memcpy(&x, &b, sizeof(x));
    return x;
}

static void fms_decode_f32(
        uint32_t* restrict dst,
        const uint32_t* restrict mult,
        const uint32_t* restrict adj,
        size_t n,
        float base)
{
    for (size_t i = 0; i < n; i++) {
        float const m      = fms_i32_bits_to_int_float(mult[i]);
        float const approx = m * base;
        uint32_t const ord = (uint32_t)(fms_f32_to_ordered(approx) + adj[i]);
        float const val    = fms_ordered_to_f32(ord);
        memcpy(&dst[i], &val, sizeof(val));
    }
}

static void fms_decode_f64(
        uint64_t* restrict dst,
        const uint64_t* restrict mult,
        const uint64_t* restrict adj,
        size_t n,
        double base)
{
    for (size_t i = 0; i < n; i++) {
        double const m     = fms_i64_bits_to_int_float(mult[i]);
        double const approx = m * base;
        uint64_t const ord = (uint64_t)(fms_f64_to_ordered(approx) + adj[i]);
        double const val   = fms_ordered_to_f64(ord);
        memcpy(&dst[i], &val, sizeof(val));
    }
}

ZL_Report DI_float_mult_split(ZL_Decoder* dictx, const ZL_Input* ins[])
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dictx);
    const ZL_Input* const multIn = ins[0];
    const ZL_Input* const adjIn  = ins[1];
    ZL_ASSERT_EQ(ZL_Input_type(multIn), ZL_Type_numeric);
    ZL_ASSERT_EQ(ZL_Input_type(adjIn), ZL_Type_numeric);

    ZL_RBuffer const hdr = ZL_Decoder_getCodecHeader(dictx);
    ZL_ERR_IF_LT(hdr.size, 1, corruption);
    const uint8_t* const ptr = (const uint8_t*)hdr.start;

    uint8_t const flags  = ptr[0];
    unsigned const wLog2 = flags & 3u;
    unsigned const hasBase = (flags >> 2) & 1u;
    ZL_ERR_IF_NE(flags >> 3, 0, corruption);
    ZL_ERR_IF_NOT(wLog2 == 2u || wLog2 == 3u, corruption);

    size_t const eltWidth = (size_t)1u << wLog2;
    ZL_ERR_IF_NE(ZL_Input_eltWidth(multIn), eltWidth, corruption);
    ZL_ERR_IF_NE(ZL_Input_eltWidth(adjIn), eltWidth, corruption);

    size_t const n    = ZL_Input_numElts(multIn);
    size_t const adjN = ZL_Input_numElts(adjIn);

    ZL_Output* const out = ZL_Decoder_create1OutStream(dictx, n, eltWidth);
    ZL_ERR_IF_NULL(out, allocation);

    if (!hasBase) {
        ZL_ERR_IF_NE(hdr.size, 1u, corruption);
        ZL_ERR_IF_NE(adjN, 0u, corruption);
        if (n > 0) {
            memcpy(ZL_Output_ptr(out), ZL_Input_ptr(multIn), n * eltWidth);
        }
        ZL_ERR_IF_ERR(ZL_Output_commit(out, n));
        return ZL_returnValue(1);
    }

    ZL_ERR_IF_NE(hdr.size, 1u + eltWidth, corruption);
    ZL_ERR_IF_NE(adjN, n, corruption);

    if (eltWidth == 4) {
        uint32_t const baseBits = ZL_readLE32(ptr + 1);
        float base;
        memcpy(&base, &baseBits, sizeof(base));
        ZL_ERR_IF_NOT(isfinite(base) && base > 0.0f, corruption);
        if (n > 0) {
            fms_decode_f32(
                    (uint32_t*)ZL_Output_ptr(out),
                    (const uint32_t*)ZL_Input_ptr(multIn),
                    (const uint32_t*)ZL_Input_ptr(adjIn),
                    n,
                    base);
        }
    } else {
        uint64_t const baseBits = ZL_readLE64(ptr + 1);
        double base;
        memcpy(&base, &baseBits, sizeof(base));
        ZL_ERR_IF_NOT(isfinite(base) && base > 0.0, corruption);
        if (n > 0) {
            fms_decode_f64(
                    (uint64_t*)ZL_Output_ptr(out),
                    (const uint64_t*)ZL_Input_ptr(multIn),
                    (const uint64_t*)ZL_Input_ptr(adjIn),
                    n,
                    base);
        }
    }

    ZL_ERR_IF_ERR(ZL_Output_commit(out, n));
    return ZL_returnValue(1);
}
