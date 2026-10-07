// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/morton3/decode_morton3_binding.h"

#include <stdint.h>

#include "openzl/codecs/morton3/morton3_table.h"
#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
#include "openzl/shared/mem.h"

/* 48 Morton key bits -> x | y << 16 | z << 32, 4 levels per lookup. */
ZL_FORCE_INLINE uint64_t compact48(uint64_t k)
{
    return ZL_morton3_T12[k & 4095] | (ZL_morton3_T12[(k >> 12) & 4095] << 4)
            | (ZL_morton3_T12[(k >> 24) & 4095] << 8)
            | (ZL_morton3_T12[(k >> 36) & 4095] << 12);
}

static void morton3Decode(uint8_t* dst, const uint64_t* gaps, size_t n)
{
    uint64_t k = 0;
    for (size_t i = 0; i < n; i++) {
        k += gaps[i];
        uint64_t const a = compact48(k);
        ZL_writeLE16(dst + 6 * i, (uint16_t)a);
        ZL_writeLE16(dst + 6 * i + 2, (uint16_t)(a >> 16));
        ZL_writeLE16(dst + 6 * i + 4, (uint16_t)(a >> 32));
    }
}

ZL_Report DI_morton3_u16(ZL_Decoder* dictx, const ZL_Input* ins[])
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dictx);
    const ZL_Input* const in = ins[0];
    ZL_ASSERT_EQ(ZL_Input_type(in), ZL_Type_numeric);
    ZL_ERR_IF_NE(ZL_Input_eltWidth(in), 8, corruption);
    size_t const n = ZL_Input_numElts(in);

    ZL_Output* const out = ZL_Decoder_create1OutStream(dictx, n, 6);
    ZL_ERR_IF_NULL(out, allocation);
    morton3Decode(
            (uint8_t*)ZL_Output_ptr(out), (const uint64_t*)ZL_Input_ptr(in), n);
    ZL_ERR_IF_ERR(ZL_Output_commit(out, n));
    return ZL_returnSuccess();
}

/* ---- morton3_i32 ---- */

/* Returns the number of points written, or SIZE_MAX if the last escape is
 * truncated. */
static size_t morton3DecodeI32(uint8_t* dst, const uint64_t* g, size_t n)
{
    uint32_t hi = 0;
    uint64_t lo = 0;
    size_t np = 0, i = 0;
    while (i < n) {
        uint64_t const v = g[i];
        if (ZL_UNLIKELY(v == UINT64_MAX)) {
            if (i + 3 > n) {
                return SIZE_MAX;
            }
            hi = (uint32_t)g[i + 1];
            lo = g[i + 2];
            i += 3;
        } else {
            uint64_t const l = lo + v;
            hi += l < lo;
            lo = l;
            i += 1;
        }
        uint64_t const a = compact48(lo);
        uint64_t const b = compact48((lo >> 48) | ((uint64_t)hi << 16));
        uint8_t* const p = dst + 12 * np;
        ZL_writeLE32(p, ((uint32_t)(a & 0xFFFF) | (uint32_t)(b << 16)) ^ 0x80000000u);
        ZL_writeLE32(p + 4,
                     ((uint32_t)((a >> 16) & 0xFFFF) | (uint32_t)(b & 0xFFFF0000))
                             ^ 0x80000000u);
        ZL_writeLE32(p + 8,
                     ((uint32_t)(a >> 32) | (uint32_t)((b >> 16) & 0xFFFF0000))
                             ^ 0x80000000u);
        np++;
    }
    return np;
}

ZL_Report DI_morton3_i32(ZL_Decoder* dictx, const ZL_Input* ins[])
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dictx);
    const ZL_Input* const in = ins[0];
    ZL_ASSERT_EQ(ZL_Input_type(in), ZL_Type_numeric);
    ZL_ERR_IF_NE(ZL_Input_eltWidth(in), 8, corruption);
    size_t const n = ZL_Input_numElts(in);

    ZL_Output* const out = ZL_Decoder_create1OutStream(dictx, n, 12);
    ZL_ERR_IF_NULL(out, allocation);
    size_t const np = morton3DecodeI32(
            (uint8_t*)ZL_Output_ptr(out), (const uint64_t*)ZL_Input_ptr(in), n);
    ZL_ERR_IF_EQ(np, SIZE_MAX, corruption, "morton3_i32: truncated escape");
    ZL_ERR_IF_ERR(ZL_Output_commit(out, np));
    return ZL_returnSuccess();
}
