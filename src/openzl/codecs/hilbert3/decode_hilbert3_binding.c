// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/hilbert3/decode_hilbert3_binding.h"

#include <stdint.h>

#include "openzl/codecs/hilbert3/hilbert3_table.h"
#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
#include "openzl/shared/mem.h"

static void writeXYZ(uint8_t* dst, uint32_t m, uint64_t A)
{
    uint32_t x = (m & 1) << 15, y = ((m >> 1) & 1) << 15, z = (m >> 2) << 15;
    for (int L = 0; L < 5; L++) {
        uint32_t const e = (uint32_t)(A >> (9 * L)) & 511;
        x |= (e & 7) << (3 * L);
        y |= ((e >> 3) & 7) << (3 * L);
        z |= ((e >> 6) & 7) << (3 * L);
    }
    ZL_writeLE16(dst, (uint16_t)x);
    ZL_writeLE16(dst + 2, (uint16_t)y);
    ZL_writeLE16(dst + 4, (uint16_t)z);
}

/* Decode the 15 low key bits of h below the top digit into 5 packed 9-bit
 * groups (x|y<<3|z<<6 per group, group L = levels 3L..3L+2). */
static uint64_t decodeLow(uint64_t h, uint32_t s)
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
    for (; i + 4 <= n; i += 4) {
        uint64_t const h0 = k += gaps[i];
        uint64_t const h1 = k += gaps[i + 1];
        uint64_t const h2 = k += gaps[i + 2];
        uint64_t const h3 = k += gaps[i + 3];
        uint32_t const d0 = (h0 >> 45) & 7, d1 = (h1 >> 45) & 7,
                       d2 = (h2 >> 45) & 7, d3 = (h3 >> 45) & 7;
        uint32_t s0 = ZL_hilbert3_topS[d0], s1 = ZL_hilbert3_topS[d1],
                 s2 = ZL_hilbert3_topS[d2], s3 = ZL_hilbert3_topS[d3];
        uint64_t A0 = 0, A1 = 0, A2 = 0, A3 = 0;
        for (int L = 4; L >= 0; L--) {
            uint32_t const sh = 9 * (uint32_t)L;
            uint32_t const e0 = ZL_hilbert3_T3[s0 * 512 + ((h0 >> sh) & 511)];
            uint32_t const e1 = ZL_hilbert3_T3[s1 * 512 + ((h1 >> sh) & 511)];
            uint32_t const e2 = ZL_hilbert3_T3[s2 * 512 + ((h2 >> sh) & 511)];
            uint32_t const e3 = ZL_hilbert3_T3[s3 * 512 + ((h3 >> sh) & 511)];
            A0 |= (uint64_t)(e0 & 511) << sh;
            A1 |= (uint64_t)(e1 & 511) << sh;
            A2 |= (uint64_t)(e2 & 511) << sh;
            A3 |= (uint64_t)(e3 & 511) << sh;
            s0 = e0 >> 9;
            s1 = e1 >> 9;
            s2 = e2 >> 9;
            s3 = e3 >> 9;
        }
        writeXYZ(dst + 6 * i, ZL_hilbert3_topM[d0], A0);
        writeXYZ(dst + 6 * (i + 1), ZL_hilbert3_topM[d1], A1);
        writeXYZ(dst + 6 * (i + 2), ZL_hilbert3_topM[d2], A2);
        writeXYZ(dst + 6 * (i + 3), ZL_hilbert3_topM[d3], A3);
    }
    for (; i < n; i++) {
        uint64_t const h = k += gaps[i];
        uint32_t const d = (h >> 45) & 7;
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
