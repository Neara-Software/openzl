// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/morton3/encode_morton3_binding.h"

#include <stdint.h>

#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
#include "openzl/shared/mem.h"

/* Spread the 16 bits of a to every third bit position. */
static uint64_t split3(uint64_t a)
{
    a = (a | (a << 16)) & 0x0000FF0000FFull;
    a = (a | (a << 8)) & 0x00F00F00F00Full;
    a = (a | (a << 4)) & 0x0C30C30C30C3ull;
    a = (a | (a << 2)) & 0x249249249249ull;
    return a;
}

/* 48-bit Morton key of 16-bit x, y, z (x lowest). */
static uint64_t morton3Key16(uint32_t x, uint32_t y, uint32_t z)
{
    return split3(x & 0xFFFF) | (split3(y & 0xFFFF) << 1)
            | (split3(z & 0xFFFF) << 2);
}

static void morton3EncodeU16(uint64_t* dst, const uint8_t* src, size_t n)
{
    uint64_t prev = 0;
    for (size_t i = 0; i < n; i++) {
        const uint8_t* const p = src + 6 * i;
        uint64_t const k       = morton3Key16(
                ZL_readLE16(p), ZL_readLE16(p + 2), ZL_readLE16(p + 4));
        dst[i] = k - prev;
        prev   = k;
    }
}

ZL_Report
EI_morton3_u16(ZL_Encoder* eictx, const ZL_Input* ins[], size_t nbIns)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(eictx);
    ZL_ASSERT_EQ(nbIns, 1);
    const ZL_Input* const in = ins[0];
    ZL_ASSERT_EQ(ZL_Input_type(in), ZL_Type_struct);
    ZL_ERR_IF_NE(
            ZL_Input_eltWidth(in),
            6,
            node_invalid_input,
            "morton3_u16 requires 6-byte records");
    size_t const n           = ZL_Input_numElts(in);
    const uint8_t* const src = (const uint8_t*)ZL_Input_ptr(in);

    ZL_Output* const out = ZL_Encoder_createTypedStream(eictx, 0, n, 8);
    ZL_ERR_IF_NULL(out, allocation);
    morton3EncodeU16((uint64_t*)ZL_Output_ptr(out), src, n);
    ZL_ERR_IF_ERR(ZL_Output_commit(out, n));
    return ZL_returnSuccess();
}

/* Writes the gap stream for n records to dst (room for 3 * n words) and
 * returns the number of words written. The 96-bit key is the 48-bit key of
 * the low 16 bits of each axis, then the 48-bit key of the high 16 bits. */
static size_t morton3EncodeI32(uint64_t* dst, const uint8_t* src, size_t n)
{
    size_t o     = 0;
    uint32_t phi = 0;
    uint64_t plo = 0;
    for (size_t i = 0; i < n; i++) {
        const uint8_t* const p = src + 12 * i;
        uint32_t const x       = ZL_readLE32(p) ^ 0x80000000u;
        uint32_t const y       = ZL_readLE32(p + 4) ^ 0x80000000u;
        uint32_t const z       = ZL_readLE32(p + 8) ^ 0x80000000u;
        uint64_t const kl      = morton3Key16(x, y, z);
        uint64_t const kh      = morton3Key16(x >> 16, y >> 16, z >> 16);
        uint64_t const lo      = kl | (kh << 48);
        uint32_t const hi      = (uint32_t)(kh >> 16);
        uint64_t const dlo     = lo - plo;
        uint32_t const dhi     = hi - phi - (lo < plo);
        int const down         = hi < phi || (hi == phi && lo < plo);
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
EI_morton3_i32(ZL_Encoder* eictx, const ZL_Input* ins[], size_t nbIns)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(eictx);
    ZL_ASSERT_EQ(nbIns, 1);
    const ZL_Input* const in = ins[0];
    ZL_ASSERT_EQ(ZL_Input_type(in), ZL_Type_struct);
    ZL_ERR_IF_NE(
            ZL_Input_eltWidth(in),
            12,
            node_invalid_input,
            "morton3_i32 requires 12-byte records");
    size_t const n = ZL_Input_numElts(in);

    ZL_Output* const out = ZL_Encoder_createTypedStream(eictx, 0, 3 * n, 8);
    ZL_ERR_IF_NULL(out, allocation);
    size_t const nbWords = morton3EncodeI32(
            (uint64_t*)ZL_Output_ptr(out), (const uint8_t*)ZL_Input_ptr(in), n);
    ZL_ERR_IF_ERR(ZL_Output_commit(out, nbWords));
    return ZL_returnSuccess();
}
