// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/hilbert3/encode_hilbert3_binding.h"

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

/* Morton -> Hilbert, 3 dims, `bits` levels (reference: tools_hk/hk16.c). */
static uint64_t mortonToHilbert3(uint64_t h, uint32_t bits)
{
    uint32_t block = bits * 3 - 3;
    uint32_t hcode = (uint32_t)((h >> block) & 7);
    uint32_t shift = 0, signs = 0;
    while (block > 0) {
        block -= 3;
        hcode <<= 2;
        uint32_t mcode = (0x20212021u >> hcode) & 3;
        shift          = (0x48u >> (7 - shift - mcode)) & 3;
        signs          = (signs | (signs << 3)) >> mcode;
        signs          = (signs ^ (0x53560300u >> hcode)) & 7;
        mcode          = (uint32_t)((h >> block) & 7);
        hcode          = mcode;
        hcode          = ((hcode | (hcode << 3)) >> shift) & 7;
        hcode ^= signs;
        h ^= ((uint64_t)(mcode ^ hcode)) << block;
    }
    uint64_t const mask = 0x492492492492ull;
    h ^= (h >> 1) & mask;
    h ^= (h & mask) >> 1;
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
    for (size_t i = 0; i < n; i++) {
        const uint8_t* const p = src + 6 * i;
        uint64_t const m       = split3(ZL_readLE16(p))
                | (split3(ZL_readLE16(p + 2)) << 1)
                | (split3(ZL_readLE16(p + 4)) << 2);
        uint64_t const k = mortonToHilbert3(m, 16);
        dst[i]           = k - prev;
        prev             = k;
    }
    ZL_ERR_IF_ERR(ZL_Output_commit(out, n));
    return ZL_returnSuccess();
}

/* 32-level 3D Hilbert key of (x, y, z) as 32 base-8 digits, most significant
 * last (reference: tools_hk/hk32.c, same math as mortonToHilbert3). Returns
 * the 96-bit key as hi (32 bits) and lo (64 bits). */
static void hilbert3Key32(uint32_t x, uint32_t y, uint32_t z, uint32_t* hi, uint64_t* lo)
{
    uint32_t d[32];
    for (int i = 0; i < 32; i++) {
        d[i] = ((x >> i) & 1) | (((y >> i) & 1) << 1) | (((z >> i) & 1) << 2);
    }
    uint32_t hcode = d[31];
    uint32_t shift = 0, signs = 0;
    for (int b = 30; b >= 0; b--) {
        hcode <<= 2;
        uint32_t const mcode = (0x20212021u >> hcode) & 3;
        shift                = (0x48u >> (7 - shift - mcode)) & 3;
        signs                = (signs | (signs << 3)) >> mcode;
        signs                = (signs ^ (0x53560300u >> hcode)) & 7;
        hcode                = (((d[b] | (d[b] << 3)) >> shift) & 7) ^ signs;
        d[b]                 = hcode;
    }
    uint64_t l = 0;
    uint32_t h = 0;
    for (int i = 0; i < 32; i++) {
        uint32_t v = d[i];
        v ^= (v >> 1) & 2;
        v ^= (v & 2) >> 1;
        if (i < 21) {
            l |= (uint64_t)v << (3 * i);
        } else if (i == 21) {
            l |= (uint64_t)(v & 1) << 63;
            h |= v >> 1;
        } else {
            h |= v << (3 * i - 64);
        }
    }
    *hi = h;
    *lo = l;
}

/* Writes the gap stream for n records to dst (room for 3 * n words) and
 * returns the number of words written. */
static size_t hilbert3EncodeI32(uint64_t* dst, const uint8_t* src, size_t n)
{
    size_t o     = 0;
    uint32_t phi = 0;
    uint64_t plo = 0;
    for (size_t i = 0; i < n; i++) {
        const uint8_t* const p = src + 12 * i;
        uint32_t hi;
        uint64_t lo;
        hilbert3Key32(
                ZL_readLE32(p) ^ 0x80000000u,
                ZL_readLE32(p + 4) ^ 0x80000000u,
                ZL_readLE32(p + 8) ^ 0x80000000u,
                &hi,
                &lo);
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
