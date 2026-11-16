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
    uint32_t phi           = 0;
    uint64_t plo           = 0;
    Hilbert3EncTop32 cache = { .prevTopKey = UINT64_MAX };
    for (size_t i = 0; i < n; i++) {
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
