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

/* ---- hilbert3_i32 ---- */

typedef struct {
    uint32_t s, x, y, z;
} Hilbert3Top;

/* Digit L (0..9) of the low 90 key bits, 3 levels per digit. */
static uint32_t keyDigit(uint32_t hi, uint64_t lo, int L)
{
    if (L < 7) {
        return (uint32_t)(lo >> (9 * L)) & 511;
    }
    if (L == 7) {
        return (uint32_t)((lo >> 63) | ((uint64_t)hi << 1)) & 511;
    }
    return (hi >> (9 * L - 64)) & 511;
}

/* Walk the top 2 levels and digits 9..4 (key bits >= 36): the state and xyz
 * bits shared by every key with the same hi and lo >> 36. */
ZL_FORCE_NOINLINE void walkTop(Hilbert3Top* t, uint32_t hi, uint64_t lo)
{
    uint32_t const d = hi >> 26;
    uint32_t const m = ZL_hilbert3_top2M[d];
    uint32_t s = ZL_hilbert3_top2S[d];
    uint32_t x = (m & 3) << 30, y = ((m >> 2) & 3) << 30, z = (m >> 4) << 30;
#define H3_TOP(L)                                                      \
    do {                                                               \
        uint32_t const e_ = ZL_hilbert3_T3[s * 512 + keyDigit(hi, lo, L)]; \
        s                 = e_ >> 9;                                   \
        x |= (e_ & 7) << (3 * L);                                      \
        y |= ((e_ >> 3) & 7) << (3 * L);                               \
        z |= ((e_ >> 6) & 7) << (3 * L);                               \
    } while (0)
    H3_TOP(9);
    H3_TOP(8);
    H3_TOP(7);
    H3_TOP(6);
    H3_TOP(5);
    H3_TOP(4);
#undef H3_TOP
    t->s = s;
    t->x = x;
    t->y = y;
    t->z = z;
}

typedef struct {
    Hilbert3Top top;
    uint32_t topHi; /* hi and lo >> 36 of the key that top was walked for */
    uint64_t topLo;
    uint32_t hi;
    uint64_t lo;
} Hilbert3State;

/* Advance to the next key (the caller checks an escape has 2 more words). */
ZL_FORCE_INLINE size_t nextKey(Hilbert3State* st, const uint64_t* g, size_t i)
{
    uint64_t const v = g[i];
    if (ZL_UNLIKELY(v == UINT64_MAX)) {
        st->hi = (uint32_t)g[i + 1];
        st->lo = g[i + 2];
        i += 3;
    } else {
        uint64_t const lo = st->lo + v;
        st->hi += lo < st->lo;
        st->lo = lo;
        i += 1;
    }
    if (ZL_UNLIKELY(st->hi != st->topHi || (st->lo >> 36) != st->topLo)) {
        st->topHi = st->hi;
        st->topLo = st->lo >> 36;
        walkTop(&st->top, st->hi, st->lo);
    }
    return i;
}

static void writeXYZ32(uint8_t* dst, uint32_t x, uint32_t y, uint32_t z)
{
    ZL_writeLE32(dst, x ^ 0x80000000u);
    ZL_writeLE32(dst + 4, y ^ 0x80000000u);
    ZL_writeLE32(dst + 8, z ^ 0x80000000u);
}

#define H3_LOW(s, x, y, z, lo, L)                                           \
    do {                                                                    \
        uint32_t const e_ =                                                 \
                ZL_hilbert3_T3[s * 512 + ((uint32_t)((lo) >> (9 * L)) & 511)]; \
        s = e_ >> 9;                                                        \
        x |= (e_ & 7) << (3 * L);                                           \
        y |= ((e_ >> 3) & 7) << (3 * L);                                    \
        z |= ((e_ >> 6) & 7) << (3 * L);                                    \
    } while (0)

/* Returns the number of points written, or SIZE_MAX if the last escape is
 * truncated. Consecutive sorted keys share their high digits, so the top
 * walk runs only when key bits >= 36 change; digits 3..0 run every point,
 * 4 points interleaved. */
static size_t hilbert3DecodeI32(uint8_t* dst, const uint64_t* g, size_t n)
{
    Hilbert3State st = { .topLo = UINT64_MAX }; /* topLo never matches */
    size_t np = 0, i = 0;
    while (i + 12 <= n) {
        uint32_t s[4], x[4], y[4], z[4];
        uint64_t lo[4];
        for (int j = 0; j < 4; j++) {
            i     = nextKey(&st, g, i);
            s[j]  = st.top.s;
            x[j]  = st.top.x;
            y[j]  = st.top.y;
            z[j]  = st.top.z;
            lo[j] = st.lo;
        }
        for (int L = 3; L >= 0; L--) {
            for (int j = 0; j < 4; j++) {
                H3_LOW(s[j], x[j], y[j], z[j], lo[j], L);
            }
        }
        for (int j = 0; j < 4; j++) {
            writeXYZ32(dst + 12 * (np + (size_t)j), x[j], y[j], z[j]);
        }
        np += 4;
    }
    while (i < n) {
        if (g[i] == UINT64_MAX && i + 3 > n) {
            return SIZE_MAX;
        }
        i          = nextKey(&st, g, i);
        uint32_t s = st.top.s, x = st.top.x, y = st.top.y, z = st.top.z;
        H3_LOW(s, x, y, z, st.lo, 3);
        H3_LOW(s, x, y, z, st.lo, 2);
        H3_LOW(s, x, y, z, st.lo, 1);
        H3_LOW(s, x, y, z, st.lo, 0);
        writeXYZ32(dst + 12 * np, x, y, z);
        np++;
    }
    return np;
}

ZL_Report DI_hilbert3_i32(ZL_Decoder* dictx, const ZL_Input* ins[])
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dictx);
    const ZL_Input* const in = ins[0];
    ZL_ASSERT_EQ(ZL_Input_type(in), ZL_Type_numeric);
    ZL_ERR_IF_NE(ZL_Input_eltWidth(in), 8, corruption);
    size_t const n = ZL_Input_numElts(in);

    ZL_Output* const out = ZL_Decoder_create1OutStream(dictx, n, 12);
    ZL_ERR_IF_NULL(out, allocation);
    size_t const np = hilbert3DecodeI32(
            (uint8_t*)ZL_Output_ptr(out), (const uint64_t*)ZL_Input_ptr(in), n);
    ZL_ERR_IF_EQ(np, SIZE_MAX, corruption, "hilbert3_i32: truncated escape");
    ZL_ERR_IF_ERR(ZL_Output_commit(out, np));
    return ZL_returnSuccess();
}
