// Copyright (c) Meta Platforms, Inc. and affiliates.

// Implementation note :
// raw transform == minimal dependency
#include "decode_dispatchN_byTag_kernel.h"
#include "openzl/shared/mem.h"
#include "openzl/shared/numeric_operations.h" // NUMOP_*
#include "openzl/shared/portability.h"

#if ZL_HAS_NEON
#    include <arm_neon.h>
#endif

#include <assert.h>
#include <string.h> // memcpy

static size_t sumArrayST(const size_t array[], size_t arraySize)
{
    size_t total = 0;
    for (size_t n = 0; n < arraySize; n++) {
        total += array[n];
    }
    return total;
}

/* Implementation notes :
 * This is a first generic implementation.
 * - It requires all inputs to be valid (see assert() below)
 *   In which case, it's necessarily successful.
 * - A different design would be to validate some or all input conditions,
 *   returning an error if input validation fails.
 *   This topic will probably come back, especially during fuzzer tests.
 * - The content of @srcs[] input array is modified by this function.
 *   This is documented, but this could be surprising for users.
 *   An alternate strategy could be to copy this array and modify the copy.
 *   But this requires to allocate space for this array.
 *   This could be done on stack for small array, but wouldn't be generic.
 *   We try to avoid any kind of dynamic memory allocation in raw transforms.
 *   It seems preferable to require this copy operation on the user side,
 *   (aka the Decoder Interface) where dynamic Allocation is available.
 * - These memcpy() invocation use variable sizes.
 *   If input consists of a few large segments, that's fine, no big deal.
 *   But if it consists of a lot of small data,
 *   for example in the 1-8 bytes ranges,
 *   then the overhead will be significant.
 *   More optimized copy strategies could be implemented if need be.
 */
size_t ZL_dispatchN_byTag_decode(
        void* restrict dst,
        size_t dstCapacity,
        const void* restrict srcs[],
        size_t nbSrcs,
        const size_t* restrict segmentSizes,
        const uint16_t* restrict bufIndex,
        size_t nbSegments)
{
    /* input validation */
    size_t const dstSize = sumArrayST(segmentSizes, nbSegments);
    assert(dstSize <= dstCapacity);
    if (dstCapacity) {
        assert(dst != NULL);
    }
    assert(srcs != NULL);
    for (size_t n = 0; n < nbSrcs; n++) {
        assert(srcs[n] != NULL);
    }
    assert(nbSrcs < 1 << 16);
    assert(NUMOP_underLimitU16(bufIndex, nbSegments, (uint16_t)nbSrcs));

    for (size_t n = 0; n < nbSegments; n++) {
        size_t const segSize = segmentSizes[n];
        uint16_t const tag   = bufIndex[n];
        switch (segSize) {
            case 1:
                ZL_write8(dst, ZL_read8(srcs[tag]));
                break;
            case 2:
                ZL_write16(dst, ZL_read16(srcs[tag]));
                break;
            case 4:
                ZL_write32(dst, ZL_read32(srcs[tag]));
                break;
            case 8:
                ZS_write64(dst, ZL_read64(srcs[tag]));
                break;
            default:
                memcpy(dst, srcs[tag], segSize);
        }
        dst       = (char*)dst + segSize;
        srcs[tag] = (const char*)srcs[tag] + segSize;
    }
    return dstSize;
}

/* ===== Round-robin specialization =====
 * Segment n has a fixed size and comes from srcs[n % nbSrcs] :
 * this is a plain interleave of nbSrcs arrays of fixed-width elements
 * (struct-of-arrays to array-of-structs).
 */

ZL_FORCE_INLINE void roundRobinScalar(
        uint8_t* restrict dst,
        const void* const restrict srcs[],
        size_t nbSrcs,
        size_t segSize,
        size_t firstRound,
        size_t nbRounds)
{
    for (size_t r = firstRound; r < nbRounds; r++) {
        for (size_t s = 0; s < nbSrcs; s++) {
            memcpy(dst + (r * nbSrcs + s) * segSize,
                   (const uint8_t*)srcs[s] + r * segSize,
                   segSize);
        }
    }
}

#if ZL_HAS_AVX2 || ZL_HAS_WASM_SIMD128

/* Byte shuffle masks interleaving 3 streams of @segSize-byte elements,
 * 16 input bytes per stream to 48 output bytes.
 * masks[s][v] selects the bytes of stream s landing in output vector v.
 * Requires segSize in {1, 2, 4, 8}. */
static void roundRobin3_buildMasks(uint8_t masks[3][3][16], size_t segSize)
{
    for (size_t j = 0; j < 48; j++) {
        size_t const elt    = j / (3 * segSize);
        size_t const stream = (j / segSize) % 3;
        size_t const byte   = j % segSize;
        for (size_t s = 0; s < 3; s++) {
            masks[s][j / 16][j % 16] = (s == stream)
                    ? (uint8_t)(elt * segSize + byte)
                    : (uint8_t)0x80;
        }
    }
}

#endif

#if ZL_HAS_AVX2

/* @return nb of rounds processed */
static size_t roundRobin3_simd(
        uint8_t* restrict dst,
        const uint8_t* restrict a,
        const uint8_t* restrict b,
        const uint8_t* restrict c,
        size_t segSize,
        size_t nbRounds)
{
    uint8_t masks[3][3][16];
    roundRobin3_buildMasks(masks, segSize);
#    define RR3_MASK(s, v)           \
        _mm256_broadcastsi128_si256( \
                _mm_loadu_si128((const __m128i_u*)(const void*)masks[s][v]))
    __m256i const ma0 = RR3_MASK(0, 0), ma1 = RR3_MASK(0, 1),
                  ma2 = RR3_MASK(0, 2);
    __m256i const mb0 = RR3_MASK(1, 0), mb1 = RR3_MASK(1, 1),
                  mb2 = RR3_MASK(1, 2);
    __m256i const mc0 = RR3_MASK(2, 0), mc1 = RR3_MASK(2, 1),
                  mc2 = RR3_MASK(2, 2);
#    undef RR3_MASK
    size_t const nbBytes = nbRounds * segSize;
    size_t pos           = 0;
    for (; pos + 32 <= nbBytes; pos += 32) {
        __m256i const va =
                _mm256_loadu_si256((const __m256i_u*)(const void*)(a + pos));
        __m256i const vb =
                _mm256_loadu_si256((const __m256i_u*)(const void*)(b + pos));
        __m256i const vc =
                _mm256_loadu_si256((const __m256i_u*)(const void*)(c + pos));
        /* each 128-bit lane o0|o1|o2 holds 48 consecutive output bytes */
        __m256i const o0 = _mm256_or_si256(
                _mm256_or_si256(
                        _mm256_shuffle_epi8(va, ma0),
                        _mm256_shuffle_epi8(vb, mb0)),
                _mm256_shuffle_epi8(vc, mc0));
        __m256i const o1 = _mm256_or_si256(
                _mm256_or_si256(
                        _mm256_shuffle_epi8(va, ma1),
                        _mm256_shuffle_epi8(vb, mb1)),
                _mm256_shuffle_epi8(vc, mc1));
        __m256i const o2 = _mm256_or_si256(
                _mm256_or_si256(
                        _mm256_shuffle_epi8(va, ma2),
                        _mm256_shuffle_epi8(vb, mb2)),
                _mm256_shuffle_epi8(vc, mc2));
        uint8_t* const out = dst + 3 * pos;
        _mm256_storeu_si256(
                (__m256i_u*)(void*)out,
                _mm256_permute2x128_si256(o0, o1, 0x20));
        _mm256_storeu_si256(
                (__m256i_u*)(void*)(out + 32),
                _mm256_permute2x128_si256(o2, o0, 0x30));
        _mm256_storeu_si256(
                (__m256i_u*)(void*)(out + 64),
                _mm256_permute2x128_si256(o1, o2, 0x31));
    }
    return pos / segSize;
}

#elif ZL_HAS_WASM_SIMD128

static size_t roundRobin3_simd(
        uint8_t* restrict dst,
        const uint8_t* restrict a,
        const uint8_t* restrict b,
        const uint8_t* restrict c,
        size_t segSize,
        size_t nbRounds)
{
    uint8_t masks[3][3][16];
    roundRobin3_buildMasks(masks, segSize);
    v128_t const ma0     = wasm_v128_load(masks[0][0]);
    v128_t const ma1     = wasm_v128_load(masks[0][1]);
    v128_t const ma2     = wasm_v128_load(masks[0][2]);
    v128_t const mb0     = wasm_v128_load(masks[1][0]);
    v128_t const mb1     = wasm_v128_load(masks[1][1]);
    v128_t const mb2     = wasm_v128_load(masks[1][2]);
    v128_t const mc0     = wasm_v128_load(masks[2][0]);
    v128_t const mc1     = wasm_v128_load(masks[2][1]);
    v128_t const mc2     = wasm_v128_load(masks[2][2]);
    size_t const nbBytes = nbRounds * segSize;
    size_t pos           = 0;
    for (; pos + 16 <= nbBytes; pos += 16) {
        v128_t const va    = wasm_v128_load(a + pos);
        v128_t const vb    = wasm_v128_load(b + pos);
        v128_t const vc    = wasm_v128_load(c + pos);
        uint8_t* const out = dst + 3 * pos;
        /* swizzle zeroes lanes whose index is >= 16 (0x80 here) */
        wasm_v128_store(
                out,
                wasm_v128_or(
                        wasm_v128_or(
                                wasm_i8x16_swizzle(va, ma0),
                                wasm_i8x16_swizzle(vb, mb0)),
                        wasm_i8x16_swizzle(vc, mc0)));
        wasm_v128_store(
                out + 16,
                wasm_v128_or(
                        wasm_v128_or(
                                wasm_i8x16_swizzle(va, ma1),
                                wasm_i8x16_swizzle(vb, mb1)),
                        wasm_i8x16_swizzle(vc, mc1)));
        wasm_v128_store(
                out + 32,
                wasm_v128_or(
                        wasm_v128_or(
                                wasm_i8x16_swizzle(va, ma2),
                                wasm_i8x16_swizzle(vb, mb2)),
                        wasm_i8x16_swizzle(vc, mc2)));
    }
    return pos / segSize;
}

#elif ZL_HAS_NEON

#    define GEN_RR3_NEON(W, T, VT, LD, ST)                           \
        static size_t roundRobin3_neon##W(                           \
                uint8_t* restrict dst,                               \
                const uint8_t* restrict a,                           \
                const uint8_t* restrict b,                           \
                const uint8_t* restrict c,                           \
                size_t nbRounds)                                     \
        {                                                            \
            size_t const step = 16 / (W);                            \
            size_t r          = 0;                                   \
            for (; r + step <= nbRounds; r += step) {                \
                VT v;                                                \
                v.val[0] = LD((const T*)(const void*)(a + r * (W))); \
                v.val[1] = LD((const T*)(const void*)(b + r * (W))); \
                v.val[2] = LD((const T*)(const void*)(c + r * (W))); \
                ST((T*)(void*)(dst + 3 * r * (W)), v);               \
            }                                                        \
            return r;                                                \
        }
GEN_RR3_NEON(1, uint8_t, uint8x16x3_t, vld1q_u8, vst3q_u8)
GEN_RR3_NEON(2, uint16_t, uint16x8x3_t, vld1q_u16, vst3q_u16)
GEN_RR3_NEON(4, uint32_t, uint32x4x3_t, vld1q_u32, vst3q_u32)
GEN_RR3_NEON(8, uint64_t, uint64x2x3_t, vld1q_u64, vst3q_u64)
#    undef GEN_RR3_NEON

static size_t roundRobin3_simd(
        uint8_t* restrict dst,
        const uint8_t* restrict a,
        const uint8_t* restrict b,
        const uint8_t* restrict c,
        size_t segSize,
        size_t nbRounds)
{
    switch (segSize) {
        case 1:
            return roundRobin3_neon1(dst, a, b, c, nbRounds);
        case 2:
            return roundRobin3_neon2(dst, a, b, c, nbRounds);
        case 4:
            return roundRobin3_neon4(dst, a, b, c, nbRounds);
        default:
            return roundRobin3_neon8(dst, a, b, c, nbRounds);
    }
}

#else

static size_t roundRobin3_simd(
        uint8_t* restrict dst,
        const uint8_t* restrict a,
        const uint8_t* restrict b,
        const uint8_t* restrict c,
        size_t segSize,
        size_t nbRounds)
{
    (void)dst, (void)a, (void)b, (void)c, (void)segSize, (void)nbRounds;
    return 0;
}

#endif

#define GEN_ROUND_ROBIN(nbSrcs, segSize)                                    \
    static void roundRobin_##nbSrcs##_##segSize(                            \
            uint8_t* restrict dst,                                          \
            const void* const restrict srcs[],                              \
            size_t firstRound,                                              \
            size_t nbRounds)                                                \
    {                                                                       \
        roundRobinScalar(dst, srcs, nbSrcs, segSize, firstRound, nbRounds); \
    }
GEN_ROUND_ROBIN(2, 1)
GEN_ROUND_ROBIN(2, 2)
GEN_ROUND_ROBIN(2, 4)
GEN_ROUND_ROBIN(2, 8)
GEN_ROUND_ROBIN(3, 1)
GEN_ROUND_ROBIN(3, 2)
GEN_ROUND_ROBIN(3, 4)
GEN_ROUND_ROBIN(3, 8)
GEN_ROUND_ROBIN(4, 1)
GEN_ROUND_ROBIN(4, 2)
GEN_ROUND_ROBIN(4, 4)
GEN_ROUND_ROBIN(4, 8)
#undef GEN_ROUND_ROBIN

void ZL_dispatchN_byTag_decodeRoundRobin(
        void* restrict dst,
        const void* const restrict srcs[],
        size_t nbSrcs,
        size_t segSize,
        size_t nbSegments)
{
    assert(nbSrcs > 0);
    assert(segSize > 0);
    uint8_t* const out    = (uint8_t*)dst;
    size_t const nbRounds = nbSegments / nbSrcs;
    size_t const nbTail   = nbSegments % nbSrcs;
    size_t done           = 0;

    if (nbSrcs == 1) {
        memcpy(out, srcs[0], nbSegments * segSize);
        return;
    }
    if (nbSrcs == 3
        && (segSize == 1 || segSize == 2 || segSize == 4 || segSize == 8)) {
        done = roundRobin3_simd(
                out,
                (const uint8_t*)srcs[0],
                (const uint8_t*)srcs[1],
                (const uint8_t*)srcs[2],
                segSize,
                nbRounds);
    }
#define ROUND_ROBIN_CASE(n, w)                           \
    if (nbSrcs == n && segSize == w) {                   \
        roundRobin_##n##_##w(out, srcs, done, nbRounds); \
    } else
    ROUND_ROBIN_CASE(2, 1)
    ROUND_ROBIN_CASE(2, 2)
    ROUND_ROBIN_CASE(2, 4)
    ROUND_ROBIN_CASE(2, 8)
    ROUND_ROBIN_CASE(3, 1)
    ROUND_ROBIN_CASE(3, 2)
    ROUND_ROBIN_CASE(3, 4)
    ROUND_ROBIN_CASE(3, 8)
    ROUND_ROBIN_CASE(4, 1)
    ROUND_ROBIN_CASE(4, 2)
    ROUND_ROBIN_CASE(4, 4)
    ROUND_ROBIN_CASE(4, 8)
    {
        roundRobinScalar(out, srcs, nbSrcs, segSize, done, nbRounds);
    }
#undef ROUND_ROBIN_CASE

    /* last, incomplete round */
    for (size_t s = 0; s < nbTail; s++) {
        memcpy(out + (nbRounds * nbSrcs + s) * segSize,
               (const uint8_t*)srcs[s] + nbRounds * segSize,
               segSize);
    }
}
