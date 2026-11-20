// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/pco_bin_split/decode_pco_bin_split_binding.h"

#include <stdint.h>
#include <string.h>

#include "openzl/codecs/zl_pco_bin_split.h"
#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
#include "openzl/shared/mem.h"
#include "openzl/shared/portability.h"
#include "openzl/shared/varint.h"

enum {
    PBS_MAX_BINS = 256,
    PBS_BATCH_N  = 256,
};

#define PBS_USE_COMB ZL_HAS_AVX2

static inline uint64_t pbs_read_bits(
        const uint8_t* src,
        size_t packedBytes,
        uint64_t bitPos,
        unsigned ob)
{
    if (ob == 0u) {
        return 0ULL;
    }
    size_t const bytePos = (size_t)(bitPos >> 3);
    unsigned const shift = (unsigned)(bitPos & 7u);
    uint64_t const mask  = (ob == 64u) ? UINT64_MAX : ((1ULL << ob) - 1ULL);

    uint64_t word = 0;
    if (ZL_LIKELY(bytePos + 8u <= packedBytes)) {
        word = ZL_readLE64(src + bytePos) >> shift;
        if (ZL_UNLIKELY(shift + ob > 64u)) {
            word |= ((uint64_t)src[bytePos + 8u]) << (64u - shift);
        }
    } else {
        uint8_t tmp[9];
        memset(tmp, 0, sizeof(tmp));
        size_t const avail = packedBytes - bytePos;
        memcpy(tmp, src + bytePos, avail < 9u ? avail : 9u);
        word = ZL_readLE64(tmp) >> shift;
        if (shift + ob > 64u) {
            word |= ((uint64_t)tmp[8]) << (64u - shift);
        }
    }
    return word & mask;
}

#define PBS_FROM_LATENT_UNSIGNED(l, eltBits) (l)
#define PBS_FROM_LATENT_SIGNED(l, eltBits) ((l) ^ (1ULL << ((eltBits) - 1u)))
#define PBS_FROM_LATENT_FLOAT(l, eltBits) \
    (((l) & (1ULL << ((eltBits) - 1u))) != 0 \
             ? ((l) ^ (1ULL << ((eltBits) - 1u))) \
             : (~(l)))

ZL_FORCE_INLINE int pbs_validate_bins(
        const uint8_t* binIds, size_t n, size_t numBins)
{
    if (numBins >= PBS_MAX_BINS) {
        return 1;
    }
    size_t i = 0;
#if ZL_HAS_AVX2
    if (n >= 32u) {
        __m256i m0 = _mm256_setzero_si256();
        __m256i m1 = _mm256_setzero_si256();
        for (; i + 64u <= n; i += 64u) {
            m0 = _mm256_max_epu8(m0, _mm256_loadu_si256((const __m256i_u*)(const void*)(binIds + i)));
            m1 = _mm256_max_epu8(m1, _mm256_loadu_si256((const __m256i_u*)(const void*)(binIds + i + 32u)));
        }
        for (; i + 32u <= n; i += 32u) {
            m0 = _mm256_max_epu8(m0, _mm256_loadu_si256((const __m256i_u*)(const void*)(binIds + i)));
        }
        m0 = _mm256_max_epu8(m0, m1);
        __m256i const over = _mm256_subs_epu8(m0, _mm256_set1_epi8((char)(uint8_t)(numBins - 1u)));
        if (ZL_UNLIKELY(!_mm256_testz_si256(over, over))) {
            return 0;
        }
    }
#elif ZL_HAS_WASM_SIMD128
    if (n >= 16u) {
        v128_t m0 = wasm_i8x16_splat(0);
        v128_t m1 = wasm_i8x16_splat(0);
        for (; i + 32u <= n; i += 32u) {
            m0 = wasm_u8x16_max(m0, wasm_v128_load(binIds + i));
            m1 = wasm_u8x16_max(m1, wasm_v128_load(binIds + i + 16u));
        }
        for (; i + 16u <= n; i += 16u) {
            m0 = wasm_u8x16_max(m0, wasm_v128_load(binIds + i));
        }
        m0 = wasm_u8x16_max(m0, m1);
        if (ZL_UNLIKELY(wasm_v128_any_true(wasm_u8x16_gt(m0, wasm_u8x16_splat((uint8_t)(numBins - 1u)))))) {
            return 0;
        }
    }
#endif
    for (; i < n; i++) {
        if (ZL_UNLIKELY((size_t)binIds[i] >= numBins)) {
            return 0;
        }
    }
    return 1;
}

#if ZL_HAS_AVX2
ZL_FORCE_INLINE __m256i pbs_avx2_scan16_u16(__m128i ob128, __m256i* accVec)
{
    __m256i const v    = _mm256_cvtepu8_epi16(ob128);
    __m256i s          = _mm256_add_epi16(v, _mm256_slli_si256(v, 2));
    s                  = _mm256_add_epi16(s, _mm256_slli_si256(s, 4));
    s                  = _mm256_add_epi16(s, _mm256_slli_si256(s, 8));
    __m256i const lasts    = _mm256_shuffle_epi8(s, _mm256_set1_epi16((short)0x0F0E));
    __m256i const carryHi  = _mm256_permute2x128_si256(lasts, lasts, 0x08);
    __m256i const fullIncl = _mm256_add_epi16(s, carryHi);
    __m256i const excl     = _mm256_add_epi16(_mm256_sub_epi16(fullIncl, v), *accVec);
    __m256i const total16  = _mm256_add_epi16(
            _mm256_permute2x128_si256(lasts, lasts, 0x00),
            _mm256_permute2x128_si256(lasts, lasts, 0x11));
    *accVec = _mm256_add_epi16(*accVec, total16);
    return excl;
}
#elif ZL_HAS_WASM_SIMD128
ZL_FORCE_INLINE v128_t pbs_wasm_scan8_u16(v128_t v, v128_t* accVec)
{
    v128_t const z = wasm_i8x16_splat(0);
    v128_t s       = wasm_i16x8_add(v, wasm_i8x16_shuffle(z, v, 0,1, 16,17, 18,19, 20,21, 22,23, 24,25, 26,27, 28,29));
    s              = wasm_i16x8_add(s, wasm_i8x16_shuffle(z, s, 0,1, 2,3, 16,17, 18,19, 20,21, 22,23, 24,25, 26,27));
    s              = wasm_i16x8_add(s, wasm_i8x16_shuffle(z, s, 0,1, 2,3, 4,5, 6,7, 16,17, 18,19, 20,21, 22,23));
    v128_t const excl   = wasm_i16x8_add(wasm_i16x8_sub(s, v), *accVec);
    v128_t const total8 = wasm_i8x16_shuffle(s, s, 14,15, 14,15, 14,15, 14,15, 14,15, 14,15, 14,15, 14,15);
    *accVec             = wasm_i16x8_add(*accVec, total8);
    return excl;
}
#endif

#if ZL_HAS_AVX2
/* Byte (shift / 8) of 32 consecutive uint32 as 32 bytes, in order. */
ZL_FORCE_INLINE __m256i pbs_avx2_comb_bytes_x32(const uint32_t* c, int shift)
{
    __m256i const m   = _mm256_set1_epi32(0xFF);
    __m128i const cnt = _mm_cvtsi32_si128(shift);
    __m256i const a0  = _mm256_and_si256(_mm256_srl_epi32(_mm256_loadu_si256((const __m256i_u*)(const void*)(c + 0)), cnt), m);
    __m256i const a1  = _mm256_and_si256(_mm256_srl_epi32(_mm256_loadu_si256((const __m256i_u*)(const void*)(c + 8)), cnt), m);
    __m256i const a2  = _mm256_and_si256(_mm256_srl_epi32(_mm256_loadu_si256((const __m256i_u*)(const void*)(c + 16)), cnt), m);
    __m256i const a3  = _mm256_and_si256(_mm256_srl_epi32(_mm256_loadu_si256((const __m256i_u*)(const void*)(c + 24)), cnt), m);
    return _mm256_permutevar8x32_epi32(
            _mm256_packus_epi16(_mm256_packus_epi32(a0, a1), _mm256_packus_epi32(a2, a3)),
            _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7));
}

/* Low 16 bits of 16 consecutive uint32 as 16 uint16, in order. */
ZL_FORCE_INLINE __m256i pbs_avx2_comb_lo16_x16(const uint32_t* c)
{
    __m256i const m  = _mm256_set1_epi32(0xFFFF);
    __m256i const a0 = _mm256_and_si256(_mm256_loadu_si256((const __m256i_u*)(const void*)(c + 0)), m);
    __m256i const a1 = _mm256_and_si256(_mm256_loadu_si256((const __m256i_u*)(const void*)(c + 8)), m);
    return _mm256_permute4x64_epi64(_mm256_packus_epi32(a0, a1), 0xD8);
}
#endif

ZL_FORCE_INLINE int pbs_pass1_batch(
        const uint8_t* binIds,
        size_t batchN,
        size_t numBins,
        const uint8_t offsetBits[PBS_MAX_BINS],
        uint16_t baseBitRem,
        uint16_t bBitPos[PBS_BATCH_N],
        uint8_t bOb[PBS_BATCH_N],
        uint32_t* outEndBitPos,
        const uint32_t* comb,
        uint32_t bComb[PBS_BATCH_N])
{
    size_t k     = 0;
    uint32_t acc = (uint32_t)baseBitRem;
    /* More than 64 bins and elements of <= 16 bits: one scalar lookup per
     * element fetches (lower | offsetBits << 16); SIMD splits the two. */
    if (comb != NULL) {
        size_t j = 0;
        for (; j + 8u <= batchN; j += 8u) {
            /* One 8-byte load, then shifts: halves the load count. */
            uint64_t const w = ZL_readLE64(binIds + j);
            bComb[j + 0u] = comb[(uint8_t)w];         bComb[j + 1u] = comb[(uint8_t)(w >> 8)];
            bComb[j + 2u] = comb[(uint8_t)(w >> 16)]; bComb[j + 3u] = comb[(uint8_t)(w >> 24)];
            bComb[j + 4u] = comb[(uint8_t)(w >> 32)]; bComb[j + 5u] = comb[(uint8_t)(w >> 40)];
            bComb[j + 6u] = comb[(uint8_t)(w >> 48)]; bComb[j + 7u] = comb[(uint8_t)(w >> 56)];
        }
        for (; j < batchN; j++) {
            bComb[j] = comb[binIds[j]];
        }
    }
#if ZL_HAS_AVX2
    if (batchN >= 32u) {
        size_t const simdN = batchN & ~(size_t)31u;
        __m256i accVec     = _mm256_set1_epi16((short)baseBitRem);
        __m256i maxB       = _mm256_setzero_si256();
        if (numBins <= 16u) {
            __m256i const lut0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i_u*)(const void*)offsetBits));
            __m256i const m0F  = _mm256_set1_epi8(0x0F);
            for (; k < simdN; k += 32u) {
                __m256i const bVec  = _mm256_loadu_si256((const __m256i_u*)(const void*)(binIds + k));
                maxB                = _mm256_max_epu8(maxB, bVec);
                __m256i const obVec = _mm256_shuffle_epi8(lut0, _mm256_and_si256(bVec, m0F));
                _mm256_storeu_si256((__m256i_u*)(void*)(bOb + k), obVec);
                _mm256_storeu_si256((__m256i_u*)(void*)(bBitPos + k), pbs_avx2_scan16_u16(_mm256_castsi256_si128(obVec), &accVec));
                _mm256_storeu_si256((__m256i_u*)(void*)(bBitPos + k + 16u), pbs_avx2_scan16_u16(_mm256_extracti128_si256(obVec, 1), &accVec));
            }
        } else if (numBins <= 64u) {
            __m256i const lut0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i_u*)(const void*)(offsetBits + 0u)));
            __m256i const lut1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i_u*)(const void*)(offsetBits + 16u)));
            __m256i const lut2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i_u*)(const void*)(offsetBits + 32u)));
            __m256i const lut3 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i_u*)(const void*)(offsetBits + 48u)));
            __m256i const m0F  = _mm256_set1_epi8(0x0F);
            for (; k < simdN; k += 32u) {
                __m256i const bVec = _mm256_loadu_si256((const __m256i_u*)(const void*)(binIds + k));
                maxB               = _mm256_max_epu8(maxB, bVec);
                __m256i const bIdx = _mm256_and_si256(bVec, m0F);
                __m256i const sel0 = _mm256_slli_epi16(bVec, 3);
                __m256i const ob01 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut0, bIdx), _mm256_shuffle_epi8(lut1, bIdx), sel0);
                __m256i const ob23 = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut2, bIdx), _mm256_shuffle_epi8(lut3, bIdx), sel0);
                __m256i const obVec = _mm256_blendv_epi8(ob01, ob23, _mm256_slli_epi16(bVec, 2));
                _mm256_storeu_si256((__m256i_u*)(void*)(bOb + k), obVec);
                _mm256_storeu_si256((__m256i_u*)(void*)(bBitPos + k), pbs_avx2_scan16_u16(_mm256_castsi256_si128(obVec), &accVec));
                _mm256_storeu_si256((__m256i_u*)(void*)(bBitPos + k + 16u), pbs_avx2_scan16_u16(_mm256_extracti128_si256(obVec, 1), &accVec));
            }
        } else {
            if (numBins < PBS_MAX_BINS) {
                for (size_t j = 0; j < simdN; j += 32u) {
                    maxB = _mm256_max_epu8(maxB, _mm256_loadu_si256((const __m256i_u*)(const void*)(binIds + j)));
                }
            }
            if (comb != NULL) {
                for (size_t j = 0; j < simdN; j += 32u) {
                    _mm256_storeu_si256((__m256i_u*)(void*)(bOb + j), pbs_avx2_comb_bytes_x32(bComb + j, 16));
                }
            } else {
                for (size_t j = 0; j < simdN; j += 8u) {
                    const uint8_t* const bp = binIds + j;
                    bOb[j + 0u] = offsetBits[bp[0]]; bOb[j + 1u] = offsetBits[bp[1]];
                    bOb[j + 2u] = offsetBits[bp[2]]; bOb[j + 3u] = offsetBits[bp[3]];
                    bOb[j + 4u] = offsetBits[bp[4]]; bOb[j + 5u] = offsetBits[bp[5]];
                    bOb[j + 6u] = offsetBits[bp[6]]; bOb[j + 7u] = offsetBits[bp[7]];
                }
            }
            for (; k < simdN; k += 32u) {
                __m256i const obVec = _mm256_loadu_si256((const __m256i_u*)(const void*)(bOb + k));
                _mm256_storeu_si256((__m256i_u*)(void*)(bBitPos + k), pbs_avx2_scan16_u16(_mm256_castsi256_si128(obVec), &accVec));
                _mm256_storeu_si256((__m256i_u*)(void*)(bBitPos + k + 16u), pbs_avx2_scan16_u16(_mm256_extracti128_si256(obVec, 1), &accVec));
            }
        }
        if (numBins < PBS_MAX_BINS) {
            __m256i const over = _mm256_subs_epu8(maxB, _mm256_set1_epi8((char)(uint8_t)(numBins - 1u)));
            if (ZL_UNLIKELY(!_mm256_testz_si256(over, over))) {
                return 0;
            }
        }
        acc = (uint32_t)(uint16_t)_mm256_extract_epi16(accVec, 0);
    }
#elif ZL_HAS_WASM_SIMD128
    if (batchN >= 16u) {
        size_t const simdN = batchN & ~(size_t)15u;
        v128_t accVec      = wasm_u16x8_splat(baseBitRem);
        v128_t maxB        = wasm_i8x16_splat(0);
        if (numBins <= 64u) {
            v128_t const lut0 = wasm_v128_load(offsetBits + 0u);
            v128_t const lut1 = wasm_v128_load(offsetBits + 16u);
            v128_t const lut2 = wasm_v128_load(offsetBits + 32u);
            v128_t const lut3 = wasm_v128_load(offsetBits + 48u);
            for (; k < simdN; k += 16u) {
                v128_t const bVec = wasm_v128_load(binIds + k);
                maxB              = wasm_u8x16_max(maxB, bVec);
                v128_t obVec      = wasm_i8x16_swizzle(lut0, bVec);
                if (numBins > 16u) {
                    obVec = wasm_v128_or(obVec, wasm_i8x16_swizzle(lut1, wasm_i8x16_sub(bVec, wasm_i8x16_splat(16))));
                }
                if (numBins > 32u) {
                    obVec = wasm_v128_or(obVec, wasm_v128_or(
                            wasm_i8x16_swizzle(lut2, wasm_i8x16_sub(bVec, wasm_i8x16_splat(32))),
                            wasm_i8x16_swizzle(lut3, wasm_i8x16_sub(bVec, wasm_i8x16_splat(48)))));
                }
                wasm_v128_store(bOb + k, obVec);
                wasm_v128_store(bBitPos + k, pbs_wasm_scan8_u16(wasm_u16x8_extend_low_u8x16(obVec), &accVec));
                wasm_v128_store(bBitPos + k + 8u, pbs_wasm_scan8_u16(wasm_u16x8_extend_high_u8x16(obVec), &accVec));
            }
        } else {
            if (numBins < PBS_MAX_BINS) {
                for (size_t j = 0; j < simdN; j += 16u) {
                    maxB = wasm_u8x16_max(maxB, wasm_v128_load(binIds + j));
                }
            }
            for (size_t j = 0; j < simdN; j += 8u) {
                const uint8_t* const bp = binIds + j;
                bOb[j + 0u] = offsetBits[bp[0]]; bOb[j + 1u] = offsetBits[bp[1]];
                bOb[j + 2u] = offsetBits[bp[2]]; bOb[j + 3u] = offsetBits[bp[3]];
                bOb[j + 4u] = offsetBits[bp[4]]; bOb[j + 5u] = offsetBits[bp[5]];
                bOb[j + 6u] = offsetBits[bp[6]]; bOb[j + 7u] = offsetBits[bp[7]];
            }
            for (; k < simdN; k += 16u) {
                v128_t const obVec = wasm_v128_load(bOb + k);
                wasm_v128_store(bBitPos + k, pbs_wasm_scan8_u16(wasm_u16x8_extend_low_u8x16(obVec), &accVec));
                wasm_v128_store(bBitPos + k + 8u, pbs_wasm_scan8_u16(wasm_u16x8_extend_high_u8x16(obVec), &accVec));
            }
        }
        if (numBins < PBS_MAX_BINS && ZL_UNLIKELY(wasm_v128_any_true(wasm_u8x16_gt(maxB, wasm_u8x16_splat((uint8_t)(numBins - 1u)))))) {
            return 0;
        }
        acc = (uint32_t)wasm_u16x8_extract_lane(accVec, 0);
    }
#endif
    for (; k < batchN; k++) {
        uint8_t const b = binIds[k];
        if (numBins < PBS_MAX_BINS && ZL_UNLIKELY((size_t)b >= numBins)) {
            return 0;
        }
        uint8_t const ob = offsetBits[b];
        bOb[k]           = ob;
        bBitPos[k]       = (uint16_t)acc;
        acc += (uint32_t)ob;
    }
    *outEndBitPos = acc;
    return 1;
}

#if ZL_HAS_AVX2
ZL_FORCE_INLINE __m256i pbs_avx2_latent8(__m256i v, int mode)
{
    return (mode == ZL_PcoBinSplit_signed)
            ? _mm256_xor_si256(v, _mm256_set1_epi8((char)0x80u))
            : v;
}

ZL_FORCE_INLINE __m256i pbs_avx2_latent16(__m256i v, int mode)
{
    if (mode == ZL_PcoBinSplit_signed) {
        return _mm256_xor_si256(v, _mm256_set1_epi16((short)0x8000u));
    }
    if (mode == ZL_PcoBinSplit_float) {
        __m256i const t = _mm256_xor_si256(v, _mm256_set1_epi16((short)0x8000u));
        return _mm256_xor_si256(t, _mm256_srli_epi16(_mm256_srai_epi16(t, 15), 1));
    }
    return v;
}

ZL_FORCE_INLINE __m256i pbs_avx2_latent32(__m256i v, int mode)
{
    if (mode == ZL_PcoBinSplit_signed) {
        return _mm256_xor_si256(v, _mm256_set1_epi32((int)0x80000000u));
    }
    if (mode == ZL_PcoBinSplit_float) {
        __m256i const t = _mm256_xor_si256(v, _mm256_set1_epi32((int)0x80000000u));
        return _mm256_xor_si256(t, _mm256_srli_epi32(_mm256_srai_epi32(t, 31), 1));
    }
    return v;
}

ZL_FORCE_INLINE __m256i pbs_avx2_latent64(__m256i v, int mode)
{
    if (mode == ZL_PcoBinSplit_signed) {
        return _mm256_xor_si256(v, _mm256_set1_epi64x((long long)0x8000000000000000ULL));
    }
    if (mode == ZL_PcoBinSplit_float) {
        __m256i const t = _mm256_xor_si256(v, _mm256_set1_epi64x((long long)0x8000000000000000ULL));
        __m256i const m = _mm256_cmpgt_epi64(_mm256_setzero_si256(), t);
        return _mm256_xor_si256(t, _mm256_srli_epi64(m, 1));
    }
    return v;
}

/* Extract 8 offsets of <= 25 bits. Each 128-bit half holds 4 consecutive
 * offsets, which span at most 7 + 4 * 25 bits = 14 bytes, so one unaligned
 * 16-byte load per half plus a pshufb replaces 8 scalar loads. Reads 16 bytes
 * starting at the byte of bpPtr[0] and of bpPtr[4]. */
ZL_FORCE_INLINE __m256i pbs_avx2_off32_x8(
        const uint8_t* batchOffPtr, const uint16_t* bpPtr, const uint8_t* obPtr)
{
    __m256i const ones   = _mm256_set1_epi32(1);
    __m256i const bp32   = _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i_u*)(const void*)bpPtr));
    __m256i const sh32   = _mm256_and_si256(bp32, _mm256_set1_epi32(7));
    __m256i const byte32 = _mm256_srli_epi32(bp32, 3);
    __m256i const rel32  = _mm256_sub_epi32(byte32, _mm256_shuffle_epi32(byte32, 0x00));
    __m256i const ctrl   = _mm256_add_epi8(
            _mm256_shuffle_epi8(rel32, _mm256_setr_epi8(
                    0,0,0,0, 4,4,4,4, 8,8,8,8, 12,12,12,12,
                    0,0,0,0, 4,4,4,4, 8,8,8,8, 12,12,12,12)),
            _mm256_set1_epi32(0x03020100));
    __m256i const data   = _mm256_inserti128_si256(
            _mm256_castsi128_si256(_mm_loadu_si128((const __m128i_u*)(const void*)(batchOffPtr + (bpPtr[0] >> 3)))),
            _mm_loadu_si128((const __m128i_u*)(const void*)(batchOffPtr + (bpPtr[4] >> 3))), 1);
    __m256i const raw32  = _mm256_shuffle_epi8(data, ctrl);
    __m256i const ob32   = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i_u*)(const void*)obPtr));
    __m256i const mask32 = _mm256_sub_epi32(_mm256_sllv_epi32(ones, ob32), ones);
    return _mm256_and_si256(_mm256_srlv_epi32(raw32, sh32), mask32);
}

/* Extract 16 offsets of <= 8 bits into 16-bit lanes. Each 128-bit half holds
 * 8 consecutive offsets, which span at most 7 + 8 * 8 bits = 9 bytes. Reads 16
 * bytes starting at the byte of bpPtr[0] and of bpPtr[8]. */
ZL_FORCE_INLINE __m256i pbs_avx2_off8_x16(
        const uint8_t* batchOffPtr, const uint16_t* bpPtr, const uint8_t* obPtr)
{
    __m256i const bp16   = _mm256_loadu_si256((const __m256i_u*)(const void*)bpPtr);
    __m256i const byte16 = _mm256_srli_epi16(bp16, 3);
    __m256i const rel16  = _mm256_sub_epi16(
            byte16, _mm256_shuffle_epi8(byte16, _mm256_set1_epi16(0x0100)));
    __m256i const ctrl   = _mm256_add_epi8(
            _mm256_shuffle_epi8(rel16, _mm256_setr_epi8(
                    0,0, 2,2, 4,4, 6,6, 8,8, 10,10, 12,12, 14,14,
                    0,0, 2,2, 4,4, 6,6, 8,8, 10,10, 12,12, 14,14)),
            _mm256_set1_epi16(0x0100));
    __m256i const data   = _mm256_inserti128_si256(
            _mm256_castsi128_si256(_mm_loadu_si128((const __m128i_u*)(const void*)(batchOffPtr + (bpPtr[0] >> 3)))),
            _mm_loadu_si128((const __m128i_u*)(const void*)(batchOffPtr + (bpPtr[8] >> 3))), 1);
    __m256i const raw16  = _mm256_shuffle_epi8(data, ctrl);
    /* (raw << (7 - sh)) >> 7 == (raw >> sh) & 0x1FF */
    __m256i const mulLut = _mm256_setr_epi8(
            (char)128, 64, 32, 16, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            (char)128, 64, 32, 16, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0);
    __m256i const maskLut = _mm256_setr_epi8(
            0, 1, 3, 7, 15, 31, 63, 127, (char)255, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 3, 7, 15, 31, 63, 127, (char)255, 0, 0, 0, 0, 0, 0, 0);
    __m256i const shIdx  = _mm256_or_si256(
            _mm256_and_si256(bp16, _mm256_set1_epi16(7)), _mm256_set1_epi16((short)0x8000u));
    __m256i const mul16  = _mm256_shuffle_epi8(mulLut, shIdx);
    __m256i const ob16   = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i_u*)(const void*)obPtr));
    __m256i const mask16 = _mm256_shuffle_epi8(maskLut, ob16);
    return _mm256_and_si256(_mm256_srli_epi16(_mm256_mullo_epi16(raw16, mul16), 7), mask16);
}

/* Extract 16 offsets of <= 15 bits into 16-bit lanes. Each 128-bit half holds
 * 8 consecutive offsets, which span at most 7 + 8 * 15 bits = 16 bytes. An
 * offset starting at bit sh of byte b is assembled from byte b (its low
 * 8 - sh bits) and from the 16-bit word at byte b + 1 (the rest). Bytes the
 * offset does not reach only land above bit offsetBits and are masked off.
 * Reads 16 bytes starting at the byte of bpPtr[0] and of bpPtr[8]. */
ZL_FORCE_INLINE __m256i pbs_avx2_off15_x16(
        const uint8_t* batchOffPtr, const uint16_t* bpPtr, const uint8_t* obPtr)
{
    __m256i const bp16   = _mm256_loadu_si256((const __m256i_u*)(const void*)bpPtr);
    __m256i const byte16 = _mm256_srli_epi16(bp16, 3);
    __m256i const rel16  = _mm256_sub_epi16(
            byte16, _mm256_shuffle_epi8(byte16, _mm256_set1_epi16(0x0100)));
    __m256i const zeroHi = _mm256_set1_epi16((short)0x8000u);
    __m256i const ctrlA  = _mm256_or_si256(rel16, zeroHi);
    __m256i const ctrlB  = _mm256_add_epi8(
            _mm256_shuffle_epi8(rel16, _mm256_setr_epi8(
                    0,0, 2,2, 4,4, 6,6, 8,8, 10,10, 12,12, 14,14,
                    0,0, 2,2, 4,4, 6,6, 8,8, 10,10, 12,12, 14,14)),
            _mm256_set1_epi16(0x0201));
    __m256i const data   = _mm256_inserti128_si256(
            _mm256_castsi128_si256(_mm_loadu_si128((const __m128i_u*)(const void*)(batchOffPtr + (bpPtr[0] >> 3)))),
            _mm_loadu_si128((const __m128i_u*)(const void*)(batchOffPtr + (bpPtr[8] >> 3))), 1);
    __m256i const mulLut = _mm256_setr_epi8(
            (char)128, 64, 32, 16, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            (char)128, 64, 32, 16, 8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0);
    __m256i const maskLoLut = _mm256_setr_epi8(
            0, 1, 3, 7, 15, 31, 63, 127, -1, -1, -1, -1, -1, -1, -1, -1,
            0, 1, 3, 7, 15, 31, 63, 127, -1, -1, -1, -1, -1, -1, -1, -1);
    __m256i const maskHiLut = _mm256_setr_epi8(
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 7, 15, 31, 63, 127,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 3, 7, 15, 31, 63, 127);
    /* mul16 = 1 << (7 - sh) */
    __m256i const mul16  = _mm256_shuffle_epi8(
            mulLut, _mm256_or_si256(_mm256_and_si256(bp16, _mm256_set1_epi16(7)), zeroHi));
    /* byte b >> sh */
    __m256i const partA  = _mm256_srli_epi16(
            _mm256_mullo_epi16(_mm256_shuffle_epi8(data, ctrlA), mul16), 7);
    /* (word at byte b + 1) << (8 - sh) */
    __m256i const partB  = _mm256_slli_epi16(
            _mm256_mullo_epi16(_mm256_shuffle_epi8(data, ctrlB), mul16), 1);
    __m256i const ob16   = _mm256_cvtepu8_epi16(_mm_loadu_si128((const __m128i_u*)(const void*)obPtr));
    __m256i const mask16 = _mm256_or_si256(
            _mm256_shuffle_epi8(maskLoLut, ob16),
            _mm256_shuffle_epi8(maskHiLut, _mm256_slli_epi16(ob16, 8)));
    return _mm256_and_si256(_mm256_or_si256(partA, partB), mask16);
}

/* 16 offsets of 16-bit elements into 16-bit lanes. */
ZL_FORCE_INLINE __m256i pbs_avx2_off16_x16(
        const uint8_t* batchOffPtr, const uint16_t* bpPtr, const uint8_t* obPtr, unsigned maxOb)
{
    if (maxOb <= 15u) {
        return pbs_avx2_off15_x16(batchOffPtr, bpPtr, obPtr);
    }
    __m256i const o0 = pbs_avx2_off32_x8(batchOffPtr, bpPtr + 0u, obPtr + 0u);
    __m256i const o1 = pbs_avx2_off32_x8(batchOffPtr, bpPtr + 8u, obPtr + 8u);
    return _mm256_permute4x64_epi64(_mm256_packus_epi32(o0, o1), 0xD8);
}

ZL_FORCE_INLINE __m256i pbs_avx2_off64_x4(
        const uint8_t* batchOffPtr, const uint16_t* bpPtr, const uint8_t* obPtr)
{
    __m256i const ones64 = _mm256_set1_epi64x(1LL);
    __m256i const bp64   = _mm256_cvtepu16_epi64(_mm_loadl_epi64((const __m128i_u*)(const void*)bpPtr));
    __m256i const sh64   = _mm256_and_si256(bp64, _mm256_set1_epi64x(7LL));
    __m256i const raw64 = _mm256_setr_epi64x(
            (long long)ZL_readLE64(batchOffPtr + (bpPtr[0] >> 3)),
            (long long)ZL_readLE64(batchOffPtr + (bpPtr[1] >> 3)),
            (long long)ZL_readLE64(batchOffPtr + (bpPtr[2] >> 3)),
            (long long)ZL_readLE64(batchOffPtr + (bpPtr[3] >> 3)));
    __m256i const ob64   = _mm256_cvtepu8_epi64(_mm_cvtsi32_si128((int)ZL_readLE32(obPtr)));
    __m256i const mask64 = _mm256_sub_epi64(_mm256_sllv_epi64(ones64, ob64), ones64);
    return _mm256_and_si256(_mm256_srlv_epi64(raw64, sh64), mask64);
}

ZL_FORCE_INLINE __m256i pbs_avx2_off64_x4_wide(
        const uint8_t* batchOffPtr, const uint16_t* bpPtr, const uint8_t* obPtr)
{
    __m256i const ones64    = _mm256_set1_epi64x(1LL);
    __m256i const bp64      = _mm256_cvtepu16_epi64(_mm_loadl_epi64((const __m128i_u*)(const void*)bpPtr));
    __m256i const sh64      = _mm256_and_si256(bp64, _mm256_set1_epi64x(7LL));
    __m256i const rsh64     = _mm256_sub_epi64(_mm256_set1_epi64x(63LL), sh64);
    const uint8_t* const p0 = batchOffPtr + (bpPtr[0] >> 3);
    const uint8_t* const p1 = batchOffPtr + (bpPtr[1] >> 3);
    const uint8_t* const p2 = batchOffPtr + (bpPtr[2] >> 3);
    const uint8_t* const p3 = batchOffPtr + (bpPtr[3] >> 3);
    __m256i const raw64     = _mm256_setr_epi64x(
            (long long)ZL_readLE64(p0), (long long)ZL_readLE64(p1),
            (long long)ZL_readLE64(p2), (long long)ZL_readLE64(p3));
    __m256i const hi64      = _mm256_setr_epi64x(
            (long long)((uint64_t)p0[8] << 1), (long long)((uint64_t)p1[8] << 1),
            (long long)((uint64_t)p2[8] << 1), (long long)((uint64_t)p3[8] << 1));
    __m256i const word64    = _mm256_or_si256(
            _mm256_srlv_epi64(raw64, sh64), _mm256_sllv_epi64(hi64, rsh64));
    __m256i const ob64      = _mm256_cvtepu8_epi64(_mm_cvtsi32_si128((int)ZL_readLE32(obPtr)));
    __m256i const mask64    = _mm256_sub_epi64(_mm256_sllv_epi64(ones64, ob64), ones64);
    return _mm256_and_si256(word64, mask64);
}
#endif

#if ZL_HAS_WASM_SIMD128
ZL_FORCE_INLINE v128_t pbs_wasm_latent8(v128_t v, int mode)
{
    return (mode == ZL_PcoBinSplit_signed) ? wasm_v128_xor(v, wasm_u8x16_splat(0x80u)) : v;
}

ZL_FORCE_INLINE v128_t pbs_wasm_latent16(v128_t v, int mode)
{
    if (mode == ZL_PcoBinSplit_signed) {
        return wasm_v128_xor(v, wasm_u16x8_splat(0x8000u));
    }
    if (mode == ZL_PcoBinSplit_float) {
        v128_t const t = wasm_v128_xor(v, wasm_u16x8_splat(0x8000u));
        return wasm_v128_xor(t, wasm_u16x8_shr(wasm_i16x8_shr(t, 15), 1));
    }
    return v;
}

ZL_FORCE_INLINE v128_t pbs_wasm_latent32(v128_t v, int mode)
{
    if (mode == ZL_PcoBinSplit_signed) {
        return wasm_v128_xor(v, wasm_u32x4_splat(0x80000000u));
    }
    if (mode == ZL_PcoBinSplit_float) {
        v128_t const t = wasm_v128_xor(v, wasm_u32x4_splat(0x80000000u));
        return wasm_v128_xor(t, wasm_u32x4_shr(wasm_i32x4_shr(t, 31), 1));
    }
    return v;
}

ZL_FORCE_INLINE v128_t pbs_wasm_latent64(v128_t v, int mode)
{
    if (mode == ZL_PcoBinSplit_signed) {
        return wasm_v128_xor(v, wasm_u64x2_splat(0x8000000000000000ULL));
    }
    if (mode == ZL_PcoBinSplit_float) {
        v128_t const t = wasm_v128_xor(v, wasm_u64x2_splat(0x8000000000000000ULL));
        return wasm_v128_xor(t, wasm_u64x2_shr(wasm_i64x2_shr(t, 63), 1));
    }
    return v;
}

ZL_FORCE_INLINE v128_t pbs_wasm_off16_x8(
        const uint8_t* batchOffPtr, const uint16_t* bpPtr, const uint8_t* obPtr)
{
    v128_t const mulLut  = wasm_u8x16_make(128,64,32,16,8,4,2,1,0,0,0,0,0,0,0,0);
    v128_t const maskLut = wasm_u8x16_make(0,1,3,7,15,31,63,127,255,255,255,255,255,255,255,255);
    v128_t const bp16    = wasm_v128_load(bpPtr);
    v128_t const mul16   = wasm_i8x16_swizzle(
            mulLut, wasm_v128_or(wasm_v128_and(bp16, wasm_u16x8_splat(7u)), wasm_u16x8_splat(0x8000u)));
    v128_t const m24  = wasm_u32x4_splat(0x00FFFFFFu);
    v128_t const raw0 = wasm_v128_and(wasm_u32x4_make(
            ZL_readLE32(batchOffPtr + (bpPtr[0] >> 3)), ZL_readLE32(batchOffPtr + (bpPtr[1] >> 3)),
            ZL_readLE32(batchOffPtr + (bpPtr[2] >> 3)), ZL_readLE32(batchOffPtr + (bpPtr[3] >> 3))), m24);
    v128_t const raw1 = wasm_v128_and(wasm_u32x4_make(
            ZL_readLE32(batchOffPtr + (bpPtr[4] >> 3)), ZL_readLE32(batchOffPtr + (bpPtr[5] >> 3)),
            ZL_readLE32(batchOffPtr + (bpPtr[6] >> 3)), ZL_readLE32(batchOffPtr + (bpPtr[7] >> 3))), m24);
    v128_t const sh0 = wasm_u32x4_shr(wasm_i32x4_mul(raw0, wasm_u32x4_extend_low_u16x8(mul16)), 7u);
    v128_t const sh1 = wasm_u32x4_shr(wasm_i32x4_mul(raw1, wasm_u32x4_extend_high_u16x8(mul16)), 7u);
    v128_t const pk  = wasm_i8x16_shuffle(sh0, sh1, 0,1,4,5,8,9,12,13,16,17,20,21,24,25,28,29);
    v128_t const ob8 = wasm_v128_load64_zero(obPtr);
    v128_t const mLo = wasm_i8x16_swizzle(maskLut, wasm_u8x16_min(ob8, wasm_u8x16_splat(15u)));
    v128_t const mHi = wasm_i8x16_swizzle(maskLut, wasm_u8x16_sub_sat(ob8, wasm_u8x16_splat(8u)));
    return wasm_v128_and(pk, wasm_i8x16_shuffle(mLo, mHi, 0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23));
}

ZL_FORCE_INLINE v128_t pbs_wasm_off32_x4(
        const uint8_t* batchOffPtr, const uint16_t* bpPtr, const uint8_t* obPtr)
{
    v128_t const mulLut  = wasm_u8x16_make(128,64,32,16,8,4,2,1,0,0,0,0,0,0,0,0);
    v128_t const maskLut = wasm_u8x16_make(0,1,3,7,15,31,63,127,255,255,255,255,255,255,255,255);
    v128_t const subVec  = wasm_u8x16_make(0,8,16,24, 0,8,16,24, 0,8,16,24, 0,8,16,24);
    v128_t const bp16    = wasm_v128_load64_zero(bpPtr);
    v128_t const mul32   = wasm_u32x4_extend_low_u16x8(wasm_i8x16_swizzle(
            mulLut, wasm_v128_or(wasm_v128_and(bp16, wasm_u16x8_splat(7u)), wasm_u16x8_splat(0x8000u))));
    v128_t const raw32 = wasm_u32x4_make(
            ZL_readLE32(batchOffPtr + (bpPtr[0] >> 3)), ZL_readLE32(batchOffPtr + (bpPtr[1] >> 3)),
            ZL_readLE32(batchOffPtr + (bpPtr[2] >> 3)), ZL_readLE32(batchOffPtr + (bpPtr[3] >> 3)));
    v128_t const lo64 = wasm_u64x2_shr(wasm_u64x2_extmul_low_u32x4(raw32, mul32), 7u);
    v128_t const hi64 = wasm_u64x2_shr(wasm_u64x2_extmul_high_u32x4(raw32, mul32), 7u);
    v128_t const sh32 = wasm_i8x16_shuffle(lo64, hi64, 0,1,2,3,8,9,10,11,16,17,18,19,24,25,26,27);
    v128_t const ob4  = wasm_u32x4_splat(ZL_readLE32(obPtr));
    v128_t const obr  = wasm_i8x16_shuffle(ob4, ob4, 0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3);
    v128_t const mask = wasm_i8x16_swizzle(maskLut, wasm_u8x16_min(wasm_u8x16_sub_sat(obr, subVec), wasm_u8x16_splat(15u)));
    return wasm_v128_and(sh32, mask);
}
#endif

ZL_FORCE_INLINE size_t pbs_simd_all_zero(
        void* dst,
        size_t eltWidth,
        const uint8_t* binIds,
        size_t n,
        size_t numBins,
        const void* decodedLowers)
{
    size_t i = 0;
#if ZL_HAS_AVX2
    if (eltWidth == 1u && numBins <= 16u && n >= 32u) {
        __m256i const lut = _mm256_broadcastsi128_si256(
                _mm_loadu_si128((const __m128i_u*)decodedLowers));
        __m256i const m0F = _mm256_set1_epi8(0x0F);
        uint8_t* const d8 = (uint8_t*)dst;
        for (; i + 32u <= n; i += 32u) {
            __m256i const b = _mm256_loadu_si256((const __m256i_u*)(const void*)(binIds + i));
            _mm256_storeu_si256((__m256i_u*)(void*)(d8 + i), _mm256_shuffle_epi8(lut, _mm256_and_si256(b, m0F)));
        }
    } else if (eltWidth == 2u && numBins <= 16u && n >= 16u) {
        const uint8_t* const lb = (const uint8_t*)decodedLowers;
        __m128i const raw0 = _mm_loadu_si128((const __m128i_u*)(const void*)(lb + 0u));
        __m128i const raw1 = _mm_loadu_si128((const __m128i_u*)(const void*)(lb + 16u));
        __m128i const shLo = _mm_setr_epi8(0,2,4,6,8,10,12,14,-1,-1,-1,-1,-1,-1,-1,-1);
        __m128i const shHi = _mm_setr_epi8(1,3,5,7,9,11,13,15,-1,-1,-1,-1,-1,-1,-1,-1);
        __m128i const lutLo = _mm_unpacklo_epi64(_mm_shuffle_epi8(raw0, shLo), _mm_shuffle_epi8(raw1, shLo));
        __m128i const lutHi = _mm_unpacklo_epi64(_mm_shuffle_epi8(raw0, shHi), _mm_shuffle_epi8(raw1, shHi));
        __m128i const m0F   = _mm_set1_epi8(0x0F);
        uint16_t* const d16 = (uint16_t*)dst;
        for (; i + 16u <= n; i += 16u) {
            __m128i const b   = _mm_and_si128(_mm_loadu_si128((const __m128i_u*)(const void*)(binIds + i)), m0F);
            __m256i const lo  = _mm256_cvtepu8_epi16(_mm_shuffle_epi8(lutLo, b));
            __m256i const hi  = _mm256_slli_epi16(_mm256_cvtepu8_epi16(_mm_shuffle_epi8(lutHi, b)), 8);
            _mm256_storeu_si256((__m256i_u*)(void*)(d16 + i), _mm256_or_si256(lo, hi));
        }
    } else if (eltWidth == 4u && numBins <= 8u && n >= 8u) {
        __m256i const lut = _mm256_loadu_si256((const __m256i_u*)decodedLowers);
        uint32_t* const d32 = (uint32_t*)dst;
        for (; i + 8u <= n; i += 8u) {
            __m256i const b32 = _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i_u*)(const void*)(binIds + i)));
            _mm256_storeu_si256((__m256i_u*)(void*)(d32 + i), _mm256_permutevar8x32_epi32(lut, b32));
        }
    }
#elif ZL_HAS_WASM_SIMD128
    if (eltWidth == 1u && numBins <= 16u && n >= 16u) {
        v128_t const lut  = wasm_v128_load(decodedLowers);
        uint8_t* const d8 = (uint8_t*)dst;
        for (; i + 16u <= n; i += 16u) {
            wasm_v128_store(d8 + i, wasm_i8x16_swizzle(lut, wasm_v128_load(binIds + i)));
        }
    } else if (eltWidth == 2u && numBins <= 16u && n >= 8u) {
        const uint8_t* const lb = (const uint8_t*)decodedLowers;
        v128_t const r0    = wasm_v128_load(lb + 0u);
        v128_t const r1    = wasm_v128_load(lb + 16u);
        v128_t const lutLo = wasm_i8x16_shuffle(r0, r1, 0,2,4,6,8,10,12,14, 16,18,20,22,24,26,28,30);
        v128_t const lutHi = wasm_i8x16_shuffle(r0, r1, 1,3,5,7,9,11,13,15, 17,19,21,23,25,27,29,31);
        uint16_t* const d16 = (uint16_t*)dst;
        for (; i + 8u <= n; i += 8u) {
            v128_t const b  = wasm_v128_load64_zero(binIds + i);
            v128_t const lo = wasm_i8x16_swizzle(lutLo, b);
            v128_t const hi = wasm_i8x16_swizzle(lutHi, b);
            wasm_v128_store(d16 + i, wasm_i8x16_shuffle(lo, hi, 0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23));
        }
    }
#else
    (void)dst; (void)eltWidth; (void)binIds; (void)n; (void)numBins; (void)decodedLowers;
#endif
    return i;
}

ZL_FORCE_INLINE size_t pbs_simd_pass2(
        void* outPtr,
        size_t eltWidth,
        int mode,
        const uint8_t* batchBins,
        const void* lowTyped,
        size_t numBins,
        unsigned maxOb,
        const uint8_t* batchOffPtr,
        size_t fastN32,
        size_t fastN64,
        const uint16_t bBitPos[PBS_BATCH_N],
        const uint8_t bOb[PBS_BATCH_N],
        const uint32_t* bComb)
{
    size_t k = 0;
#if ZL_HAS_AVX2
    if (eltWidth == 1u) {
        uint8_t* const d8       = (uint8_t*)outPtr;
        const uint8_t* const l8 = (const uint8_t*)lowTyped;
        if (numBins <= 16u) {
            __m256i const lut = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i_u*)l8));
            __m256i const m0F = _mm256_set1_epi8(0x0F);
            for (; k + 32u <= fastN32; k += 32u) {
                __m256i const o0   = pbs_avx2_off8_x16(batchOffPtr, bBitPos + k + 0u, bOb + k + 0u);
                __m256i const o1   = pbs_avx2_off8_x16(batchOffPtr, bBitPos + k + 16u, bOb + k + 16u);
                __m256i const off8 = _mm256_permute4x64_epi64(_mm256_packus_epi16(o0, o1), 0xD8);
                __m256i const bins = _mm256_loadu_si256((const __m256i_u*)(const void*)(batchBins + k));
                __m256i const lowVec = _mm256_shuffle_epi8(lut, _mm256_and_si256(bins, m0F));
                _mm256_storeu_si256((__m256i_u*)(void*)(d8 + k), pbs_avx2_latent8(_mm256_add_epi8(lowVec, off8), mode));
            }
        } else if (numBins <= 64u) {
            __m256i const lut0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i_u*)(l8 + 0u)));
            __m256i const lut1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i_u*)(l8 + 16u)));
            __m256i const lut2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i_u*)(l8 + 32u)));
            __m256i const lut3 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i_u*)(l8 + 48u)));
            __m256i const m0F  = _mm256_set1_epi8(0x0F);
            for (; k + 32u <= fastN32; k += 32u) {
                __m256i const o0   = pbs_avx2_off8_x16(batchOffPtr, bBitPos + k + 0u, bOb + k + 0u);
                __m256i const o1   = pbs_avx2_off8_x16(batchOffPtr, bBitPos + k + 16u, bOb + k + 16u);
                __m256i const off8 = _mm256_permute4x64_epi64(_mm256_packus_epi16(o0, o1), 0xD8);
                __m256i const bins = _mm256_loadu_si256((const __m256i_u*)(const void*)(batchBins + k));
                __m256i const bIdx = _mm256_and_si256(bins, m0F);
                __m256i const sel0 = _mm256_slli_epi16(bins, 3);
                __m256i const l01  = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut0, bIdx), _mm256_shuffle_epi8(lut1, bIdx), sel0);
                __m256i const l23  = _mm256_blendv_epi8(_mm256_shuffle_epi8(lut2, bIdx), _mm256_shuffle_epi8(lut3, bIdx), sel0);
                __m256i const lowVec = _mm256_blendv_epi8(l01, l23, _mm256_slli_epi16(bins, 2));
                _mm256_storeu_si256((__m256i_u*)(void*)(d8 + k), pbs_avx2_latent8(_mm256_add_epi8(lowVec, off8), mode));
            }
        } else {
            for (; k + 32u <= fastN32; k += 32u) {
                __m256i const o0   = pbs_avx2_off8_x16(batchOffPtr, bBitPos + k + 0u, bOb + k + 0u);
                __m256i const o1   = pbs_avx2_off8_x16(batchOffPtr, bBitPos + k + 16u, bOb + k + 16u);
                __m256i const off8 = _mm256_permute4x64_epi64(_mm256_packus_epi16(o0, o1), 0xD8);
                __m256i lowVec;
                if (bComb != NULL) {
                    lowVec = pbs_avx2_comb_bytes_x32(bComb + k, 0);
                } else {
                    const uint8_t* const bp = batchBins + k;
                    lowVec = _mm256_setr_epi8(
                            (char)l8[bp[0]], (char)l8[bp[1]], (char)l8[bp[2]], (char)l8[bp[3]],
                            (char)l8[bp[4]], (char)l8[bp[5]], (char)l8[bp[6]], (char)l8[bp[7]],
                            (char)l8[bp[8]], (char)l8[bp[9]], (char)l8[bp[10]], (char)l8[bp[11]],
                            (char)l8[bp[12]], (char)l8[bp[13]], (char)l8[bp[14]], (char)l8[bp[15]],
                            (char)l8[bp[16]], (char)l8[bp[17]], (char)l8[bp[18]], (char)l8[bp[19]],
                            (char)l8[bp[20]], (char)l8[bp[21]], (char)l8[bp[22]], (char)l8[bp[23]],
                            (char)l8[bp[24]], (char)l8[bp[25]], (char)l8[bp[26]], (char)l8[bp[27]],
                            (char)l8[bp[28]], (char)l8[bp[29]], (char)l8[bp[30]], (char)l8[bp[31]]);
                }
                _mm256_storeu_si256((__m256i_u*)(void*)(d8 + k), pbs_avx2_latent8(_mm256_add_epi8(lowVec, off8), mode));
            }
        }
    } else if (eltWidth == 2u) {
        uint16_t* const d16       = (uint16_t*)outPtr;
        const uint16_t* const l16 = (const uint16_t*)lowTyped;
        __m256i const deint = _mm256_setr_epi8(
                0,2,4,6,8,10,12,14, 1,3,5,7,9,11,13,15, 0,2,4,6,8,10,12,14, 1,3,5,7,9,11,13,15);
        if (numBins <= 16u && fastN32 >= 16u) {
            __m256i const p0    = _mm256_permute4x64_epi64(
                    _mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i_u*)(const void*)l16), deint), 0xD8);
            __m128i const lutLo = _mm256_castsi256_si128(p0);
            __m128i const lutHi = _mm256_extracti128_si256(p0, 1);
            for (; k + 16u <= fastN32; k += 16u) {
                __m256i const off16 = pbs_avx2_off16_x16(batchOffPtr, bBitPos + k, bOb + k, maxOb);
                __m128i const bins   = _mm_loadu_si128((const __m128i_u*)(const void*)(batchBins + k));
                __m128i const loB    = _mm_shuffle_epi8(lutLo, bins);
                __m128i const hiB    = _mm_shuffle_epi8(lutHi, bins);
                __m256i const lowVec = _mm256_set_m128i(_mm_unpackhi_epi8(loB, hiB), _mm_unpacklo_epi8(loB, hiB));
                _mm256_storeu_si256((__m256i_u*)(void*)(d16 + k), pbs_avx2_latent16(_mm256_add_epi16(lowVec, off16), mode));
            }
        } else if (numBins <= 64u && fastN32 >= 16u) {
            __m256i const p0 = _mm256_permute4x64_epi64(_mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i_u*)(const void*)(l16 + 0)), deint), 0xD8);
            __m256i const p1 = _mm256_permute4x64_epi64(_mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i_u*)(const void*)(l16 + 16)), deint), 0xD8);
            __m256i const p2 = _mm256_permute4x64_epi64(_mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i_u*)(const void*)(l16 + 32)), deint), 0xD8);
            __m256i const p3 = _mm256_permute4x64_epi64(_mm256_shuffle_epi8(_mm256_loadu_si256((const __m256i_u*)(const void*)(l16 + 48)), deint), 0xD8);
            __m128i const lo0 = _mm256_castsi256_si128(p0), hi0 = _mm256_extracti128_si256(p0, 1);
            __m128i const lo1 = _mm256_castsi256_si128(p1), hi1 = _mm256_extracti128_si256(p1, 1);
            __m128i const lo2 = _mm256_castsi256_si128(p2), hi2 = _mm256_extracti128_si256(p2, 1);
            __m128i const lo3 = _mm256_castsi256_si128(p3), hi3 = _mm256_extracti128_si256(p3, 1);
            __m128i const m0F = _mm_set1_epi8(0x0F);
            for (; k + 16u <= fastN32; k += 16u) {
                __m256i const off16 = pbs_avx2_off16_x16(batchOffPtr, bBitPos + k, bOb + k, maxOb);
                __m128i const bins   = _mm_loadu_si128((const __m128i_u*)(const void*)(batchBins + k));
                __m128i const idx    = _mm_and_si128(bins, m0F);
                __m128i const m1     = _mm_slli_epi16(bins, 3);
                __m128i const m2     = _mm_slli_epi16(bins, 2);
                __m128i const loB    = _mm_blendv_epi8(
                        _mm_blendv_epi8(_mm_shuffle_epi8(lo0, idx), _mm_shuffle_epi8(lo1, idx), m1),
                        _mm_blendv_epi8(_mm_shuffle_epi8(lo2, idx), _mm_shuffle_epi8(lo3, idx), m1),
                        m2);
                __m128i const hiB    = _mm_blendv_epi8(
                        _mm_blendv_epi8(_mm_shuffle_epi8(hi0, idx), _mm_shuffle_epi8(hi1, idx), m1),
                        _mm_blendv_epi8(_mm_shuffle_epi8(hi2, idx), _mm_shuffle_epi8(hi3, idx), m1),
                        m2);
                __m256i const lowVec = _mm256_set_m128i(_mm_unpackhi_epi8(loB, hiB), _mm_unpacklo_epi8(loB, hiB));
                _mm256_storeu_si256((__m256i_u*)(void*)(d16 + k), pbs_avx2_latent16(_mm256_add_epi16(lowVec, off16), mode));
            }
        } else {
            for (; k + 16u <= fastN32; k += 16u) {
                __m256i const off16 = pbs_avx2_off16_x16(batchOffPtr, bBitPos + k, bOb + k, maxOb);
                __m256i lowVec;
                if (bComb != NULL) {
                    lowVec = pbs_avx2_comb_lo16_x16(bComb + k);
                } else {
                    const uint8_t* const bp = batchBins + k;
                    lowVec = _mm256_setr_epi16(
                            (short)l16[bp[0]], (short)l16[bp[1]], (short)l16[bp[2]], (short)l16[bp[3]],
                            (short)l16[bp[4]], (short)l16[bp[5]], (short)l16[bp[6]], (short)l16[bp[7]],
                            (short)l16[bp[8]], (short)l16[bp[9]], (short)l16[bp[10]], (short)l16[bp[11]],
                            (short)l16[bp[12]], (short)l16[bp[13]], (short)l16[bp[14]], (short)l16[bp[15]]);
                }
                _mm256_storeu_si256((__m256i_u*)(void*)(d16 + k), pbs_avx2_latent16(_mm256_add_epi16(lowVec, off16), mode));
            }
        }
    } else if (eltWidth == 4u) {
        uint32_t* const d32       = (uint32_t*)outPtr;
        const uint32_t* const l32 = (const uint32_t*)lowTyped;
        size_t const lim = (maxOb <= 25u) ? fastN32 : fastN64;
        for (; k + 8u <= lim; k += 8u) {
            __m256i off32;
            if (maxOb <= 25u) {
                off32 = pbs_avx2_off32_x8(batchOffPtr, bBitPos + k, bOb + k);
            } else {
                __m256i const q0 = pbs_avx2_off64_x4(batchOffPtr, bBitPos + k + 0u, bOb + k + 0u);
                __m256i const q1 = pbs_avx2_off64_x4(batchOffPtr, bBitPos + k + 4u, bOb + k + 4u);
                off32 = _mm256_permute4x64_epi64(_mm256_castps_si256(
                        _mm256_shuffle_ps(_mm256_castsi256_ps(q0), _mm256_castsi256_ps(q1), 0x88)), 0xD8);
            }
            const uint8_t* const bp = batchBins + k;
            __m256i const lowVec = _mm256_setr_epi32(
                    (int)l32[bp[0]], (int)l32[bp[1]], (int)l32[bp[2]], (int)l32[bp[3]],
                    (int)l32[bp[4]], (int)l32[bp[5]], (int)l32[bp[6]], (int)l32[bp[7]]);
            _mm256_storeu_si256((__m256i_u*)(void*)(d32 + k), pbs_avx2_latent32(_mm256_add_epi32(lowVec, off32), mode));
        }
    } else if (maxOb <= 25u) {
        uint64_t* const d64       = (uint64_t*)outPtr;
        const uint64_t* const l64 = (const uint64_t*)lowTyped;
        for (; k + 8u <= fastN32; k += 8u) {
            __m256i const off32     = pbs_avx2_off32_x8(batchOffPtr, bBitPos + k, bOb + k);
            __m256i const off64Lo   = _mm256_cvtepu32_epi64(_mm256_castsi256_si128(off32));
            __m256i const off64Hi   = _mm256_cvtepu32_epi64(_mm256_extracti128_si256(off32, 1));
            const uint8_t* const bp = batchBins + k;
            __m256i const lowLo = _mm256_setr_epi64x(
                    (long long)l64[bp[0]], (long long)l64[bp[1]], (long long)l64[bp[2]], (long long)l64[bp[3]]);
            __m256i const lowHi = _mm256_setr_epi64x(
                    (long long)l64[bp[4]], (long long)l64[bp[5]], (long long)l64[bp[6]], (long long)l64[bp[7]]);
            _mm256_storeu_si256((__m256i_u*)(void*)(d64 + k + 0u), pbs_avx2_latent64(_mm256_add_epi64(lowLo, off64Lo), mode));
            _mm256_storeu_si256((__m256i_u*)(void*)(d64 + k + 4u), pbs_avx2_latent64(_mm256_add_epi64(lowHi, off64Hi), mode));
        }
    } else if (maxOb <= 57u) {
        uint64_t* const d64       = (uint64_t*)outPtr;
        const uint64_t* const l64 = (const uint64_t*)lowTyped;
        for (; k + 4u <= fastN64; k += 4u) {
            __m256i const off64     = pbs_avx2_off64_x4(batchOffPtr, bBitPos + k, bOb + k);
            const uint8_t* const bp = batchBins + k;
            __m256i const low64 = _mm256_setr_epi64x(
                    (long long)l64[bp[0]], (long long)l64[bp[1]], (long long)l64[bp[2]], (long long)l64[bp[3]]);
            _mm256_storeu_si256((__m256i_u*)(void*)(d64 + k), pbs_avx2_latent64(_mm256_add_epi64(low64, off64), mode));
        }
    } else {
        uint64_t* const d64       = (uint64_t*)outPtr;
        const uint64_t* const l64 = (const uint64_t*)lowTyped;
        for (; k + 4u <= fastN64; k += 4u) {
            __m256i const off64     = pbs_avx2_off64_x4_wide(batchOffPtr, bBitPos + k, bOb + k);
            const uint8_t* const bp = batchBins + k;
            __m256i const low64 = _mm256_setr_epi64x(
                    (long long)l64[bp[0]], (long long)l64[bp[1]], (long long)l64[bp[2]], (long long)l64[bp[3]]);
            _mm256_storeu_si256((__m256i_u*)(void*)(d64 + k), pbs_avx2_latent64(_mm256_add_epi64(low64, off64), mode));
        }
    }
#elif ZL_HAS_WASM_SIMD128
    if (eltWidth == 1u) {
        uint8_t* const d8       = (uint8_t*)outPtr;
        const uint8_t* const l8 = (const uint8_t*)lowTyped;
        for (; k + 16u <= fastN32; k += 16u) {
            v128_t const o0   = pbs_wasm_off16_x8(batchOffPtr, bBitPos + k + 0u, bOb + k + 0u);
            v128_t const o1   = pbs_wasm_off16_x8(batchOffPtr, bBitPos + k + 8u, bOb + k + 8u);
            v128_t const off8 = wasm_i8x16_shuffle(o0, o1, 0,2,4,6,8,10,12,14, 16,18,20,22,24,26,28,30);
            v128_t lowVec;
            if (numBins <= 16u) {
                lowVec = wasm_i8x16_swizzle(wasm_v128_load(l8), wasm_v128_load(batchBins + k));
            } else if (numBins <= 64u) {
                v128_t const b = wasm_v128_load(batchBins + k);
                lowVec = wasm_v128_or(
                        wasm_v128_or(
                                wasm_i8x16_swizzle(wasm_v128_load(l8 + 0u), b),
                                wasm_i8x16_swizzle(wasm_v128_load(l8 + 16u), wasm_i8x16_sub(b, wasm_u8x16_splat(16u)))),
                        wasm_v128_or(
                                wasm_i8x16_swizzle(wasm_v128_load(l8 + 32u), wasm_i8x16_sub(b, wasm_u8x16_splat(32u))),
                                wasm_i8x16_swizzle(wasm_v128_load(l8 + 48u), wasm_i8x16_sub(b, wasm_u8x16_splat(48u)))));
            } else {
                const uint8_t* const bp = batchBins + k;
                lowVec = wasm_u8x16_make(
                        l8[bp[0]], l8[bp[1]], l8[bp[2]], l8[bp[3]], l8[bp[4]], l8[bp[5]], l8[bp[6]], l8[bp[7]],
                        l8[bp[8]], l8[bp[9]], l8[bp[10]], l8[bp[11]], l8[bp[12]], l8[bp[13]], l8[bp[14]], l8[bp[15]]);
            }
            wasm_v128_store(d8 + k, pbs_wasm_latent8(wasm_i8x16_add(lowVec, off8), mode));
        }
    } else if (eltWidth == 2u) {
        uint16_t* const d16       = (uint16_t*)outPtr;
        const uint16_t* const l16 = (const uint16_t*)lowTyped;
        if (numBins <= 16u && fastN32 >= 16u) {
            const uint8_t* const lb = (const uint8_t*)l16;
            v128_t const r0         = wasm_v128_load(lb + 0u);
            v128_t const r1         = wasm_v128_load(lb + 16u);
            v128_t const lutLo      = wasm_i8x16_shuffle(r0, r1, 0,2,4,6,8,10,12,14, 16,18,20,22,24,26,28,30);
            v128_t const lutHi      = wasm_i8x16_shuffle(r0, r1, 1,3,5,7,9,11,13,15, 17,19,21,23,25,27,29,31);
            for (; k + 16u <= fastN32; k += 16u) {
                v128_t const o0 = pbs_wasm_off16_x8(batchOffPtr, bBitPos + k + 0u, bOb + k + 0u);
                v128_t const o1 = pbs_wasm_off16_x8(batchOffPtr, bBitPos + k + 8u, bOb + k + 8u);
                v128_t const b  = wasm_v128_load(batchBins + k);
                v128_t const lo = wasm_i8x16_swizzle(lutLo, b);
                v128_t const hi = wasm_i8x16_swizzle(lutHi, b);
                v128_t const w0 = wasm_i8x16_shuffle(lo, hi, 0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23);
                v128_t const w1 = wasm_i8x16_shuffle(lo, hi, 8,24,9,25,10,26,11,27,12,28,13,29,14,30,15,31);
                wasm_v128_store(d16 + k + 0u, pbs_wasm_latent16(wasm_i16x8_add(w0, o0), mode));
                wasm_v128_store(d16 + k + 8u, pbs_wasm_latent16(wasm_i16x8_add(w1, o1), mode));
            }
        } else if (numBins <= 64u && fastN32 >= 16u) {
            const uint8_t* const lb = (const uint8_t*)l16;
            v128_t lutLo[4], lutHi[4];
            for (unsigned q = 0; q < 4u; ++q) {
                v128_t const r0 = wasm_v128_load(lb + q * 32u + 0u);
                v128_t const r1 = wasm_v128_load(lb + q * 32u + 16u);
                lutLo[q] = wasm_i8x16_shuffle(r0, r1, 0,2,4,6,8,10,12,14, 16,18,20,22,24,26,28,30);
                lutHi[q] = wasm_i8x16_shuffle(r0, r1, 1,3,5,7,9,11,13,15, 17,19,21,23,25,27,29,31);
            }
            for (; k + 16u <= fastN32; k += 16u) {
                v128_t const o0  = pbs_wasm_off16_x8(batchOffPtr, bBitPos + k + 0u, bOb + k + 0u);
                v128_t const o1  = pbs_wasm_off16_x8(batchOffPtr, bBitPos + k + 8u, bOb + k + 8u);
                v128_t const b0  = wasm_v128_load(batchBins + k);
                v128_t const b1  = wasm_i8x16_sub(b0, wasm_u8x16_splat(16u));
                v128_t const b2  = wasm_i8x16_sub(b0, wasm_u8x16_splat(32u));
                v128_t const b3  = wasm_i8x16_sub(b0, wasm_u8x16_splat(48u));
                v128_t const lo  = wasm_v128_or(
                        wasm_v128_or(wasm_i8x16_swizzle(lutLo[0], b0), wasm_i8x16_swizzle(lutLo[1], b1)),
                        wasm_v128_or(wasm_i8x16_swizzle(lutLo[2], b2), wasm_i8x16_swizzle(lutLo[3], b3)));
                v128_t const hi  = wasm_v128_or(
                        wasm_v128_or(wasm_i8x16_swizzle(lutHi[0], b0), wasm_i8x16_swizzle(lutHi[1], b1)),
                        wasm_v128_or(wasm_i8x16_swizzle(lutHi[2], b2), wasm_i8x16_swizzle(lutHi[3], b3)));
                v128_t const w0 = wasm_i8x16_shuffle(lo, hi, 0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23);
                v128_t const w1 = wasm_i8x16_shuffle(lo, hi, 8,24,9,25,10,26,11,27,12,28,13,29,14,30,15,31);
                wasm_v128_store(d16 + k + 0u, pbs_wasm_latent16(wasm_i16x8_add(w0, o0), mode));
                wasm_v128_store(d16 + k + 8u, pbs_wasm_latent16(wasm_i16x8_add(w1, o1), mode));
            }
        }
        for (; k + 8u <= fastN32; k += 8u) {
            v128_t const off16      = pbs_wasm_off16_x8(batchOffPtr, bBitPos + k, bOb + k);
            const uint8_t* const bp = batchBins + k;
            v128_t const lowVec     = wasm_u16x8_make(
                    l16[bp[0]], l16[bp[1]], l16[bp[2]], l16[bp[3]],
                    l16[bp[4]], l16[bp[5]], l16[bp[6]], l16[bp[7]]);
            wasm_v128_store(d16 + k, pbs_wasm_latent16(wasm_i16x8_add(lowVec, off16), mode));
        }
    } else if (eltWidth == 4u && maxOb <= 25u) {
        uint32_t* const d32       = (uint32_t*)outPtr;
        const uint32_t* const l32 = (const uint32_t*)lowTyped;
        for (; k + 4u <= fastN32; k += 4u) {
            v128_t const off32      = pbs_wasm_off32_x4(batchOffPtr, bBitPos + k, bOb + k);
            const uint8_t* const bp = batchBins + k;
            v128_t const lowVec     = wasm_u32x4_make(l32[bp[0]], l32[bp[1]], l32[bp[2]], l32[bp[3]]);
            wasm_v128_store(d32 + k, pbs_wasm_latent32(wasm_i32x4_add(lowVec, off32), mode));
        }
    } else if (eltWidth == 8u && maxOb <= 25u) {
        uint64_t* const d64       = (uint64_t*)outPtr;
        const uint64_t* const l64 = (const uint64_t*)lowTyped;
        for (; k + 4u <= fastN32; k += 4u) {
            v128_t const off32      = pbs_wasm_off32_x4(batchOffPtr, bBitPos + k, bOb + k);
            const uint8_t* const bp = batchBins + k;
            v128_t const lowLo      = wasm_u64x2_make(l64[bp[0]], l64[bp[1]]);
            v128_t const lowHi      = wasm_u64x2_make(l64[bp[2]], l64[bp[3]]);
            wasm_v128_store(d64 + k + 0u, pbs_wasm_latent64(wasm_i64x2_add(lowLo, wasm_u64x2_extend_low_u32x4(off32)), mode));
            wasm_v128_store(d64 + k + 2u, pbs_wasm_latent64(wasm_i64x2_add(lowHi, wasm_u64x2_extend_high_u32x4(off32)), mode));
        }
    }
    (void)fastN64;
#else
    (void)outPtr; (void)eltWidth; (void)mode; (void)batchBins; (void)lowTyped;
    (void)maxOb; (void)batchOffPtr; (void)fastN32; (void)fastN64; (void)bBitPos; (void)bOb;
#endif
    (void)numBins; (void)bComb;
    return k;
}

#define PBS_DEFINE_DECODER(NAME, TYPE, ELT_BITS, MODE, FROM_LATENT)           \
    static ZL_Report pbs_decode_##NAME(                                       \
            ZL_Decoder* dictx,                                                \
            TYPE* dst,                                                        \
            size_t n,                                                         \
            int singleBin,                                                    \
            int allZeroOffsets,                                               \
            size_t numBins,                                                   \
            const uint64_t lowers[PBS_MAX_BINS],                              \
            const uint8_t offsetBits[PBS_MAX_BINS],                           \
            const uint64_t masks[PBS_MAX_BINS],                               \
            unsigned maxOb,                                                   \
            const uint8_t* binIds,                                            \
            const uint8_t* offPtr,                                            \
            size_t packedBytes)                                               \
    {                                                                         \
        ZL_RESULT_DECLARE_SCOPE_REPORT(dictx);                                \
        if (singleBin) {                                                      \
            uint64_t const lo       = lowers[0];                              \
            unsigned const ob       = offsetBits[0];                          \
            uint64_t const mask     = masks[0];                               \
            uint64_t const needBits = (uint64_t)n * (uint64_t)ob;             \
            ZL_ERR_IF_GT((needBits + 7ULL) >> 3, (uint64_t)packedBytes, corruption); \
            if (ob == 0u) {                                                   \
                TYPE const val = (TYPE)(FROM_LATENT(lo, (ELT_BITS)));         \
                for (size_t i = 0; i < n; i++) {                              \
                    dst[i] = val;                                             \
                }                                                             \
                return ZL_returnValue(1);                                     \
            }                                                                 \
            size_t i        = 0;                                              \
            uint64_t bitPos = 0;                                              \
            if (ob <= 57u && packedBytes >= 8u) {                             \
                uint64_t const safeLimit = (uint64_t)(packedBytes - 8u) * 8ULL + 7ULL; \
                size_t fastN             = (needBits - ob <= safeLimit)       \
                        ? n                                                   \
                        : ((size_t)(safeLimit / ob) + 1u);                    \
                if (fastN > n) {                                              \
                    fastN = n;                                                \
                }                                                             \
                while (fastN > 0u && (uint64_t)(fastN - 1u) * ob > safeLimit) { \
                    fastN--;                                                  \
                }                                                             \
                for (; i + 8u <= fastN; i += 8u) {                            \
                    uint64_t const bp0 = bitPos + (uint64_t)ob * 0u;          \
                    uint64_t const bp1 = bitPos + (uint64_t)ob * 1u;          \
                    uint64_t const bp2 = bitPos + (uint64_t)ob * 2u;          \
                    uint64_t const bp3 = bitPos + (uint64_t)ob * 3u;          \
                    uint64_t const bp4 = bitPos + (uint64_t)ob * 4u;          \
                    uint64_t const bp5 = bitPos + (uint64_t)ob * 5u;          \
                    uint64_t const bp6 = bitPos + (uint64_t)ob * 6u;          \
                    uint64_t const bp7 = bitPos + (uint64_t)ob * 7u;          \
                    dst[i + 0u] = (TYPE)(FROM_LATENT(lo + ((ZL_readLE64(offPtr + (bp0 >> 3)) >> (bp0 & 7u)) & mask), (ELT_BITS))); \
                    dst[i + 1u] = (TYPE)(FROM_LATENT(lo + ((ZL_readLE64(offPtr + (bp1 >> 3)) >> (bp1 & 7u)) & mask), (ELT_BITS))); \
                    dst[i + 2u] = (TYPE)(FROM_LATENT(lo + ((ZL_readLE64(offPtr + (bp2 >> 3)) >> (bp2 & 7u)) & mask), (ELT_BITS))); \
                    dst[i + 3u] = (TYPE)(FROM_LATENT(lo + ((ZL_readLE64(offPtr + (bp3 >> 3)) >> (bp3 & 7u)) & mask), (ELT_BITS))); \
                    dst[i + 4u] = (TYPE)(FROM_LATENT(lo + ((ZL_readLE64(offPtr + (bp4 >> 3)) >> (bp4 & 7u)) & mask), (ELT_BITS))); \
                    dst[i + 5u] = (TYPE)(FROM_LATENT(lo + ((ZL_readLE64(offPtr + (bp5 >> 3)) >> (bp5 & 7u)) & mask), (ELT_BITS))); \
                    dst[i + 6u] = (TYPE)(FROM_LATENT(lo + ((ZL_readLE64(offPtr + (bp6 >> 3)) >> (bp6 & 7u)) & mask), (ELT_BITS))); \
                    dst[i + 7u] = (TYPE)(FROM_LATENT(lo + ((ZL_readLE64(offPtr + (bp7 >> 3)) >> (bp7 & 7u)) & mask), (ELT_BITS))); \
                    bitPos += (uint64_t)ob * 8u;                              \
                }                                                             \
                for (; i < fastN; i++) {                                      \
                    uint64_t const off = (ZL_readLE64(offPtr + (bitPos >> 3)) >> (bitPos & 7u)) & mask; \
                    dst[i]             = (TYPE)(FROM_LATENT(lo + off, (ELT_BITS))); \
                    bitPos += ob;                                             \
                }                                                             \
            }                                                                 \
            for (; i < n; i++) {                                              \
                uint64_t const off = pbs_read_bits(offPtr, packedBytes, bitPos, ob); \
                bitPos += ob;                                                 \
                dst[i] = (TYPE)(FROM_LATENT(lo + off, (ELT_BITS)));           \
            }                                                                 \
            return ZL_returnValue(1);                                         \
        }                                                                     \
        if (allZeroOffsets) {                                                 \
            ZL_ERR_IF_NE(packedBytes, 0, corruption);                         \
            ZL_ERR_IF(!pbs_validate_bins(binIds, n, numBins), corruption);    \
            TYPE decodedLowers[PBS_MAX_BINS];                                 \
            for (size_t b = 0; b < numBins; b++) {                            \
                decodedLowers[b] = (TYPE)(FROM_LATENT(lowers[b], (ELT_BITS)));\
            }                                                                 \
            if (numBins < 16u) {                                              \
                memset(decodedLowers + numBins, 0, (16u - numBins) * sizeof(TYPE)); \
            }                                                                 \
            size_t i = pbs_simd_all_zero(dst, sizeof(TYPE), binIds, n, numBins, decodedLowers); \
            for (; i + 8u <= n; i += 8u) {                                    \
                dst[i + 0u] = decodedLowers[binIds[i + 0u]];                  \
                dst[i + 1u] = decodedLowers[binIds[i + 1u]];                  \
                dst[i + 2u] = decodedLowers[binIds[i + 2u]];                  \
                dst[i + 3u] = decodedLowers[binIds[i + 3u]];                  \
                dst[i + 4u] = decodedLowers[binIds[i + 4u]];                  \
                dst[i + 5u] = decodedLowers[binIds[i + 5u]];                  \
                dst[i + 6u] = decodedLowers[binIds[i + 6u]];                  \
                dst[i + 7u] = decodedLowers[binIds[i + 7u]];                  \
            }                                                                 \
            for (; i < n; i++) {                                              \
                dst[i] = decodedLowers[binIds[i]];                            \
            }                                                                 \
            return ZL_returnValue(1);                                         \
        }                                                                     \
        TYPE lowTyped[PBS_MAX_BINS];                                          \
        if (sizeof(TYPE) < 8u) {                                              \
            for (size_t b = 0; b < numBins; b++) {                            \
                lowTyped[b] = (TYPE)lowers[b];                                \
            }                                                                 \
            size_t const padTo = (sizeof(TYPE) <= 2u) ? 64u : 16u;            \
            if (numBins < padTo) {                                            \
                memset(lowTyped + numBins, 0, (padTo - numBins) * sizeof(TYPE)); \
            }                                                                 \
        }                                                                     \
        uint32_t comb[PBS_MAX_BINS];                                          \
        int const useComb = PBS_USE_COMB && sizeof(TYPE) <= 2u && numBins > 64u; \
        if (useComb) {                                                        \
            for (size_t b = 0; b < numBins; b++) {                            \
                comb[b] = (uint32_t)lowTyped[b] | ((uint32_t)offsetBits[b] << 16); \
            }                                                                 \
            if (numBins < PBS_MAX_BINS) {                                     \
                memset(comb + numBins, 0, (PBS_MAX_BINS - numBins) * sizeof(comb[0])); \
            }                                                                 \
        }                                                                     \
        uint64_t bitBase = 0;                                                 \
        size_t i         = 0;                                                 \
        while (i < n) {                                                       \
            size_t const batchN = (n - i < PBS_BATCH_N) ? (n - i) : PBS_BATCH_N; \
            uint16_t bBitPos[PBS_BATCH_N];                                    \
            uint8_t bOb[PBS_BATCH_N];                                         \
            uint32_t bComb[PBS_BATCH_N];                                      \
            uint32_t endBitPos = 0;                                           \
            ZL_ERR_IF(                                                        \
                    !pbs_pass1_batch(                                         \
                            binIds + i,                                       \
                            batchN,                                           \
                            numBins,                                          \
                            offsetBits,                                       \
                            (uint16_t)(bitBase & 7ULL),                       \
                            bBitPos,                                          \
                            bOb,                                              \
                            &endBitPos,                                       \
                            useComb ? comb : NULL,                            \
                            bComb),                                           \
                    corruption);                                              \
            uint64_t const baseAligned = bitBase & ~(uint64_t)7ULL;           \
            uint64_t const endBit      = baseAligned + (uint64_t)endBitPos;   \
            ZL_ERR_IF_GT((endBit + 7ULL) >> 3, (uint64_t)packedBytes, corruption); \
            size_t const baseByte     = (size_t)(bitBase >> 3);               \
            size_t const remBytes     = packedBytes - baseByte;               \
            const uint8_t* batchOffP  = offPtr + baseByte;                    \
            uint8_t padBuf[PBS_BATCH_N * 8u + 40u];                           \
            size_t const endByteBound = ((size_t)endBitPos >> 3) + 16u;       \
            if (ZL_UNLIKELY(endByteBound > remBytes)) {                       \
                if (remBytes > 0u) {                                          \
                    memcpy(padBuf, batchOffP, remBytes);                      \
                }                                                             \
                memset(padBuf + remBytes, 0, 16u);                            \
                batchOffP = padBuf;                                           \
            }                                                                 \
            size_t const fastN32 = (maxOb <= 25u) ? batchN : 0u;              \
            size_t const fastN64 = batchN;                                    \
            TYPE* const outPtr   = dst + i;                                   \
            size_t k             = pbs_simd_pass2(                            \
                    outPtr,                                                   \
                    sizeof(TYPE),                                             \
                    (MODE),                                                   \
                    binIds + i,                                               \
                    (sizeof(TYPE) == 8u) ? (const void*)lowers : (const void*)lowTyped, \
                    numBins,                                                  \
                    maxOb,                                                    \
                    batchOffP,                                                \
                    fastN32,                                                  \
                    fastN64,                                                  \
                    bBitPos,                                                  \
                    bOb,                                                      \
                    useComb ? bComb : NULL);                                  \
            if (maxOb <= 57u) {                                               \
                for (; k < fastN64; k++) {                                    \
                    uint8_t const b    = binIds[i + k];                       \
                    uint16_t const bp  = bBitPos[k];                          \
                    uint64_t const off = (ZL_readLE64(batchOffP + (bp >> 3)) >> (bp & 7u)) & masks[b]; \
                    outPtr[k]          = (TYPE)(FROM_LATENT(lowers[b] + off, (ELT_BITS))); \
                }                                                             \
            } else {                                                          \
                for (; k < fastN64; k++) {                                    \
                    uint8_t const b        = binIds[i + k];                   \
                    uint16_t const bp      = bBitPos[k];                      \
                    const uint8_t* const p = batchOffP + (bp >> 3);           \
                    unsigned const sh      = (unsigned)(bp & 7u);             \
                    uint64_t const word    = (ZL_readLE64(p) >> sh) | (((uint64_t)p[8] << 1) << (63u - sh)); \
                    uint64_t const off     = word & masks[b];                 \
                    outPtr[k]              = (TYPE)(FROM_LATENT(lowers[b] + off, (ELT_BITS))); \
                }                                                             \
            }                                                                 \
            bitBase = endBit;                                                 \
            i += batchN;                                                      \
        }                                                                     \
        return ZL_returnValue(1);                                             \
    }

PBS_DEFINE_DECODER(u8, uint8_t, 8u, ZL_PcoBinSplit_unsigned, PBS_FROM_LATENT_UNSIGNED)
PBS_DEFINE_DECODER(s8, uint8_t, 8u, ZL_PcoBinSplit_signed, PBS_FROM_LATENT_SIGNED)
PBS_DEFINE_DECODER(u16, uint16_t, 16u, ZL_PcoBinSplit_unsigned, PBS_FROM_LATENT_UNSIGNED)
PBS_DEFINE_DECODER(s16, uint16_t, 16u, ZL_PcoBinSplit_signed, PBS_FROM_LATENT_SIGNED)
PBS_DEFINE_DECODER(f16, uint16_t, 16u, ZL_PcoBinSplit_float, PBS_FROM_LATENT_FLOAT)
PBS_DEFINE_DECODER(u32, uint32_t, 32u, ZL_PcoBinSplit_unsigned, PBS_FROM_LATENT_UNSIGNED)
PBS_DEFINE_DECODER(s32, uint32_t, 32u, ZL_PcoBinSplit_signed, PBS_FROM_LATENT_SIGNED)
PBS_DEFINE_DECODER(f32, uint32_t, 32u, ZL_PcoBinSplit_float, PBS_FROM_LATENT_FLOAT)
PBS_DEFINE_DECODER(u64, uint64_t, 64u, ZL_PcoBinSplit_unsigned, PBS_FROM_LATENT_UNSIGNED)
PBS_DEFINE_DECODER(s64, uint64_t, 64u, ZL_PcoBinSplit_signed, PBS_FROM_LATENT_SIGNED)
PBS_DEFINE_DECODER(f64, uint64_t, 64u, ZL_PcoBinSplit_float, PBS_FROM_LATENT_FLOAT)

ZL_Report DI_pco_bin_split(ZL_Decoder* dictx, const ZL_Input* ins[])
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dictx);
    const ZL_Input* const binIn = ins[0];
    const ZL_Input* const offIn = ins[1];
    ZL_ASSERT_EQ(ZL_Input_type(binIn), ZL_Type_numeric);
    ZL_ASSERT_EQ(ZL_Input_type(offIn), ZL_Type_serial);
    ZL_ERR_IF_NE(ZL_Input_eltWidth(binIn), 1, corruption);

    ZL_RBuffer const hdr = ZL_Decoder_getCodecHeader(dictx);
    ZL_ERR_IF_LT(hdr.size, 2, corruption);
    const uint8_t* ptr       = (const uint8_t*)hdr.start;
    const uint8_t* const end = ptr + hdr.size;

    uint8_t const flags      = *ptr++;
    unsigned const wLog2     = flags & 3u;
    int const latentMode     = (int)((flags >> 2) & 3u);
    int const singleBin      = (int)((flags >> 4) & 1u);
    int const allZeroOffsets = (int)((flags >> 5) & 1u);
    ZL_ERR_IF_GT(latentMode, 2, corruption);

    size_t const eltWidth  = (size_t)1u << wLog2;
    unsigned const eltBits = (unsigned)(eltWidth * 8u);

    ZL_TRY_LET(uint64_t, nU64, ZL_varintDecode(&ptr, end));
    size_t const n = (size_t)nU64;
    ZL_ERR_IF_NE((uint64_t)n, nU64, corruption);

    ZL_Output* const out = ZL_Decoder_create1OutStream(dictx, n, eltWidth);
    ZL_ERR_IF_NULL(out, allocation);

    if (n == 0) {
        ZL_ERR_IF_NE(ZL_Input_numElts(binIn), 0, corruption);
        ZL_ERR_IF_NE(ZL_Input_numElts(offIn), 0, corruption);
        ZL_ERR_IF_ERR(ZL_Output_commit(out, 0));
        return ZL_returnValue(1);
    }

    size_t numBins = 1u;
    if (!singleBin) {
        ZL_ERR_IF_GE(ptr, end, corruption);
        numBins = (size_t)(*ptr++) + 1u;
    }

    uint64_t lowers[PBS_MAX_BINS];
    uint8_t offsetBits[PBS_MAX_BINS];
    uint64_t masks[PBS_MAX_BINS];
    uint64_t prevLower = 0;
    unsigned maxOb     = 0;
    for (size_t b = 0; b < numBins; b++) {
        ZL_TRY_LET(uint64_t, deltaLo, ZL_varintDecode(&ptr, end));
        lowers[b] = prevLower + deltaLo;
        prevLower = lowers[b];
        if (allZeroOffsets) {
            offsetBits[b] = 0;
            masks[b]      = 0;
        } else {
            ZL_ERR_IF_GE(ptr, end, corruption);
            uint8_t const ob = *ptr++;
            ZL_ERR_IF_GT(ob, eltBits, corruption);
            offsetBits[b] = ob;
            masks[b] = (ob == 0u) ? 0ULL
                    : (ob == 64u) ? UINT64_MAX
                                  : ((1ULL << ob) - 1ULL);
            if (ob > maxOb) {
                maxOb = ob;
            }
        }
    }
    ZL_ERR_IF_NE(ptr, end, corruption);
    if (!singleBin && !allZeroOffsets && numBins < PBS_MAX_BINS) {
        memset(offsetBits + numBins, 0, PBS_MAX_BINS - numBins);
    }

    ZL_ERR_IF_NE(ZL_Input_numElts(binIn), singleBin ? 0 : n, corruption);
    const uint8_t* const offPtr = (const uint8_t*)ZL_Input_ptr(offIn);
    size_t const packedBytes    = ZL_Input_numElts(offIn);
    const uint8_t* const binIds = (const uint8_t*)ZL_Input_ptr(binIn);
    void* const dst             = ZL_Output_ptr(out);

    unsigned const dispatchKey = (wLog2 << 2)
            | (unsigned)((wLog2 == 0u && latentMode == ZL_PcoBinSplit_float)
                                 ? ZL_PcoBinSplit_unsigned
                                 : latentMode);
    ZL_Report decRes = ZL_returnValue(1);
    switch (dispatchKey) {
        case (0u << 2) | ZL_PcoBinSplit_unsigned:
            decRes = pbs_decode_u8(
                    dictx, (uint8_t*)dst, n, singleBin, allZeroOffsets, numBins,
                    lowers, offsetBits, masks, maxOb, binIds, offPtr, packedBytes);
            break;
        case (0u << 2) | ZL_PcoBinSplit_signed:
            decRes = pbs_decode_s8(
                    dictx, (uint8_t*)dst, n, singleBin, allZeroOffsets, numBins,
                    lowers, offsetBits, masks, maxOb, binIds, offPtr, packedBytes);
            break;
        case (1u << 2) | ZL_PcoBinSplit_unsigned:
            decRes = pbs_decode_u16(
                    dictx, (uint16_t*)dst, n, singleBin, allZeroOffsets, numBins,
                    lowers, offsetBits, masks, maxOb, binIds, offPtr, packedBytes);
            break;
        case (1u << 2) | ZL_PcoBinSplit_signed:
            decRes = pbs_decode_s16(
                    dictx, (uint16_t*)dst, n, singleBin, allZeroOffsets, numBins,
                    lowers, offsetBits, masks, maxOb, binIds, offPtr, packedBytes);
            break;
        case (1u << 2) | ZL_PcoBinSplit_float:
            decRes = pbs_decode_f16(
                    dictx, (uint16_t*)dst, n, singleBin, allZeroOffsets, numBins,
                    lowers, offsetBits, masks, maxOb, binIds, offPtr, packedBytes);
            break;
        case (2u << 2) | ZL_PcoBinSplit_unsigned:
            decRes = pbs_decode_u32(
                    dictx, (uint32_t*)dst, n, singleBin, allZeroOffsets, numBins,
                    lowers, offsetBits, masks, maxOb, binIds, offPtr, packedBytes);
            break;
        case (2u << 2) | ZL_PcoBinSplit_signed:
            decRes = pbs_decode_s32(
                    dictx, (uint32_t*)dst, n, singleBin, allZeroOffsets, numBins,
                    lowers, offsetBits, masks, maxOb, binIds, offPtr, packedBytes);
            break;
        case (2u << 2) | ZL_PcoBinSplit_float:
            decRes = pbs_decode_f32(
                    dictx, (uint32_t*)dst, n, singleBin, allZeroOffsets, numBins,
                    lowers, offsetBits, masks, maxOb, binIds, offPtr, packedBytes);
            break;
        case (3u << 2) | ZL_PcoBinSplit_unsigned:
            decRes = pbs_decode_u64(
                    dictx, (uint64_t*)dst, n, singleBin, allZeroOffsets, numBins,
                    lowers, offsetBits, masks, maxOb, binIds, offPtr, packedBytes);
            break;
        case (3u << 2) | ZL_PcoBinSplit_signed:
            decRes = pbs_decode_s64(
                    dictx, (uint64_t*)dst, n, singleBin, allZeroOffsets, numBins,
                    lowers, offsetBits, masks, maxOb, binIds, offPtr, packedBytes);
            break;
        case (3u << 2) | ZL_PcoBinSplit_float:
            decRes = pbs_decode_f64(
                    dictx, (uint64_t*)dst, n, singleBin, allZeroOffsets, numBins,
                    lowers, offsetBits, masks, maxOb, binIds, offPtr, packedBytes);
            break;
        default:
            ZL_ASSERT_FAIL("Unhandled pco_bin_split dispatch key");
            return ZL_REPORT_ERROR(GENERIC);
    }
    ZL_ERR_IF_ERR(decRes);
    ZL_ERR_IF_ERR(ZL_Output_commit(out, n));
    return ZL_returnValue(1);
}
