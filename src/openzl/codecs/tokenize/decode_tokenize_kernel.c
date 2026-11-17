// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/tokenize/decode_tokenize_kernel.h"

#include <string.h>

#include "openzl/codecs/common/copy.h"
#include "openzl/common/assertion.h"
#include "openzl/shared/portability.h"

ZL_FORCE_INLINE uint64_t
readIndexAt(void const* indices, size_t i, size_t idxWidth)
{
    if (idxWidth == 1) {
        return *((uint8_t const*)indices + i);
    }
    if (idxWidth == 2) {
        return *((uint16_t const*)indices + i);
    }
    if (idxWidth == 4) {
        return *((uint32_t const*)indices + i);
    }
    if (idxWidth == 8) {
        return *((uint64_t const*)indices + i);
    }
    ZL_ASSERT_FAIL("Impossible");
    return 0;
}

ZL_FORCE_INLINE bool tokenizeValidateIndicesImpl(
        size_t alphabetSize,
        void const* indices,
        size_t nbElts,
        size_t idxWidth)
{
    if (alphabetSize == 0) {
        return nbElts == 0;
    }
    if (idxWidth == 1) {
        if (alphabetSize >= 256u) {
            return true;
        }
        uint8_t const* const p = (uint8_t const*)indices;
        uint8_t const maxOk    = (uint8_t)(alphabetSize - 1u);
        size_t i               = 0;
#if ZL_HAS_AVX2
        __m256i const lim = _mm256_set1_epi8((char)maxOk);
        for (; i + 32u <= nbElts; i += 32u) {
            __m256i const v    = _mm256_loadu_si256((const __m256i_u*)(p + i));
            __m256i const diff = _mm256_subs_epu8(v, lim);
            if (ZL_UNLIKELY(!_mm256_testz_si256(diff, diff))) {
                return false;
            }
        }
#elif ZL_HAS_WASM_SIMD128
        v128_t const lim = wasm_u8x16_splat(maxOk);
        for (; i + 16u <= nbElts; i += 16u) {
            v128_t const v   = wasm_v128_load(p + i);
            v128_t const bad = wasm_u8x16_gt(v, lim);
            if (ZL_UNLIKELY(wasm_v128_any_true(bad))) {
                return false;
            }
        }
#endif
        for (; i < nbElts; ++i) {
            if (ZL_UNLIKELY(p[i] > maxOk)) {
                return false;
            }
        }
        return true;
    }
    if (idxWidth == 2) {
        if (alphabetSize >= 65536u) {
            return true;
        }
        uint16_t const* const p = (uint16_t const*)indices;
        uint16_t const maxOk    = (uint16_t)(alphabetSize - 1u);
        size_t i                = 0;
#if ZL_HAS_AVX2
        __m256i const lim = _mm256_set1_epi16((short)maxOk);
        for (; i + 16u <= nbElts; i += 16u) {
            __m256i const v    = _mm256_loadu_si256((const __m256i_u*)(p + i));
            __m256i const diff = _mm256_subs_epu16(v, lim);
            if (ZL_UNLIKELY(!_mm256_testz_si256(diff, diff))) {
                return false;
            }
        }
#elif ZL_HAS_WASM_SIMD128
        v128_t const lim = wasm_u16x8_splat(maxOk);
        for (; i + 8u <= nbElts; i += 8u) {
            v128_t const v   = wasm_v128_load(p + i);
            v128_t const bad = wasm_u16x8_gt(v, lim);
            if (ZL_UNLIKELY(wasm_v128_any_true(bad))) {
                return false;
            }
        }
#endif
        for (; i < nbElts; ++i) {
            if (ZL_UNLIKELY(p[i] > maxOk)) {
                return false;
            }
        }
        return true;
    }
    bool bad = false;
    for (size_t i = 0; i < nbElts; ++i) {
        bad |= readIndexAt(indices, i, idxWidth) >= alphabetSize;
    }
    return !bad;
}

#define GEN_TOKENIZE_VALIDATE_INDICES(idxWidth)                      \
    ZL_FORCE_NOINLINE bool tokenizeValidateIndices##idxWidth(        \
            size_t alphabetSize, void const* indices, size_t nbElts) \
    {                                                                \
        return tokenizeValidateIndicesImpl(                          \
                alphabetSize, indices, nbElts, idxWidth);            \
    }

GEN_TOKENIZE_VALIDATE_INDICES(1)
GEN_TOKENIZE_VALIDATE_INDICES(2)
GEN_TOKENIZE_VALIDATE_INDICES(4)
GEN_TOKENIZE_VALIDATE_INDICES(8)

bool ZS_tokenizeValidateIndices(
        size_t alphabetSize,
        const void* indices,
        size_t nbElts,
        size_t idxWidth)
{
    switch (idxWidth) {
        case 1:
            return tokenizeValidateIndices1(alphabetSize, indices, nbElts);
        case 2:
            return tokenizeValidateIndices2(alphabetSize, indices, nbElts);
        case 4:
            return tokenizeValidateIndices4(alphabetSize, indices, nbElts);
        case 8:
            return tokenizeValidateIndices8(alphabetSize, indices, nbElts);
        default:
            ZL_ASSERT_FAIL("Bad idxWidth = %u", (unsigned)idxWidth);
            return false;
    }
}

ZL_FORCE_INLINE void writeSymbolAt(
        void* dst,
        size_t dstIdx,
        void const* alphabet,
        size_t idx,
        size_t eltWidth)
{
    void* const dstPtr       = (uint8_t*)dst + dstIdx * eltWidth;
    void const* const srcPtr = (uint8_t const*)alphabet + idx * eltWidth;
    memcpy(dstPtr, srcPtr, eltWidth);
}

ZL_FORCE_INLINE bool tokenizeDecodeEltWidthIdxWidth(
        void* dst,
        void const* alphabet,
        size_t alphabetSize,
        void const* indices,
        size_t nbElts,
        size_t eltWidth,
        size_t idxWidth)
{
    ZL_ASSERT_NE(alphabetSize, 0, "Already validated");
    if (idxWidth == 1 && eltWidth == 2) {
        uint16_t const* const alpha16 = (uint16_t const*)alphabet;
        uint8_t const* const idx8     = (uint8_t const*)indices;
        uint16_t* const dst16         = (uint16_t*)dst;
        size_t i                      = 0;
#if ZL_HAS_AVX2
        if (alphabetSize <= 16u && nbElts >= 32u) {
            uint8_t aLo[16], aHi[16];
            uint16_t const a0 = alpha16[0];
            for (size_t a = 0; a < 16u; ++a) {
                uint16_t const v = (a < alphabetSize) ? alpha16[a] : a0;
                aLo[a]           = (uint8_t)(v & 0xFFu);
                aHi[a]           = (uint8_t)(v >> 8);
            }
            __m256i const vLo = _mm256_broadcastsi128_si256(
                    _mm_loadu_si128((const __m128i*)(const void*)aLo));
            __m256i const vHi = _mm256_broadcastsi128_si256(
                    _mm_loadu_si128((const __m128i*)(const void*)aHi));
            __m256i const vMaxIdx =
                    _mm256_set1_epi8((char)(uint8_t)(alphabetSize - 1u));
            __m256i anyBad = _mm256_setzero_si256();
            for (; i + 32u <= nbElts; i += 32u) {
                __m256i const idx = _mm256_loadu_si256(
                        (const __m256i*)(const void*)(idx8 + i));
                anyBad = _mm256_or_si256(
                        anyBad, _mm256_subs_epu8(idx, vMaxIdx));
                __m256i const rLo = _mm256_shuffle_epi8(vLo, idx);
                __m256i const rHi = _mm256_shuffle_epi8(vHi, idx);
                __m256i const p0  = _mm256_unpacklo_epi8(rLo, rHi);
                __m256i const p1  = _mm256_unpackhi_epi8(rLo, rHi);
                _mm256_storeu_si256((__m256i*)(void*)(dst16 + i),
                        _mm256_permute2x128_si256(p0, p1, 0x20));
                _mm256_storeu_si256((__m256i*)(void*)(dst16 + i + 16u),
                        _mm256_permute2x128_si256(p0, p1, 0x31));
            }
            if (!_mm256_testz_si256(anyBad, anyBad)) {
                i = 0; /* Out-of-range index on corrupted input: run scalar */
            }
        } else if (alphabetSize <= 32u && nbElts >= 32u) {
            uint8_t aLo0[16], aHi0[16], aLo1[16], aHi1[16];
            uint16_t const a0 = alpha16[0];
            for (size_t a = 0; a < 16u; ++a) {
                uint16_t const v0 = alpha16[a];
                uint16_t const v1 =
                        (16u + a < alphabetSize) ? alpha16[16u + a] : a0;
                aLo0[a] = (uint8_t)(v0 & 0xFFu);
                aHi0[a] = (uint8_t)(v0 >> 8);
                aLo1[a] = (uint8_t)(v1 & 0xFFu);
                aHi1[a] = (uint8_t)(v1 >> 8);
            }
            __m256i const vLo0 = _mm256_broadcastsi128_si256(
                    _mm_loadu_si128((const __m128i*)(const void*)aLo0));
            __m256i const vHi0 = _mm256_broadcastsi128_si256(
                    _mm_loadu_si128((const __m128i*)(const void*)aHi0));
            __m256i const vLo1 = _mm256_broadcastsi128_si256(
                    _mm_loadu_si128((const __m128i*)(const void*)aLo1));
            __m256i const vHi1 = _mm256_broadcastsi128_si256(
                    _mm_loadu_si128((const __m128i*)(const void*)aHi1));
            __m256i const vMaxIdx =
                    _mm256_set1_epi8((char)(uint8_t)(alphabetSize - 1u));
            __m256i anyBad = _mm256_setzero_si256();
            for (; i + 32u <= nbElts; i += 32u) {
                __m256i const idx = _mm256_loadu_si256(
                        (const __m256i*)(const void*)(idx8 + i));
                anyBad = _mm256_or_si256(
                        anyBad, _mm256_subs_epu8(idx, vMaxIdx));
                __m256i const sel = _mm256_slli_epi16(idx, 3); /* bit4 -> bit7 */
                __m256i const rLo = _mm256_blendv_epi8(
                        _mm256_shuffle_epi8(vLo0, idx),
                        _mm256_shuffle_epi8(vLo1, idx),
                        sel);
                __m256i const rHi = _mm256_blendv_epi8(
                        _mm256_shuffle_epi8(vHi0, idx),
                        _mm256_shuffle_epi8(vHi1, idx),
                        sel);
                __m256i const p0 = _mm256_unpacklo_epi8(rLo, rHi);
                __m256i const p1 = _mm256_unpackhi_epi8(rLo, rHi);
                _mm256_storeu_si256((__m256i*)(void*)(dst16 + i),
                        _mm256_permute2x128_si256(p0, p1, 0x20));
                _mm256_storeu_si256((__m256i*)(void*)(dst16 + i + 16u),
                        _mm256_permute2x128_si256(p0, p1, 0x31));
            }
            if (!_mm256_testz_si256(anyBad, anyBad)) {
                i = 0;
            }
        }
#elif ZL_HAS_WASM_SIMD128
        if (alphabetSize <= 16u && nbElts >= 16u) {
            uint8_t aLo[16], aHi[16];
            uint16_t const a0 = alpha16[0];
            for (size_t a = 0; a < 16u; ++a) {
                uint16_t const v = (a < alphabetSize) ? alpha16[a] : a0;
                aLo[a]           = (uint8_t)(v & 0xFFu);
                aHi[a]           = (uint8_t)(v >> 8);
            }
            v128_t const vLo     = wasm_v128_load(aLo);
            v128_t const vHi     = wasm_v128_load(aHi);
            v128_t const vMaxIdx = wasm_u8x16_splat((uint8_t)(alphabetSize - 1u));
            v128_t anyBad        = wasm_i64x2_const(0, 0);
            for (; i + 16u <= nbElts; i += 16u) {
                v128_t const idx = wasm_v128_load(idx8 + i);
                anyBad           = wasm_v128_or(anyBad, wasm_u8x16_sub_sat(idx, vMaxIdx));
                v128_t const rLo = wasm_i8x16_swizzle(vLo, idx);
                v128_t const rHi = wasm_i8x16_swizzle(vHi, idx);
                wasm_v128_store(dst16 + i,
                        wasm_i8x16_shuffle(rLo, rHi, 0, 16, 1, 17, 2, 18, 3, 19,
                                           4, 20, 5, 21, 6, 22, 7, 23));
                wasm_v128_store(dst16 + i + 8u,
                        wasm_i8x16_shuffle(rLo, rHi, 8, 24, 9, 25, 10, 26, 11, 27,
                                           12, 28, 13, 29, 14, 30, 15, 31));
            }
            if (wasm_v128_any_true(anyBad)) {
                i = 0;
            }
        }
#endif
        if (i < nbElts) {
            uint16_t padded16[256];
            uint16_t const* alphaPtr16 = alpha16;
            if (alphabetSize < 256u) {
                uint16_t const a0 = alpha16[0];
                for (size_t a = 0; a < alphabetSize; ++a) {
                    padded16[a] = alpha16[a];
                }
                for (size_t a = alphabetSize; a < 256u; ++a) {
                    padded16[a] = a0;
                }
                alphaPtr16 = padded16;
            }
            for (; i + 8u <= nbElts; i += 8u) {
                dst16[i + 0u] = alphaPtr16[idx8[i + 0u]];
                dst16[i + 1u] = alphaPtr16[idx8[i + 1u]];
                dst16[i + 2u] = alphaPtr16[idx8[i + 2u]];
                dst16[i + 3u] = alphaPtr16[idx8[i + 3u]];
                dst16[i + 4u] = alphaPtr16[idx8[i + 4u]];
                dst16[i + 5u] = alphaPtr16[idx8[i + 5u]];
                dst16[i + 6u] = alphaPtr16[idx8[i + 6u]];
                dst16[i + 7u] = alphaPtr16[idx8[i + 7u]];
            }
            for (; i < nbElts; ++i) {
                dst16[i] = alphaPtr16[idx8[i]];
            }
        }
        return true;
    }
    if (idxWidth == 1 && eltWidth <= 8) {
        uint8_t padded[256 * 8];
        void const* alphaPtr = alphabet;
        if (alphabetSize < 256u) {
            memcpy(padded, alphabet, alphabetSize * eltWidth);
            for (size_t a = alphabetSize; a < 256u; ++a) {
                memcpy(padded + a * eltWidth, alphabet, eltWidth);
            }
            alphaPtr = padded;
        }
        uint8_t const* const idx8 = (uint8_t const*)indices;
        size_t i                  = 0;
        for (; i + 8u <= nbElts; i += 8u) {
            writeSymbolAt(dst, i + 0u, alphaPtr, idx8[i + 0u], eltWidth);
            writeSymbolAt(dst, i + 1u, alphaPtr, idx8[i + 1u], eltWidth);
            writeSymbolAt(dst, i + 2u, alphaPtr, idx8[i + 2u], eltWidth);
            writeSymbolAt(dst, i + 3u, alphaPtr, idx8[i + 3u], eltWidth);
            writeSymbolAt(dst, i + 4u, alphaPtr, idx8[i + 4u], eltWidth);
            writeSymbolAt(dst, i + 5u, alphaPtr, idx8[i + 5u], eltWidth);
            writeSymbolAt(dst, i + 6u, alphaPtr, idx8[i + 6u], eltWidth);
            writeSymbolAt(dst, i + 7u, alphaPtr, idx8[i + 7u], eltWidth);
        }
        for (; i < nbElts; ++i) {
            writeSymbolAt(dst, i, alphaPtr, idx8[i], eltWidth);
        }
        return true;
    }
    if (idxWidth == 2 && eltWidth <= 8
        && ZL_LIKELY(tokenizeValidateIndicesImpl(alphabetSize, indices, nbElts, 2))) {
        uint16_t const* const idx16 = (uint16_t const*)indices;
        size_t i                    = 0;
        for (; i + 8u <= nbElts; i += 8u) {
            writeSymbolAt(dst, i + 0u, alphabet, idx16[i + 0u], eltWidth);
            writeSymbolAt(dst, i + 1u, alphabet, idx16[i + 1u], eltWidth);
            writeSymbolAt(dst, i + 2u, alphabet, idx16[i + 2u], eltWidth);
            writeSymbolAt(dst, i + 3u, alphabet, idx16[i + 3u], eltWidth);
            writeSymbolAt(dst, i + 4u, alphabet, idx16[i + 4u], eltWidth);
            writeSymbolAt(dst, i + 5u, alphabet, idx16[i + 5u], eltWidth);
            writeSymbolAt(dst, i + 6u, alphabet, idx16[i + 6u], eltWidth);
            writeSymbolAt(dst, i + 7u, alphabet, idx16[i + 7u], eltWidth);
        }
        for (; i < nbElts; ++i) {
            writeSymbolAt(dst, i, alphabet, idx16[i], eltWidth);
        }
        return true;
    }
    for (size_t i = 0; i < nbElts; ++i) {
        uint64_t index = readIndexAt(indices, i, idxWidth);
        index          = ZL_UNLIKELY(index >= alphabetSize) ? 0 : index;
        writeSymbolAt(dst, i, alphabet, index, eltWidth);
    }
    return true;
}

ZL_FORCE_INLINE bool tokenizeDecodeEltWidth(
        void* dst,
        void const* alphabet,
        size_t alphabetSize,
        void const* indices,
        size_t nbElts,
        size_t eltWidth,
        size_t idxWidth)
{
    if (idxWidth == 1) {
        return tokenizeDecodeEltWidthIdxWidth(
                dst, alphabet, alphabetSize, indices, nbElts, eltWidth, 1);
    }
    if (idxWidth == 2) {
        return tokenizeDecodeEltWidthIdxWidth(
                dst, alphabet, alphabetSize, indices, nbElts, eltWidth, 2);
    }
    if (idxWidth == 4) {
        return tokenizeDecodeEltWidthIdxWidth(
                dst, alphabet, alphabetSize, indices, nbElts, eltWidth, 4);
    }
    if (idxWidth == 8) {
        return tokenizeDecodeEltWidthIdxWidth(
                dst, alphabet, alphabetSize, indices, nbElts, eltWidth, 8);
    }
    ZL_ASSERT_FAIL("Impossible");
    return false;
}

#define GEN_TOKENIZE_DECODE(eltWidth)                \
    ZL_FORCE_NOINLINE bool tokenizeDecode##eltWidth( \
            void* dst,                               \
            void const* alphabet,                    \
            size_t alphabetSize,                     \
            void const* indices,                     \
            size_t nbElts,                           \
            size_t idxWidth)                         \
    {                                                \
        return tokenizeDecodeEltWidth(               \
                dst,                                 \
                alphabet,                            \
                alphabetSize,                        \
                indices,                             \
                nbElts,                              \
                eltWidth,                            \
                idxWidth);                           \
    }

GEN_TOKENIZE_DECODE(1)
GEN_TOKENIZE_DECODE(2)
GEN_TOKENIZE_DECODE(4)
GEN_TOKENIZE_DECODE(8)

ZL_FORCE_NOINLINE bool tokenizeDecodeGeneric(
        void* dst,
        void const* alphabet,
        size_t alphabetSize,
        void const* indices,
        size_t nbElts,
        size_t eltWidth,
        size_t idxWidth)
{
    return tokenizeDecodeEltWidth(
            dst, alphabet, alphabetSize, indices, nbElts, eltWidth, idxWidth);
}

bool ZS_tokenizeDecode(
        void* dst,
        void const* alphabet,
        size_t alphabetSize,
        void const* indices,
        size_t nbElts,
        size_t eltWidth,
        size_t idxWidth)
{
    if (alphabetSize == 0) {
        return nbElts == 0;
    }
    switch (eltWidth) {
        case 1:
            return tokenizeDecode1(
                    dst, alphabet, alphabetSize, indices, nbElts, idxWidth);
        case 2:
            return tokenizeDecode2(
                    dst, alphabet, alphabetSize, indices, nbElts, idxWidth);
        case 4:
            return tokenizeDecode4(
                    dst, alphabet, alphabetSize, indices, nbElts, idxWidth);
        case 8:
            return tokenizeDecode8(
                    dst, alphabet, alphabetSize, indices, nbElts, idxWidth);
        default:
            return tokenizeDecodeGeneric(
                    dst,
                    alphabet,
                    alphabetSize,
                    indices,
                    nbElts,
                    eltWidth,
                    idxWidth);
    }
}

size_t ZS_tokenizeComputeVSFContentSize(
        const uint8_t* const indices,
        size_t const idxWidth,
        size_t const nbElts,
        const uint32_t* const alphabetFieldSizes,
        size_t alphabetSize)
{
    ZL_ASSERT(ZS_tokenizeValidateIndices(
            alphabetSize, indices, nbElts, idxWidth));
    ZL_ASSERT_EQ(sizeof(size_t), 8);
    size_t totalSize = 0;
    for (size_t i = 0; i < nbElts; ++i) {
        totalSize += alphabetFieldSizes[readIndexAt(indices, i, idxWidth)];
    }
    return totalSize;
}

size_t ZS_tokenizeVSFDecodeWorkspaceSize(
        size_t const alphabetSize,
        size_t const alphabetFieldSizesSum)
{
    size_t alphabetStartsSize = sizeof(uint8_t*) * alphabetSize;
    size_t alphabetWildcopyBufferSize =
            alphabetFieldSizesSum + ZS_WILDCOPY_OVERLENGTH;
    return alphabetStartsSize + alphabetWildcopyBufferSize;
}

void ZS_tokenizeVSFDecode(
        const uint8_t* const alphabet,
        size_t const alphabetSize,
        const uint8_t* const indices,
        const uint32_t* const alphabetFieldSizes,
        size_t const alphabetFieldSizesSum,
        uint8_t* const out,
        uint32_t* const dstFieldSizes,
        size_t const dstNbElts,
        size_t const dstNbBytes,
        size_t const idxWidth,
        void* workspace)
{
    ZL_ASSERT(ZS_tokenizeValidateIndices(
            alphabetSize, indices, dstNbElts, idxWidth));

    // Move alphabet into a larger buffer to guarantee wildcopy read safety
    uint8_t* const alphabetBuffer =
            (uint8_t*)workspace + sizeof(uint8_t*) * alphabetSize;
    memcpy(alphabetBuffer, alphabet, alphabetFieldSizesSum);

    // Store the start of each token start for quick access
    const uint8_t** const alphabetStarts = (const uint8_t**)workspace;
    const uint8_t* nextSrcEltPtr         = alphabetBuffer;
    for (size_t i = 0; i < alphabetSize; ++i) {
        alphabetStarts[i] = nextSrcEltPtr;
        nextSrcEltPtr += alphabetFieldSizes[i];
    }

    // Find until when it's safe to wildcopy
    size_t wildcopySafeIndexLimit = dstNbElts > 0 ? dstNbElts - 1 : 0;
    {
        size_t tailSum = 0;
        for (; wildcopySafeIndexLimit > 0; --wildcopySafeIndexLimit) {
            uint64_t const idx =
                    readIndexAt(indices, wildcopySafeIndexLimit, idxWidth);
            ZL_ASSERT_LT(idx, alphabetSize);
            uint32_t const eltWidth = alphabetFieldSizes[idx];
            tailSum += eltWidth;
            if (tailSum > ZS_WILDCOPY_OVERLENGTH) {
                break;
            }
        }
    }
    // Copy tokens to output, use wildcopy until we can't do so safely
    uint8_t* nextDstEltPtr = out;
#if defined(__GNUC__) && defined(__x86_64__)
    // Set alignment to maintain stability in benchmarks
    __asm__(".p2align 6");
#endif
    for (size_t i = 0; i < wildcopySafeIndexLimit; ++i) {
        uint64_t const idx      = readIndexAt(indices, i, idxWidth);
        uint32_t const eltWidth = alphabetFieldSizes[idx];
        ZL_ASSERT_LE(
                nextDstEltPtr + eltWidth + ZS_WILDCOPY_OVERLENGTH,
                out + dstNbBytes);
        ZS_wildcopy(
                nextDstEltPtr, alphabetStarts[idx], eltWidth, ZS_wo_no_overlap);
        nextDstEltPtr += eltWidth;
        dstFieldSizes[i] = eltWidth;
    }
    for (size_t i = wildcopySafeIndexLimit; i < dstNbElts; ++i) {
        uint64_t const idx      = readIndexAt(indices, i, idxWidth);
        uint32_t const eltWidth = alphabetFieldSizes[idx];
        memcpy(nextDstEltPtr, alphabetStarts[idx], eltWidth);
        nextDstEltPtr += eltWidth;
        dstFieldSizes[i] = eltWidth;
    }
}
