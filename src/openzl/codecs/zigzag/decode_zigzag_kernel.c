// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "decode_zigzag_kernel.h"

#include <assert.h>
#include <stdbool.h>

#include "openzl/shared/portability.h"

#if ZL_HAS_NEON
#    include <arm_neon.h>
#endif

/* SIMD kernels : zigzagDecodeSimd##BITS()
 * dst[i] = (src[i] >> 1) ^ -(src[i] & 1), for a prefix of the array.
 * @return nb of elements processed (the scalar loop finishes the job) */
#if ZL_HAS_AVX2

/* AVX2 has no 8-bit shift : shift 16-bit lanes, then clear the bit which
 * crossed over from the neighbour byte. */
#    define ZZ_SRLI1_8(v) \
        _mm256_and_si256(_mm256_srli_epi16(v, 1), _mm256_set1_epi8(0x7F))
#    define ZZ_SRLI1_16(v) _mm256_srli_epi16(v, 1)
#    define ZZ_SRLI1_32(v) _mm256_srli_epi32(v, 1)
#    define ZZ_SRLI1_64(v) _mm256_srli_epi64(v, 1)
#    define ZZ_ONE_8 _mm256_set1_epi8(1)
#    define ZZ_ONE_16 _mm256_set1_epi16(1)
#    define ZZ_ONE_32 _mm256_set1_epi32(1)
#    define ZZ_ONE_64 _mm256_set1_epi64x(1)

#    define GEN_ZIGZAG_DECODE_SIMD(BITS)                                      \
        static size_t zigzagDecodeSimd##BITS(                                 \
                int##BITS##_t* dst, const uint##BITS##_t* src, size_t nbElts) \
        {                                                                     \
            size_t const step  = 256 / BITS;                                  \
            __m256i const one  = ZZ_ONE_##BITS;                               \
            __m256i const zero = _mm256_setzero_si256();                      \
            size_t i           = 0;                                           \
            for (; i + step <= nbElts; i += step) {                           \
                __m256i const z = _mm256_loadu_si256(                         \
                        (const __m256i_u*)(const void*)(src + i));            \
                __m256i const mask =                                          \
                        _mm256_sub_epi##BITS(zero, _mm256_and_si256(z, one)); \
                _mm256_storeu_si256(                                          \
                        (__m256i_u*)(void*)(dst + i),                         \
                        _mm256_xor_si256(ZZ_SRLI1_##BITS(z), mask));          \
            }                                                                 \
            return i;                                                         \
        }

GEN_ZIGZAG_DECODE_SIMD(8)
GEN_ZIGZAG_DECODE_SIMD(16)
GEN_ZIGZAG_DECODE_SIMD(32)
GEN_ZIGZAG_DECODE_SIMD(64)
#    undef GEN_ZIGZAG_DECODE_SIMD

#elif ZL_HAS_WASM_SIMD128

#    define GEN_ZIGZAG_DECODE_SIMD(BITS, LANES)                               \
        static size_t zigzagDecodeSimd##BITS(                                 \
                int##BITS##_t* dst, const uint##BITS##_t* src, size_t nbElts) \
        {                                                                     \
            size_t const step = LANES;                                        \
            v128_t const one  = wasm_i##BITS##x##LANES##_splat(1);            \
            size_t i          = 0;                                            \
            for (; i + step <= nbElts; i += step) {                           \
                v128_t const z = wasm_v128_load(src + i);                     \
                v128_t const mask =                                           \
                        wasm_i##BITS##x##LANES##_neg(wasm_v128_and(z, one));  \
                wasm_v128_store(                                              \
                        dst + i,                                              \
                        wasm_v128_xor(                                        \
                                wasm_u##BITS##x##LANES##_shr(z, 1), mask));   \
            }                                                                 \
            return i;                                                         \
        }

GEN_ZIGZAG_DECODE_SIMD(8, 16)
GEN_ZIGZAG_DECODE_SIMD(16, 8)
GEN_ZIGZAG_DECODE_SIMD(32, 4)
GEN_ZIGZAG_DECODE_SIMD(64, 2)
#    undef GEN_ZIGZAG_DECODE_SIMD

#elif ZL_HAS_NEON

#    define GEN_ZIGZAG_DECODE_SIMD(BITS, LANES)                               \
        static size_t zigzagDecodeSimd##BITS(                                 \
                int##BITS##_t* dst, const uint##BITS##_t* src, size_t nbElts) \
        {                                                                     \
            size_t const step                  = LANES;                       \
            uint##BITS##x##LANES##_t const one = vdupq_n_u##BITS(1);          \
            size_t i                           = 0;                           \
            for (; i + step <= nbElts; i += step) {                           \
                uint##BITS##x##LANES##_t const z = vld1q_u##BITS(src + i);    \
                int##BITS##x##LANES##_t const mask =                          \
                        vnegq_s##BITS(vreinterpretq_s##BITS##_u##BITS(        \
                                vandq_u##BITS(z, one)));                      \
                vst1q_s##BITS(                                                \
                        dst + i,                                              \
                        veorq_s##BITS(                                        \
                                vreinterpretq_s##BITS##_u##BITS(              \
                                        vshrq_n_u##BITS(z, 1)),               \
                                mask));                                       \
            }                                                                 \
            return i;                                                         \
        }

GEN_ZIGZAG_DECODE_SIMD(8, 16)
GEN_ZIGZAG_DECODE_SIMD(16, 8)
GEN_ZIGZAG_DECODE_SIMD(32, 4)
GEN_ZIGZAG_DECODE_SIMD(64, 2)
#    undef GEN_ZIGZAG_DECODE_SIMD

#else

#    define GEN_ZIGZAG_DECODE_SIMD(BITS)                                      \
        static size_t zigzagDecodeSimd##BITS(                                 \
                int##BITS##_t* dst, const uint##BITS##_t* src, size_t nbElts) \
        {                                                                     \
            (void)dst, (void)src, (void)nbElts;                               \
            return 0;                                                         \
        }
GEN_ZIGZAG_DECODE_SIMD(8)
GEN_ZIGZAG_DECODE_SIMD(16)
GEN_ZIGZAG_DECODE_SIMD(32)
GEN_ZIGZAG_DECODE_SIMD(64)
#    undef GEN_ZIGZAG_DECODE_SIMD

#endif

#define GEN_ZIGZAG_DECODE(BITS)                                               \
    void ZL_zigzagDecode##BITS(                                               \
            int##BITS##_t* dst, const uint##BITS##_t* src, size_t nbElts)     \
    {                                                                         \
        for (size_t i = zigzagDecodeSimd##BITS(dst, src, nbElts); i < nbElts; \
             i++) {                                                           \
            uint##BITS##_t const z = src[i];                                  \
            uint##BITS##_t const mask =                                       \
                    (uint##BITS##_t)((uint##BITS##_t)0 - (z & 0x1));          \
            dst[i] = (int##BITS##_t)((z >> 1) ^ mask);                        \
        }                                                                     \
    }

GEN_ZIGZAG_DECODE(64)
GEN_ZIGZAG_DECODE(32)
GEN_ZIGZAG_DECODE(16)
GEN_ZIGZAG_DECODE(8)
#undef GEN_ZIGZAG_DECODE

void ZL_zigzagDecode(void* dst, const void* src, size_t nbElts, size_t eltWidth)
{
    switch (eltWidth) {
        case 1:
            ZL_zigzagDecode8((int8_t*)dst, (uint8_t const*)src, nbElts);
            break;
        case 2:
            ZL_zigzagDecode16((int16_t*)dst, (uint16_t const*)src, nbElts);
            break;
        case 4:
            ZL_zigzagDecode32((int32_t*)dst, (uint32_t const*)src, nbElts);
            break;
        case 8:
            ZL_zigzagDecode64((int64_t*)dst, (uint64_t const*)src, nbElts);
            break;
        default:
            assert(false);
            break;
    }
}
