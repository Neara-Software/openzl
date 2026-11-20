// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/range_pack/decode_range_pack_kernel.h"

#include "openzl/common/debug.h"
#include "openzl/common/logging.h"
#include "openzl/shared/mem.h"

#include "openzl/shared/portability.h"

#if ZL_HAS_NEON
#    include <arm_neon.h>
#endif

/* SIMD kernels : rangeUnpackSimd_SrcInt_DstInt()
 * dst[i] = (DstInt)src[i] + minValue, for a prefix of the array.
 * @return nb of elements processed (the scalar loop finishes the job) */
#if ZL_HAS_AVX2

#    define RU_LOAD4(p) _mm_cvtsi32_si128((int)ZL_read32(p))
#    define RU_LOAD8(p) _mm_loadl_epi64((const __m128i_u*)(const void*)(p))
#    define RU_LOAD16(p) _mm_loadu_si128((const __m128i_u*)(const void*)(p))
#    define RU_LOAD32(p) _mm256_loadu_si256((const __m256i_u*)(const void*)(p))
#    define RU_NOP(v) (v)

/* STEP: nb elements per iteration, producing one 256-bit vector */
#    define GEN_RANGE_UNPACK_SIMD(                                   \
            DstInt, SrcInt, STEP, LOAD, WIDEN, SET1, ADD)            \
        static size_t rangeUnpackSimd_##SrcInt##_##DstInt(           \
                DstInt* dst,                                         \
                const SrcInt* src,                                   \
                size_t nbElts,                                       \
                DstInt minValue)                                     \
        {                                                            \
            __m256i const vmin = SET1(minValue);                     \
            size_t i           = 0;                                  \
            for (; i + (STEP) <= nbElts; i += (STEP)) {              \
                __m256i const v = WIDEN(LOAD(src + i));              \
                _mm256_storeu_si256(                                 \
                        (__m256i_u*)(void*)(dst + i), ADD(v, vmin)); \
            }                                                        \
            return i;                                                \
        }

#    define RU_SET1_64(v) _mm256_set1_epi64x((long long)(v))
#    define RU_SET1_32(v) _mm256_set1_epi32((int)(v))
#    define RU_SET1_16(v) _mm256_set1_epi16((short)(v))
#    define RU_SET1_8(v) _mm256_set1_epi8((char)(v))

GEN_RANGE_UNPACK_SIMD(
        uint64_t,
        uint64_t,
        4,
        RU_LOAD32,
        RU_NOP,
        RU_SET1_64,
        _mm256_add_epi64)
GEN_RANGE_UNPACK_SIMD(
        uint64_t,
        uint32_t,
        4,
        RU_LOAD16,
        _mm256_cvtepu32_epi64,
        RU_SET1_64,
        _mm256_add_epi64)
GEN_RANGE_UNPACK_SIMD(
        uint64_t,
        uint16_t,
        4,
        RU_LOAD8,
        _mm256_cvtepu16_epi64,
        RU_SET1_64,
        _mm256_add_epi64)
GEN_RANGE_UNPACK_SIMD(
        uint64_t,
        uint8_t,
        4,
        RU_LOAD4,
        _mm256_cvtepu8_epi64,
        RU_SET1_64,
        _mm256_add_epi64)
GEN_RANGE_UNPACK_SIMD(
        uint32_t,
        uint32_t,
        8,
        RU_LOAD32,
        RU_NOP,
        RU_SET1_32,
        _mm256_add_epi32)
GEN_RANGE_UNPACK_SIMD(
        uint32_t,
        uint16_t,
        8,
        RU_LOAD16,
        _mm256_cvtepu16_epi32,
        RU_SET1_32,
        _mm256_add_epi32)
GEN_RANGE_UNPACK_SIMD(
        uint32_t,
        uint8_t,
        8,
        RU_LOAD8,
        _mm256_cvtepu8_epi32,
        RU_SET1_32,
        _mm256_add_epi32)
GEN_RANGE_UNPACK_SIMD(
        uint16_t,
        uint16_t,
        16,
        RU_LOAD32,
        RU_NOP,
        RU_SET1_16,
        _mm256_add_epi16)
GEN_RANGE_UNPACK_SIMD(
        uint16_t,
        uint8_t,
        16,
        RU_LOAD16,
        _mm256_cvtepu8_epi16,
        RU_SET1_16,
        _mm256_add_epi16)
GEN_RANGE_UNPACK_SIMD(
        uint8_t,
        uint8_t,
        32,
        RU_LOAD32,
        RU_NOP,
        RU_SET1_8,
        _mm256_add_epi8)
#    undef GEN_RANGE_UNPACK_SIMD

#elif ZL_HAS_WASM_SIMD128

#    define RU_LOAD16(p) wasm_v128_load(p)
#    define RU_NOP(v) (v)
#    define RU_8TO32(p) wasm_u32x4_extend_low_u16x8(wasm_u16x8_load8x8(p))
#    define RU_8TO64(p)              \
        wasm_u64x2_extend_low_u32x4( \
                wasm_u32x4_extend_low_u16x8(wasm_u16x8_load8x8(p)))
#    define RU_16TO64(p) wasm_u64x2_extend_low_u32x4(wasm_u32x4_load16x4(p))

/* STEP: nb elements per iteration, producing one 128-bit vector
 * READ: nb elements read by one load (>= STEP) */
#    define GEN_RANGE_UNPACK_SIMD(                             \
            DstInt, SrcInt, STEP, READ, LOADWIDEN, SPLAT, ADD) \
        static size_t rangeUnpackSimd_##SrcInt##_##DstInt(     \
                DstInt* dst,                                   \
                const SrcInt* src,                             \
                size_t nbElts,                                 \
                DstInt minValue)                               \
        {                                                      \
            v128_t const vmin = SPLAT(minValue);               \
            size_t i          = 0;                             \
            for (; i + (READ) <= nbElts; i += (STEP)) {        \
                v128_t const v = LOADWIDEN(src + i);           \
                wasm_v128_store(dst + i, ADD(v, vmin));        \
            }                                                  \
            return i;                                          \
        }

GEN_RANGE_UNPACK_SIMD(
        uint64_t,
        uint64_t,
        2,
        2,
        RU_LOAD16,
        wasm_u64x2_splat,
        wasm_i64x2_add)
GEN_RANGE_UNPACK_SIMD(
        uint64_t,
        uint32_t,
        2,
        2,
        wasm_u64x2_load32x2,
        wasm_u64x2_splat,
        wasm_i64x2_add)
GEN_RANGE_UNPACK_SIMD(
        uint64_t,
        uint16_t,
        2,
        4,
        RU_16TO64,
        wasm_u64x2_splat,
        wasm_i64x2_add)
GEN_RANGE_UNPACK_SIMD(
        uint64_t,
        uint8_t,
        2,
        8,
        RU_8TO64,
        wasm_u64x2_splat,
        wasm_i64x2_add)
GEN_RANGE_UNPACK_SIMD(
        uint32_t,
        uint32_t,
        4,
        4,
        RU_LOAD16,
        wasm_u32x4_splat,
        wasm_i32x4_add)
GEN_RANGE_UNPACK_SIMD(
        uint32_t,
        uint16_t,
        4,
        4,
        wasm_u32x4_load16x4,
        wasm_u32x4_splat,
        wasm_i32x4_add)
GEN_RANGE_UNPACK_SIMD(
        uint32_t,
        uint8_t,
        4,
        8,
        RU_8TO32,
        wasm_u32x4_splat,
        wasm_i32x4_add)
GEN_RANGE_UNPACK_SIMD(
        uint16_t,
        uint16_t,
        8,
        8,
        RU_LOAD16,
        wasm_u16x8_splat,
        wasm_i16x8_add)
GEN_RANGE_UNPACK_SIMD(
        uint16_t,
        uint8_t,
        8,
        8,
        wasm_u16x8_load8x8,
        wasm_u16x8_splat,
        wasm_i16x8_add)
GEN_RANGE_UNPACK_SIMD(
        uint8_t,
        uint8_t,
        16,
        16,
        RU_LOAD16,
        wasm_u8x16_splat,
        wasm_i8x16_add)
#    undef GEN_RANGE_UNPACK_SIMD

#elif ZL_HAS_NEON

#    define RU_8TO16(p) vmovl_u8(vld1_u8(p))
#    define RU_8TO32(p) vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(p))))
#    define RU_8TO64(p) \
        vmovl_u32(vget_low_u32(vmovl_u16(vget_low_u16(vmovl_u8(vld1_u8(p))))))
#    define RU_16TO32(p) vmovl_u16(vld1_u16(p))
#    define RU_16TO64(p) vmovl_u32(vget_low_u32(vmovl_u16(vld1_u16(p))))
#    define RU_32TO64(p) vmovl_u32(vld1_u32(p))

/* STEP: nb elements per iteration, producing one 128-bit vector
 * READ: nb elements read by one load (>= STEP) */
#    define GEN_RANGE_UNPACK_SIMD(DstInt, SrcInt, STEP, READ, LOADWIDEN, BITS) \
        static size_t rangeUnpackSimd_##SrcInt##_##DstInt(                     \
                DstInt* dst,                                                   \
                const SrcInt* src,                                             \
                size_t nbElts,                                                 \
                DstInt minValue)                                               \
        {                                                                      \
            uint##BITS##x##STEP##_t const vmin = vdupq_n_u##BITS(minValue);    \
            size_t i                           = 0;                            \
            for (; i + (READ) <= nbElts; i += (STEP)) {                        \
                uint##BITS##x##STEP##_t const v = LOADWIDEN(src + i);          \
                vst1q_u##BITS(dst + i, vaddq_u##BITS(v, vmin));                \
            }                                                                  \
            return i;                                                          \
        }

GEN_RANGE_UNPACK_SIMD(uint64_t, uint64_t, 2, 2, vld1q_u64, 64)
GEN_RANGE_UNPACK_SIMD(uint64_t, uint32_t, 2, 2, RU_32TO64, 64)
GEN_RANGE_UNPACK_SIMD(uint64_t, uint16_t, 2, 4, RU_16TO64, 64)
GEN_RANGE_UNPACK_SIMD(uint64_t, uint8_t, 2, 8, RU_8TO64, 64)
GEN_RANGE_UNPACK_SIMD(uint32_t, uint32_t, 4, 4, vld1q_u32, 32)
GEN_RANGE_UNPACK_SIMD(uint32_t, uint16_t, 4, 4, RU_16TO32, 32)
GEN_RANGE_UNPACK_SIMD(uint32_t, uint8_t, 4, 8, RU_8TO32, 32)
GEN_RANGE_UNPACK_SIMD(uint16_t, uint16_t, 8, 8, vld1q_u16, 16)
GEN_RANGE_UNPACK_SIMD(uint16_t, uint8_t, 8, 8, RU_8TO16, 16)
GEN_RANGE_UNPACK_SIMD(uint8_t, uint8_t, 16, 16, vld1q_u8, 8)
#    undef GEN_RANGE_UNPACK_SIMD

#else

#    define GEN_RANGE_UNPACK_SIMD(DstInt, SrcInt)               \
        static size_t rangeUnpackSimd_##SrcInt##_##DstInt(      \
                DstInt* dst,                                    \
                const SrcInt* src,                              \
                size_t nbElts,                                  \
                DstInt minValue)                                \
        {                                                       \
            (void)dst, (void)src, (void)nbElts, (void)minValue; \
            return 0;                                           \
        }
GEN_RANGE_UNPACK_SIMD(uint64_t, uint64_t)
GEN_RANGE_UNPACK_SIMD(uint64_t, uint32_t)
GEN_RANGE_UNPACK_SIMD(uint64_t, uint16_t)
GEN_RANGE_UNPACK_SIMD(uint64_t, uint8_t)
GEN_RANGE_UNPACK_SIMD(uint32_t, uint32_t)
GEN_RANGE_UNPACK_SIMD(uint32_t, uint16_t)
GEN_RANGE_UNPACK_SIMD(uint32_t, uint8_t)
GEN_RANGE_UNPACK_SIMD(uint16_t, uint16_t)
GEN_RANGE_UNPACK_SIMD(uint16_t, uint8_t)
GEN_RANGE_UNPACK_SIMD(uint8_t, uint8_t)
#    undef GEN_RANGE_UNPACK_SIMD

#endif

#define GEN_RANGE_UNPACK(DstInt, SrcInt)                                    \
    static void rangeUnpack_##SrcInt##_##DstInt(                            \
            DstInt* dst, const SrcInt* src, size_t nbElts, DstInt minValue) \
    {                                                                       \
        if (minValue == 0 && sizeof(SrcInt) == sizeof(DstInt)) {            \
            memcpy(dst, src, sizeof(SrcInt) * nbElts);                      \
            return;                                                         \
        }                                                                   \
        for (size_t i = rangeUnpackSimd_##SrcInt##_##DstInt(                \
                     dst, src, nbElts, minValue);                           \
             i < nbElts;                                                    \
             i++) {                                                         \
            dst[i] = (DstInt)((DstInt)src[i] + minValue);                   \
        }                                                                   \
    }

GEN_RANGE_UNPACK(uint64_t, uint64_t)
GEN_RANGE_UNPACK(uint64_t, uint32_t)
GEN_RANGE_UNPACK(uint64_t, uint16_t)
GEN_RANGE_UNPACK(uint64_t, uint8_t)
GEN_RANGE_UNPACK(uint32_t, uint32_t)
GEN_RANGE_UNPACK(uint32_t, uint16_t)
GEN_RANGE_UNPACK(uint32_t, uint8_t)
GEN_RANGE_UNPACK(uint16_t, uint16_t)
GEN_RANGE_UNPACK(uint16_t, uint8_t)
GEN_RANGE_UNPACK(uint8_t, uint8_t)

#undef GEN_RANGE_UNPACK

void rangePackDecode(
        void* dst,
        size_t dstWidth,
        const void* src,
        size_t srcWidth,
        size_t nbElts,
        size_t dstMinValue)
{
    ZL_ASSERT_LE(srcWidth, dstWidth);

    // Use a macro to define the different cases
#define RANGE_PACK_DECODE_CASE(DstInt, SrcInt)                      \
    if (srcWidth == sizeof(SrcInt) && dstWidth == sizeof(DstInt)) { \
        rangeUnpack_##SrcInt##_##DstInt(                            \
                (DstInt*)dst,                                       \
                (const SrcInt*)src,                                 \
                nbElts,                                             \
                (DstInt)dstMinValue);                               \
        return;                                                     \
    }
    RANGE_PACK_DECODE_CASE(uint64_t, uint64_t)
    RANGE_PACK_DECODE_CASE(uint64_t, uint32_t)
    RANGE_PACK_DECODE_CASE(uint64_t, uint16_t)
    RANGE_PACK_DECODE_CASE(uint64_t, uint8_t)
    RANGE_PACK_DECODE_CASE(uint32_t, uint32_t)
    RANGE_PACK_DECODE_CASE(uint32_t, uint16_t)
    RANGE_PACK_DECODE_CASE(uint32_t, uint8_t)
    RANGE_PACK_DECODE_CASE(uint16_t, uint16_t)
    RANGE_PACK_DECODE_CASE(uint16_t, uint8_t)
    RANGE_PACK_DECODE_CASE(uint8_t, uint8_t)
#undef RANGE_PACK_DECODE_CASE
    ZL_ASSERT_FAIL(
            "Executing rangePackDecode with mismatched widths - %zu bytes to %zu bytes",
            srcWidth,
            dstWidth);
}
