// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/ctx_split/decode_ctx_split_binding.h"

#include <stdbool.h>
#include <stdint.h>
#include <string.h>

#include "openzl/codecs/ctx_split/common_ctx_split.h"
#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
#include "openzl/decompress/dctx2.h"
#include "openzl/decompress/dictx.h"
#include "openzl/shared/overflow.h"
#include "openzl/shared/varint.h"
#include "openzl/zl_output.h"

static ZL_Output* ctx_createOutStream(
        ZL_Decoder* dictx,
        size_t eltsCapacity,
        size_t eltWidth,
        ZL_Type st)
{
    size_t dstBytes = 0;
    if (eltWidth == 0 || ZL_overflowMulST(eltsCapacity, eltWidth, &dstBytes)) {
        return NULL;
    }
    if (dictx == NULL || dictx->nbRegens < 1) {
        return NULL;
    }
    return ZL_codemodDataAsOutput(DCTX_newStream(
            dictx->dctx, dictx->regensID[0], st, eltWidth, eltsCapacity));
}

/*
 * Reconstruction is an order-1 state machine: the class of the previous value
 * selects the stream that provides the next value. That dependency is serial,
 * except inside a "self run": while consecutive values stay in class c, they
 * are all read from stream c. Every class is a contiguous value range
 * [lo, lo + span], so a self run is the longest prefix of stream c that stays
 * in that range. The run kernels below find and copy that prefix with SIMD.
 *
 * The scalar state machine enters the run kernel only after
 * CTX_RUN_STREAK consecutive values of one class, so data with short runs
 * keeps the cheap scalar loop.
 */
#define CTX_RUN_STREAK 8

#if defined(__AVX2__)
#    include <immintrin.h>
#    define CTX_HAS_RUN_SIMD 1
#    define CTX_DEFINE_RUN_SIMD(W, T)                                         \
        static size_t ctx_copyRunSimd_##W(                                    \
                T* dst, const T* src, size_t maxN, T lo, T span)              \
        {                                                                     \
            size_t const step  = 32 / sizeof(T);                              \
            __m256i const vlo  = _mm256_set1_epi##W((int##W##_t)lo);          \
            __m256i const vspn = _mm256_set1_epi##W((int##W##_t)span);        \
            size_t n           = 0;                                           \
            while (maxN - n >= step) {                                        \
                __m256i const v =                                             \
                        _mm256_loadu_si256((const __m256i*)(src + n));        \
                _mm256_storeu_si256((__m256i*)(dst + n), v);                  \
                __m256i const d  = _mm256_sub_epi##W(v, vlo);                 \
                __m256i const ok = _mm256_cmpeq_epi##W(                       \
                        _mm256_min_epu##W(d, vspn), d);                       \
                uint32_t const bad = ~(uint32_t)_mm256_movemask_epi8(ok);     \
                if (bad != 0) {                                               \
                    return n + (size_t)ZL_ctz32(bad) / sizeof(T);             \
                }                                                             \
                n += step;                                                    \
            }                                                                 \
            return n;                                                         \
        }
CTX_DEFINE_RUN_SIMD(8, uint8_t)
CTX_DEFINE_RUN_SIMD(16, uint16_t)
CTX_DEFINE_RUN_SIMD(32, uint32_t)
#elif defined(__aarch64__)
#    include <arm_neon.h>
#    define CTX_HAS_RUN_SIMD 1
#    define CTX_NEON_AS_U16_8(x) vreinterpretq_u16_u8(x)
#    define CTX_NEON_AS_U16_16(x) (x)
#    define CTX_NEON_AS_U16_32(x) vreinterpretq_u16_u32(x)
#    define CTX_DEFINE_RUN_SIMD(W, T, VT)                                     \
        static size_t ctx_copyRunSimd_##W(                                    \
                T* dst, const T* src, size_t maxN, T lo, T span)              \
        {                                                                     \
            size_t const step = 16 / sizeof(T);                               \
            VT const vlo      = vdupq_n_u##W(lo);                             \
            VT const vspn     = vdupq_n_u##W(span);                           \
            size_t n          = 0;                                            \
            while (maxN - n >= step) {                                        \
                VT const v = vld1q_u##W(src + n);                             \
                vst1q_u##W(dst + n, v);                                       \
                VT const ok = vcleq_u##W(vsubq_u##W(v, vlo), vspn);           \
                /* 4 mask bits per byte */                                    \
                uint64_t const bad = ~vget_lane_u64(                          \
                        vreinterpret_u64_u8(vshrn_n_u16(                      \
                                CTX_NEON_AS_U16_##W(ok), 4)),                 \
                        0);                                                   \
                if (bad != 0) {                                               \
                    return n + (size_t)ZL_ctz64(bad) / (4 * sizeof(T));       \
                }                                                             \
                n += step;                                                    \
            }                                                                 \
            return n;                                                         \
        }
CTX_DEFINE_RUN_SIMD(8, uint8_t, uint8x16_t)
CTX_DEFINE_RUN_SIMD(16, uint16_t, uint16x8_t)
CTX_DEFINE_RUN_SIMD(32, uint32_t, uint32x4_t)
#elif defined(__wasm_simd128__)
#    include <wasm_simd128.h>
#    define CTX_HAS_RUN_SIMD 1
#    define CTX_DEFINE_RUN_SIMD(W, N, T)                                      \
        static size_t ctx_copyRunSimd_##W(                                    \
                T* dst, const T* src, size_t maxN, T lo, T span)              \
        {                                                                     \
            size_t const step = 16 / sizeof(T);                               \
            v128_t const vlo  = wasm_i##W##x##N##_splat((int##W##_t)lo);      \
            v128_t const vspn = wasm_i##W##x##N##_splat((int##W##_t)span);    \
            size_t n          = 0;                                            \
            while (maxN - n >= step) {                                        \
                v128_t const v = wasm_v128_load(src + n);                     \
                wasm_v128_store(dst + n, v);                                  \
                v128_t const ok = wasm_u##W##x##N##_le(                       \
                        wasm_i##W##x##N##_sub(v, vlo), vspn);                 \
                uint32_t const bad =                                          \
                        ~(uint32_t)wasm_i8x16_bitmask(ok) & 0xFFFFu;          \
                if (bad != 0) {                                               \
                    return n + (size_t)ZL_ctz32(bad) / sizeof(T);             \
                }                                                             \
                n += step;                                                    \
            }                                                                 \
            return n;                                                         \
        }
CTX_DEFINE_RUN_SIMD(8, 16, uint8_t)
CTX_DEFINE_RUN_SIMD(16, 8, uint16_t)
CTX_DEFINE_RUN_SIMD(32, 4, uint32_t)
#else
#    define CTX_HAS_RUN_SIMD 0
#endif

#if CTX_HAS_RUN_SIMD
#    define CTX_RUN_SIMD_8(dst, src, maxN, lo, span) \
        ctx_copyRunSimd_8(dst, src, maxN, lo, span)
#    define CTX_RUN_SIMD_16(dst, src, maxN, lo, span) \
        ctx_copyRunSimd_16(dst, src, maxN, lo, span)
#    define CTX_RUN_SIMD_32(dst, src, maxN, lo, span) \
        ctx_copyRunSimd_32(dst, src, maxN, lo, span)
#else
#    define CTX_RUN_SIMD_8(dst, src, maxN, lo, span) 0
#    define CTX_RUN_SIMD_16(dst, src, maxN, lo, span) 0
#    define CTX_RUN_SIMD_32(dst, src, maxN, lo, span) 0
#endif
#define CTX_RUN_SIMD_64(dst, src, maxN, lo, span) 0

/*
 * Copies the longest prefix of src (at most maxN elements) whose values are in
 * [lo, lo + span], and returns its length. The SIMD kernels store whole
 * vectors, so they may write past the returned length, but never past
 * dst + maxN.
 */
#define CTX_DEFINE_COPY_RUN(W, T)                                              \
    static size_t ctx_copyRun_##W(                                             \
            T* dst, const T* src, size_t maxN, T lo, T span)                   \
    {                                                                          \
        size_t n = CTX_RUN_SIMD_##W(dst, src, maxN, lo, span);                 \
        while (n < maxN) {                                                     \
            T const v = src[n];                                                \
            if ((T)(v - lo) > span) {                                          \
                break;                                                         \
            }                                                                  \
            dst[n++] = v;                                                      \
        }                                                                      \
        return n;                                                              \
    }

CTX_DEFINE_COPY_RUN(8, uint8_t)
CTX_DEFINE_COPY_RUN(16, uint16_t)
CTX_DEFINE_COPY_RUN(32, uint32_t)
CTX_DEFINE_COPY_RUN(64, uint64_t)

typedef struct {
    unsigned mode;
    unsigned shift;
    uint64_t minVal;
    uint32_t maxCls;
    /* Value range of each class: [lo, lo + span]. */
    uint64_t lo[CTX_SPLIT_MAX_K];
    uint64_t span[CTX_SPLIT_MAX_K];
    /* 0 when the class holds no value of the element type. */
    uint8_t nonEmpty[CTX_SPLIT_MAX_K];
} CtxClassifier;

/* Computes minVal + (c << shift). Returns 0 when it does not fit in 64 bits. */
static int
ctx_linearBound(uint64_t minVal, unsigned shift, uint32_t c, uint64_t* bound)
{
    if ((uint64_t)c > (UINT64_MAX >> shift)) {
        return 0;
    }
    uint64_t const scaled = (uint64_t)c << shift;
    if (scaled > UINT64_MAX - minVal) {
        return 0;
    }
    *bound = minVal + scaled;
    return 1;
}

static void ctx_initClassifier(
        CtxClassifier* cls,
        unsigned mode,
        unsigned shift,
        uint64_t minVal,
        uint32_t maxCls,
        size_t eltWidth)
{
    uint64_t const typeMax =
            eltWidth >= 8 ? UINT64_MAX : (((uint64_t)1 << (8 * eltWidth)) - 1);
    memset(cls, 0, sizeof(*cls));
    cls->mode   = mode;
    cls->shift  = shift;
    cls->minVal = minVal;
    cls->maxCls = maxCls;
    for (uint32_t c = 0; c <= maxCls; c++) {
        uint64_t lo = 0;
        uint64_t hi = UINT64_MAX;
        int ok      = 1;
        if (mode == CTX_SPLIT_MODE_LINEAR) {
            if (c > 0) {
                ok = ctx_linearBound(minVal, shift, c, &lo);
            }
            uint64_t next;
            if (c < maxCls && ctx_linearBound(minVal, shift, c + 1, &next)) {
                hi = next - 1;
            }
        } else {
            if (c > 0) {
                lo = (uint64_t)1 << (c - 1);
            }
            if (c < maxCls) {
                hi = ((uint64_t)1 << c) - 1;
            }
        }
        if (hi > typeMax) {
            hi = typeMax;
        }
        if (ok && lo <= hi) {
            cls->lo[c]       = lo;
            cls->span[c]     = hi - lo;
            cls->nonEmpty[c] = 1;
        }
    }
}

ZL_FORCE_INLINE uint32_t ctx_classify(const CtxClassifier* cls, uint64_t prev)
{
    return cls->mode == CTX_SPLIT_MODE_LINEAR
            ? ctx_cls_mode0(prev, cls->minVal, cls->shift, cls->maxCls)
            : ctx_cls_mode1(prev, cls->maxCls);
}

/*
 * CLS_EXPR computes the class of `val`. Returns 0 if a stream runs out before
 * the output is complete.
 */
#define CTX_GATHER_BODY(W, T, CLS_EXPR)                                        \
    const T* cur[CTX_SPLIT_MAX_K];                                             \
    const T* end[CTX_SPLIT_MAX_K];                                             \
    for (size_t k = 0; k < CTX_SPLIT_MAX_K; k++) {                             \
        cur[k] = (const T*)srcPtrs[k];                                         \
        end[k] = cur[k] + srcCounts[k];                                        \
    }                                                                          \
    uint32_t c = (CLS_EXPR(0));                                                \
    T* out     = dst;                                                          \
    /* sum(end - cur) == dstEnd - out holds on every iteration */              \
    T* const dstEnd = dst + totalElts;                                         \
    while (out < dstEnd) {                                                     \
        const T* p          = cur[c];                                          \
        const T* const pEnd = end[c];                                          \
        if (ZL_UNLIKELY(p == pEnd)) {                                          \
            return 0;                                                          \
        }                                                                      \
        size_t const avail = (size_t)(pEnd - p);                               \
        const T* const pStreak =                                               \
                p + (avail < CTX_RUN_STREAK ? avail : CTX_RUN_STREAK);         \
        uint32_t nc;                                                           \
        /* Self transitions stay in this loop: the cursor is in a register */  \
        /* and the loop branch hides the classification latency. */            \
        do {                                                                   \
            T const val = *p++;                                                \
            *out++      = val;                                                 \
            nc          = (CLS_EXPR(val));                                     \
        } while (nc == c && p != pStreak);                                     \
        if (nc == c && p != pEnd && cls->nonEmpty[c]) {                        \
            size_t const n = ctx_copyRun_##W(                                  \
                    out,                                                       \
                    p,                                                         \
                    (size_t)(pEnd - p),                                        \
                    (T)cls->lo[c],                                             \
                    (T)cls->span[c]);                                          \
            p += n;                                                            \
            out += n;                                                          \
        }                                                                      \
        cur[c] = p;                                                            \
        c      = nc;                                                           \
    }                                                                          \
    return 1;

#define CTX_CLS_MODE0(v) ctx_cls_mode0((uint64_t)(v), minVal, shift, maxCls)
#define CTX_CLS_MODE1(v) ctx_cls_mode1((uint64_t)(v), maxCls)
#define CTX_CLS_LUT(v) lut[(uint8_t)(v)]

#define CTX_DEFINE_GATHER(W, T)                                                \
    static int ctx_gather_mode0_##W(                                           \
            T* restrict dst,                                                   \
            size_t totalElts,                                                  \
            const void* const srcPtrs[CTX_SPLIT_MAX_K],                        \
            const size_t srcCounts[CTX_SPLIT_MAX_K],                           \
            const CtxClassifier* cls)                                          \
    {                                                                          \
        uint64_t const minVal = cls->minVal;                                   \
        unsigned const shift  = cls->shift;                                    \
        uint32_t const maxCls = cls->maxCls;                                   \
        CTX_GATHER_BODY(W, T, CTX_CLS_MODE0)                                   \
    }                                                                          \
                                                                               \
    static int ctx_gather_mode1_##W(                                           \
            T* restrict dst,                                                   \
            size_t totalElts,                                                  \
            const void* const srcPtrs[CTX_SPLIT_MAX_K],                        \
            const size_t srcCounts[CTX_SPLIT_MAX_K],                           \
            const CtxClassifier* cls)                                          \
    {                                                                          \
        uint32_t const maxCls = cls->maxCls;                                   \
        CTX_GATHER_BODY(W, T, CTX_CLS_MODE1)                                   \
    }

CTX_DEFINE_GATHER(16, uint16_t)
CTX_DEFINE_GATHER(32, uint32_t)
CTX_DEFINE_GATHER(64, uint64_t)

/*
 * 1-byte elements.
 *
 * The plain state machine has a long serial dependency per element: class ->
 * load stream cursor -> load value -> classify. This kernel shortens it by
 * keeping the next 3 bytes of every stream in registers: byte k of head0 /
 * head1 / head2 is the 1st / 2nd / 3rd unread byte of stream k. Selecting the
 * next value is then a register shift, and the memory load that refills the
 * queue of the consumed stream is off the critical path.
 *
 * The steps do no bounds check. Instead, every CTX_CHECK_STEPS steps
 * ctx_prepStreams() makes sure that every stream can be read
 * CTX_CHECK_STEPS + CTX_HEAD_DEPTH bytes past its cursor: a stream that is
 * too close to its end is moved to a zero-padded tail buffer, and a stream
 * that was read past its end is reported as corruption.
 */
#define CTX_HEAD_DEPTH 3
#define CTX_RUN_CHECK_STEPS 16
#define CTX_CHECK_STEPS 64
#define CTX_TAIL_MARGIN (CTX_CHECK_STEPS + CTX_HEAD_DEPTH)
#define CTX_TAIL_SIZE (2 * CTX_TAIL_MARGIN)

typedef struct {
    /* base[k] + idx[k] is the cursor of stream k, cnt[k] its size */
    const uint8_t* base[CTX_SPLIT_MAX_K];
    size_t idx[CTX_SPLIT_MAX_K];
    size_t cnt[CTX_SPLIT_MAX_K];
    uint32_t inTail; /* bit k: stream k was moved to tail[k] */
    uint8_t tail[CTX_SPLIT_MAX_K][CTX_TAIL_SIZE];
} CtxStreams8;

/*
 * Returns 0 if a stream was read past its end. Otherwise, on return each
 * stream is readable up to base[k][idx[k] + CTX_TAIL_MARGIN - 1].
 */
static int ctx_prepStreams(CtxStreams8* s)
{
    for (uint32_t k = 0; k < CTX_SPLIT_MAX_K; k++) {
        if (ZL_LIKELY(s->cnt[k] - s->idx[k] >= CTX_TAIL_MARGIN
                      && s->idx[k] <= s->cnt[k])) {
            continue;
        }
        if (s->idx[k] > s->cnt[k]) {
            return 0;
        }
        if ((s->inTail & (1u << k)) == 0) {
            size_t const rem = s->cnt[k] - s->idx[k];
            memset(s->tail[k], 0, CTX_TAIL_SIZE);
            if (rem > 0) {
                memcpy(s->tail[k], s->base[k] + s->idx[k], rem);
            }
            s->base[k] = s->tail[k];
            s->idx[k]  = 0;
            s->cnt[k]  = rem;
            s->inTail |= 1u << k;
        }
    }
    return 1;
}

/* Load element k of an array of 8-byte elements from its bit offset 8 * k. */
#define CTX_LANE_PTR(type, array, c8) ((type*)(void*)((char*)(array) + (c8)))

static int ctx_gather_8(
        uint8_t* restrict dst,
        size_t totalElts,
        const void* const srcPtrs[CTX_SPLIT_MAX_K],
        const size_t srcCounts[CTX_SPLIT_MAX_K],
        const CtxClassifier* cls)
{
    /* lut[v] = 8 * class(v): the bit offset of the class lane */
    uint8_t lut[256];
    for (unsigned v = 0; v < 256; v++) {
        lut[v] = (uint8_t)(8u * ctx_classify(cls, v));
    }

    CtxStreams8 s;
    s.inTail = 0;
    for (size_t k = 0; k < CTX_SPLIT_MAX_K; k++) {
        s.base[k] = (const uint8_t*)srcPtrs[k];
        s.idx[k]  = 0;
        s.cnt[k]  = srcCounts[k];
    }
    if (!ctx_prepStreams(&s)) {
        return 0;
    }

    uint64_t head0 = 0;
    uint64_t head1 = 0;
    uint64_t head2 = 0;
    for (size_t k = 0; k < CTX_SPLIT_MAX_K; k++) {
        head0 |= (uint64_t)s.base[k][0] << (8 * k);
        head1 |= (uint64_t)s.base[k][1] << (8 * k);
        head2 |= (uint64_t)s.base[k][2] << (8 * k);
    }
    size_t c8 = lut[0];
    size_t i  = 0;

#define CTX_STEP_8()                                                           \
    do {                                                                       \
        uint64_t const lane = (uint64_t)0xFF << c8;                            \
        uint8_t const val   = (uint8_t)(head0 >> c8);                          \
        size_t const pos    = ++*CTX_LANE_PTR(size_t, s.idx, c8);              \
        uint64_t const next = (*CTX_LANE_PTR(const uint8_t*, s.base, c8))      \
                [pos + (CTX_HEAD_DEPTH - 1)];                                  \
        dst[i++]            = val;                                             \
        head0               = (head0 & ~lane) | (head1 & lane);                \
        head1               = (head1 & ~lane) | (head2 & lane);                \
        head2               = (head2 & ~lane) | (next << c8);                  \
        c8                  = lut[val];                                        \
    } while (0)

    while (i < totalElts) {
        size_t const left = totalElts - i;
        size_t const stop =
                i + (left < CTX_CHECK_STEPS ? left : CTX_CHECK_STEPS);
        bool tryRun = false;
        while (stop - i >= CTX_RUN_CHECK_STEPS) {
            size_t const c8Start = c8;
            for (size_t j = 0; j < CTX_RUN_CHECK_STEPS / 4; j++) {
                CTX_STEP_8();
                CTX_STEP_8();
                CTX_STEP_8();
                CTX_STEP_8();
            }
            /*
             * Cheap test for a long self run. It only decides whether to try
             * the run kernel, which is exact for any input.
             */
            if (ZL_UNLIKELY(
                        c8 == c8Start
                        && lut[dst[i - CTX_RUN_CHECK_STEPS / 2]] == c8)) {
                tryRun = true;
                break;
            }
        }
        if (!tryRun) {
            while (i < stop) {
                CTX_STEP_8();
            }
        }
        if (!ctx_prepStreams(&s)) {
            return 0;
        }
        size_t const c = c8 >> 3;
        if (tryRun && cls->nonEmpty[c] && s.idx[c] < s.cnt[c]) {
            size_t const avail = s.cnt[c] - s.idx[c];
            size_t const room  = totalElts - i;
            size_t const n     = ctx_copyRun_8(
                    dst + i,
                    s.base[c] + s.idx[c],
                    avail < room ? avail : room,
                    (uint8_t)cls->lo[c],
                    (uint8_t)cls->span[c]);
            if (n > 0) {
                i += n;
                s.idx[c] += n;
                if (!ctx_prepStreams(&s)) {
                    return 0;
                }
                uint64_t const lane    = (uint64_t)0xFF << c8;
                const uint8_t* const p = s.base[c] + s.idx[c];
                head0 = (head0 & ~lane) | ((uint64_t)p[0] << c8);
                head1 = (head1 & ~lane) | ((uint64_t)p[1] << c8);
                head2 = (head2 & ~lane) | ((uint64_t)p[2] << c8);
            }
        }
    }
#undef CTX_STEP_8

    for (size_t k = 0; k < CTX_SPLIT_MAX_K; k++) {
        if (s.idx[k] != s.cnt[k]) {
            return 0;
        }
    }
    return 1;
}

static int ctx_isValidTypeWidth(ZL_Type st, size_t eltWidth)
{
    if (st == ZL_Type_serial) {
        return eltWidth == 1;
    }
    if (st == ZL_Type_numeric) {
        return eltWidth == 1 || eltWidth == 2 || eltWidth == 4 || eltWidth == 8;
    }
    return 0;
}

ZL_Report DI_ctx_split(
        ZL_Decoder* dictx,
        const ZL_Input* compulsorySrcs[],
        size_t nbCompulsorySrcs,
        const ZL_Input* variableSrcs[],
        size_t nbVariableSrcs)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dictx);
    ZL_ASSERT_NN(dictx);
    (void)compulsorySrcs;
    ZL_ERR_IF_NE(nbCompulsorySrcs, 0, corruption);

    ZL_RBuffer const hdr = ZL_Decoder_getCodecHeader(dictx);
    ZL_ERR_IF_LT(hdr.size, 1, corruption);

    const uint8_t* ptr       = (const uint8_t*)hdr.start;
    const uint8_t* const end = ptr + hdr.size;

    uint8_t const flagsByte = *ptr++;
    unsigned const mode     = (unsigned)(flagsByte >> 4);
    unsigned const nCtxLog  = (unsigned)(flagsByte & 0x0Fu);

    ZL_ERR_IF_GT(mode, CTX_SPLIT_MODE_PASSTHROUGH, corruption);
    ZL_ERR_IF_LT(nCtxLog, CTX_SPLIT_MIN_NCTX_LOG, corruption);
    ZL_ERR_IF_GT(nCtxLog, CTX_SPLIT_MAX_NCTX_LOG, corruption);

    if (mode == CTX_SPLIT_MODE_PASSTHROUGH) {
        ZL_ERR_IF_NE(ptr, end, corruption);
        ZL_ERR_IF_NE(nbVariableSrcs, 1, corruption);
        const ZL_Input* const src0 = variableSrcs[0];
        ZL_ERR_IF_NULL(src0, corruption);

        ZL_Type const srcType = ZL_Input_type(src0);
        size_t const eltWidth = ZL_Input_eltWidth(src0);
        size_t const nbElts   = ZL_Input_numElts(src0);
        ZL_ERR_IF_NOT(ctx_isValidTypeWidth(srcType, eltWidth), corruption);

        size_t totalBytes = 0;
        ZL_ERR_IF(ZL_overflowMulST(nbElts, eltWidth, &totalBytes), corruption);

        ZL_Output* const out =
                ctx_createOutStream(dictx, nbElts, eltWidth, srcType);
        ZL_ERR_IF_NULL(out, allocation);
        if (totalBytes > 0) {
            memcpy(ZL_Output_ptr(out), ZL_Input_ptr(src0), totalBytes);
        }
        ZL_ERR_IF_ERR(ZL_Output_commit(out, nbElts));
        return ZL_returnSuccess();
    }

    /* mode == 0 (LINEAR) or mode == 1 (LOG2) */
    ZL_ERR_IF_GE(ptr, end, corruption);
    uint8_t const activeMask = *ptr++;
    uint32_t const K         = 1u << nCtxLog;
    uint32_t const maxCls    = K - 1u;

    ZL_ERR_IF_EQ(activeMask, 0, corruption);
    if (K < 8u) {
        ZL_ERR_IF_NE((unsigned)activeMask >> K, 0u, corruption);
    }

    size_t const expectedStreams = (size_t)ZL_popcount64((uint64_t)activeMask);
    ZL_ERR_IF_NE(nbVariableSrcs, expectedStreams, corruption);

    unsigned shift  = 0;
    uint64_t minVal = 0;
    if (mode == CTX_SPLIT_MODE_LINEAR) {
        ZL_ERR_IF_LT((size_t)(end - ptr), 2u, corruption);
        shift = (unsigned)(*ptr++);
        ZL_TRY_LET(uint64_t, decodedMin, ZL_varintDecode(&ptr, end));
        minVal = decodedMin;
    }
    ZL_ERR_IF_NE(ptr, end, corruption);

    const ZL_Input* const firstSrc = variableSrcs[0];
    ZL_ERR_IF_NULL(firstSrc, corruption);
    ZL_Type const srcType = ZL_Input_type(firstSrc);
    size_t const eltWidth = ZL_Input_eltWidth(firstSrc);
    ZL_ERR_IF_NOT(ctx_isValidTypeWidth(srcType, eltWidth), corruption);

    if (mode == CTX_SPLIT_MODE_LINEAR) {
        ZL_ERR_IF_GE(shift, (unsigned)(eltWidth * 8u), corruption);
        if (eltWidth < 8u) {
            ZL_ERR_IF_NE(minVal >> (eltWidth * 8u), 0u, corruption);
        }
    }

    const void* srcPtrs[CTX_SPLIT_MAX_K] = { NULL };
    size_t srcCounts[CTX_SPLIT_MAX_K]    = { 0 };
    size_t totalElts                     = 0;
    size_t sIdx                          = 0;

    for (uint32_t c = 0; c < K; c++) {
        if ((activeMask & (1u << c)) != 0) {
            const ZL_Input* const s = variableSrcs[sIdx++];
            ZL_ERR_IF_NULL(s, corruption);
            ZL_ERR_IF_NE(ZL_Input_type(s), srcType, corruption);
            ZL_ERR_IF_NE(ZL_Input_eltWidth(s), eltWidth, corruption);
            size_t const cnt = ZL_Input_numElts(s);
            ZL_ERR_IF_EQ(cnt, 0, corruption);
            ZL_ERR_IF(ZL_overflowAddST(totalElts, cnt, &totalElts), corruption);
            srcPtrs[c]   = ZL_Input_ptr(s);
            srcCounts[c] = cnt;
        }
    }
    ZL_ASSERT_EQ(sIdx, expectedStreams);

    size_t totalBytes = 0;
    ZL_ERR_IF(ZL_overflowMulST(totalElts, eltWidth, &totalBytes), corruption);

    ZL_Output* const out =
            ctx_createOutStream(dictx, totalElts, eltWidth, srcType);
    ZL_ERR_IF_NULL(out, allocation);

    CtxClassifier cls;
    ctx_initClassifier(&cls, mode, shift, minVal, maxCls, eltWidth);

    void* const dst = ZL_Output_ptr(out);
    int ok          = 0;
    bool const lin  = mode == CTX_SPLIT_MODE_LINEAR;
    switch (eltWidth) {
        case 1:
            ok = ctx_gather_8(
                    (uint8_t*)dst, totalElts, srcPtrs, srcCounts, &cls);
            break;
        case 2:
            ok = lin ? ctx_gather_mode0_16(
                               (uint16_t*)dst,
                               totalElts,
                               srcPtrs,
                               srcCounts,
                               &cls)
                     : ctx_gather_mode1_16(
                               (uint16_t*)dst,
                               totalElts,
                               srcPtrs,
                               srcCounts,
                               &cls);
            break;
        case 4:
            ok = lin ? ctx_gather_mode0_32(
                               (uint32_t*)dst,
                               totalElts,
                               srcPtrs,
                               srcCounts,
                               &cls)
                     : ctx_gather_mode1_32(
                               (uint32_t*)dst,
                               totalElts,
                               srcPtrs,
                               srcCounts,
                               &cls);
            break;
        case 8:
            ok = lin ? ctx_gather_mode0_64(
                               (uint64_t*)dst,
                               totalElts,
                               srcPtrs,
                               srcCounts,
                               &cls)
                     : ctx_gather_mode1_64(
                               (uint64_t*)dst,
                               totalElts,
                               srcPtrs,
                               srcCounts,
                               &cls);
            break;
    }
    ZL_ERR_IF_NOT(ok, corruption);

    ZL_ERR_IF_ERR(ZL_Output_commit(out, totalElts));
    return ZL_returnSuccess();
}
