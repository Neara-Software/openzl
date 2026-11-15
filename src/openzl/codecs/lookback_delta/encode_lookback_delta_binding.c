// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/lookback_delta/encode_lookback_delta_binding.h"

#include <stdint.h>
#include <string.h>

#include "openzl/codecs/zl_lookback_delta.h"
#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
#include "openzl/shared/bits.h"

enum {
    LBD_MODE_RAW         = 0,
    LBD_MODE_CONSECUTIVE = 1,
    LBD_MODE_LOOKBACK    = 2,

    LBD_PROPOSED_N  = 16,
    LBD_BRUTE_N     = 6,
    LBD_REPEATING_N = 4,
    LBD_DEFAULT_WLOG = 15,
};

static inline uint32_t lbd_hash(uint64_t x, size_t hashMask)
{
    x = (x ^ (x >> 32)) * 11400714819323197441ULL;
    x = x ^ (x >> 32);
    return (uint32_t)(x & (uint64_t)hashMask);
}

static inline unsigned lbd_clz64(uint64_t x)
{
    return x == 0 ? 64u : (unsigned)ZL_clz64(x);
}

static inline unsigned lbd_clz32(uint32_t x)
{
    return x == 0 ? 32u : (unsigned)ZL_clz32(x);
}

#define LBD_DEFINE_WIDTH(W, T, CLZ_FN, EBITS)                                          static void lbd_consec_##W(T* dst, const T* src, size_t n)                         {                                                                                      dst[0] = src[0];                                                                   for (size_t i = 1; i < n; i++) {                                                       dst[i] = (T)(src[i] - src[i - 1]);                                             }                                                                              }                                                                                  static int lbd_encode_##W(                                                                 T* resDst,                                                                         uint16_t* lbDst,                                                                   const T* src,                                                                      size_t n,                                                                          size_t windowN,                                                                    size_t hashTableN,                                                                 uint32_t* idxHashTable,                                                            uint32_t* lbCounts,                                                                uint8_t* lbGoodness,                                                               int forceLookback)                                                         {                                                                                      size_t const hashMask = hashTableN - 1;                                            size_t const nLb      = windowN < n ? windowN : n;                                 for (size_t k = 0; k < nLb; k++) {                                                     lbCounts[k]   = 1;                                                                 lbGoodness[k] = 1;                                                             }                                                                                  memset(idxHashTable, 0, 2 * hashTableN * sizeof(uint32_t));                        size_t proposed[LBD_PROPOSED_N];                                                   for (size_t k = 0; k < LBD_PROPOSED_N; k++) {                                          proposed[k] = 1;                                                               }                                                                                                                                                                     resDst[0]                 = src[0];                                                size_t bestLb             = 1;                                                     size_t repeatingIdx       = 0;                                                     size_t nonOneCount        = 0;                                                     uint64_t consecBits       = 0;                                                     uint64_t lbDeltaBits      = 0;                                                     uint64_t lbMetaBitsX4     = 0;                                                                                                                                        for (size_t i = 1; i < n; i++) {                                                       T const l = src[i];                                                                size_t const newBrute = i < LBD_PROPOSED_N ? i : LBD_PROPOSED_N;                   proposed[newBrute - 1] = newBrute;                                                                                                                                    uint64_t const lu64 = (uint64_t)l;                                                 size_t pIdx         = LBD_BRUTE_N + LBD_REPEATING_N;                               for (int c = 0; c < 2; c++) {                                                          uint64_t const bucket = (c == 0) ? lu64 : (lu64 >> 8);                             size_t const off      = (size_t)c * hashTableN;                                    uint32_t const h0     = lbd_hash(bucket - 1ULL, hashMask);                         uint32_t const h1     = lbd_hash(bucket, hashMask);                                uint32_t const h2     = lbd_hash(bucket + 1ULL, hashMask);                         uint32_t const hs[3]  = { h0, h1, h2 };                                            for (int b = 0; b < 3; b++) {                                                          size_t const d = i - (size_t)idxHashTable[off + hs[b]];                            proposed[pIdx] = d <= windowN ? d : (pIdx < i ? pIdx : i);                         pIdx++;                                                                        }                                                                                  idxHashTable[off + h1] = (uint32_t)i;                                          }                                                                                                                                                                     unsigned bestGoodness     = 0;                                                     unsigned bestDeltaGoodness = 0;                                                    size_t newBestLb          = 1;                                                     for (size_t k = 0; k < LBD_PROPOSED_N; k++) {                                          size_t const cand = proposed[k];                                                   T const other     = src[i - cand];                                                 T const dFwd      = (T)(l - other);                                                T const dRev      = (T)(other - l);                                                T const dAbs      = dFwd < dRev ? dFwd : dRev;                                     unsigned const dg = CLZ_FN(dAbs);                                                  unsigned const g  = (unsigned)lbGoodness[cand - 1] + dg;                           if (g > bestGoodness) {                                                                bestGoodness      = g;                                                             bestDeltaGoodness = dg;                                                            newBestLb         = cand;                                                      }                                                                              }                                                                                                                                                                     T const prev1     = src[i - 1];                                                    T const c1Fwd     = (T)(l - prev1);                                                T const c1Rev     = (T)(prev1 - l);                                                T const c1Abs     = c1Fwd < c1Rev ? c1Fwd : c1Rev;                                 unsigned const c1g = CLZ_FN(c1Abs);                                                consecBits += (uint64_t)((EBITS - c1g) + 1u);                                      lbDeltaBits += (uint64_t)((EBITS - bestDeltaGoodness) + 1u);                       lbMetaBitsX4 += (newBestLb == bestLb) ? 1ULL : 24ULL;                              if (newBestLb != 1) {                                                                  nonOneCount++;                                                                 }                                                                                  if (newBestLb != bestLb) {                                                             repeatingIdx++;                                                                }                                                                                  proposed[LBD_BRUTE_N + (repeatingIdx % LBD_REPEATING_N)] = newBestLb;              bestLb                                                   = newBestLb;                                                                                                 resDst[i]     = (T)(l - src[i - newBestLb]);                                       lbDst[i - 1]  = (uint16_t)(newBestLb - 1);                                         uint32_t const cnt = ++lbCounts[newBestLb - 1];                                    if ((cnt & (cnt - 1u)) == 0u) {                                                        lbGoodness[newBestLb - 1]++;                                                   }                                                                              }                                                                                                                                                                     if (!forceLookback                                                                     && (nonOneCount == 0                                                                   || (lbDeltaBits * 4ULL + lbMetaBitsX4) * 100ULL                                            >= consecBits * 4ULL * 98ULL)) {                                       lbd_consec_##W(resDst, src, n);                                                    return LBD_MODE_CONSECUTIVE;                                                   }                                                                                  return LBD_MODE_LOOKBACK;                                                      }

static inline unsigned lbd_clz8(uint8_t x)
{
    return x == 0 ? 8u : ((unsigned)ZL_clz32((uint32_t)x) - 24u);
}

static inline unsigned lbd_clz16(uint16_t x)
{
    return x == 0 ? 16u : ((unsigned)ZL_clz32((uint32_t)x) - 16u);
}

LBD_DEFINE_WIDTH(1, uint8_t, lbd_clz8, 8u)
LBD_DEFINE_WIDTH(2, uint16_t, lbd_clz16, 16u)
LBD_DEFINE_WIDTH(4, uint32_t, lbd_clz32, 32u)
LBD_DEFINE_WIDTH(8, uint64_t, lbd_clz64, 64u)

ZL_Report EI_lookback_delta(
        ZL_Encoder* eictx,
        const ZL_Input* ins[],
        size_t nbIns)
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(eictx);
    ZL_ASSERT_EQ(nbIns, 1);
    const ZL_Input* const in = ins[0];
    ZL_ASSERT_EQ(ZL_Input_type(in), ZL_Type_numeric);

    size_t const eltWidth = ZL_Input_eltWidth(in);
    size_t const n        = ZL_Input_numElts(in);
    ZL_ERR_IF_NOT(
            eltWidth == 1 || eltWidth == 2 || eltWidth == 4 || eltWidth == 8,
            node_invalid_input);
    ZL_ERR_IF_GT(n, (size_t)UINT32_MAX, node_invalid_input);

    int windowLog = LBD_DEFAULT_WLOG;
    ZL_IntParam const wlParam =
            ZL_Encoder_getLocalIntParam(eictx, ZL_LOOKBACK_DELTA_WINDOW_LOG_PID);
    if (wlParam.paramId == ZL_LOOKBACK_DELTA_WINDOW_LOG_PID) {
        ZL_ERR_IF(
                wlParam.paramValue < 4 || wlParam.paramValue > 16,
                nodeParameter_invalidValue);
        windowLog = wlParam.paramValue;
    }

    int forceLookback = 0;
    ZL_IntParam const forceParam =
            ZL_Encoder_getLocalIntParam(eictx, ZL_LOOKBACK_DELTA_FORCE_PID);
    if (forceParam.paramId == ZL_LOOKBACK_DELTA_FORCE_PID) {
        forceLookback = forceParam.paramValue != 0;
    }

    ZL_Output* const resOut = ZL_Encoder_createTypedStream(eictx, 0, n, eltWidth);
    ZL_ERR_IF_NULL(resOut, allocation);

    size_t const lbCap = (n > LBD_PROPOSED_N) ? (n - 1) : 0;
    ZL_Output* const lbOut = ZL_Encoder_createTypedStream(eictx, 1, lbCap, 2);
    ZL_ERR_IF_NULL(lbOut, allocation);

    if (n <= 1) {
        if (n == 1) {
            memcpy(ZL_Output_ptr(resOut), ZL_Input_ptr(in), eltWidth);
        }
        uint8_t const hdr = (uint8_t)LBD_MODE_RAW;
        ZL_Encoder_sendCodecHeader(eictx, &hdr, 1);
        ZL_ERR_IF_ERR(ZL_Output_commit(resOut, n));
        ZL_ERR_IF_ERR(ZL_Output_commit(lbOut, 0));
        return ZL_returnSuccess();
    }

    if (n <= LBD_PROPOSED_N) {
        switch (eltWidth) {
            case 1:
                lbd_consec_1(
                        (uint8_t*)ZL_Output_ptr(resOut),
                        (const uint8_t*)ZL_Input_ptr(in),
                        n);
                break;
            case 2:
                lbd_consec_2(
                        (uint16_t*)ZL_Output_ptr(resOut),
                        (const uint16_t*)ZL_Input_ptr(in),
                        n);
                break;
            case 4:
                lbd_consec_4(
                        (uint32_t*)ZL_Output_ptr(resOut),
                        (const uint32_t*)ZL_Input_ptr(in),
                        n);
                break;
            case 8:
                lbd_consec_8(
                        (uint64_t*)ZL_Output_ptr(resOut),
                        (const uint64_t*)ZL_Input_ptr(in),
                        n);
                break;
        }
        uint8_t const hdr = (uint8_t)LBD_MODE_CONSECUTIVE;
        ZL_Encoder_sendCodecHeader(eictx, &hdr, 1);
        ZL_ERR_IF_ERR(ZL_Output_commit(resOut, n));
        ZL_ERR_IF_ERR(ZL_Output_commit(lbOut, 0));
        return ZL_returnSuccess();
    }

    size_t const windowN    = (size_t)1u << windowLog;
    size_t const hashTableN = (size_t)1u << (windowLog + 1);
    size_t const nLb        = windowN < n ? windowN : n;

    uint32_t* const idxHashTable = (uint32_t*)ZL_Encoder_getScratchSpace(
            eictx, 2 * hashTableN * sizeof(uint32_t));
    uint32_t* const lbCounts = (uint32_t*)ZL_Encoder_getScratchSpace(
            eictx, nLb * sizeof(uint32_t));
    uint8_t* const lbGoodness = (uint8_t*)ZL_Encoder_getScratchSpace(
            eictx, nLb * sizeof(uint8_t));
    ZL_ERR_IF(
            idxHashTable == NULL || lbCounts == NULL || lbGoodness == NULL,
            allocation);

    int mode = LBD_MODE_RAW;
    switch (eltWidth) {
        case 1:
            mode = lbd_encode_1(
                    (uint8_t*)ZL_Output_ptr(resOut),
                    (uint16_t*)ZL_Output_ptr(lbOut),
                    (const uint8_t*)ZL_Input_ptr(in),
                    n,
                    windowN,
                    hashTableN,
                    idxHashTable,
                    lbCounts,
                    lbGoodness,
                    forceLookback);
            break;
        case 2:
            mode = lbd_encode_2(
                    (uint16_t*)ZL_Output_ptr(resOut),
                    (uint16_t*)ZL_Output_ptr(lbOut),
                    (const uint16_t*)ZL_Input_ptr(in),
                    n,
                    windowN,
                    hashTableN,
                    idxHashTable,
                    lbCounts,
                    lbGoodness,
                    forceLookback);
            break;
        case 4:
            mode = lbd_encode_4(
                    (uint32_t*)ZL_Output_ptr(resOut),
                    (uint16_t*)ZL_Output_ptr(lbOut),
                    (const uint32_t*)ZL_Input_ptr(in),
                    n,
                    windowN,
                    hashTableN,
                    idxHashTable,
                    lbCounts,
                    lbGoodness,
                    forceLookback);
            break;
        case 8:
            mode = lbd_encode_8(
                    (uint64_t*)ZL_Output_ptr(resOut),
                    (uint16_t*)ZL_Output_ptr(lbOut),
                    (const uint64_t*)ZL_Input_ptr(in),
                    n,
                    windowN,
                    hashTableN,
                    idxHashTable,
                    lbCounts,
                    lbGoodness,
                    forceLookback);
            break;
    }

    uint8_t const hdr = (uint8_t)mode;
    ZL_Encoder_sendCodecHeader(eictx, &hdr, 1);
    ZL_ERR_IF_ERR(ZL_Output_commit(resOut, n));
    ZL_ERR_IF_ERR(ZL_Output_commit(
            lbOut, mode == LBD_MODE_LOOKBACK ? (n - 1) : 0));
    return ZL_returnSuccess();
}
