// Copyright (c) Meta Platforms, Inc. and affiliates.

#include "openzl/codecs/int_mult_split/decode_int_mult_split_binding.h"

#include <stdint.h>
#include <string.h>

#include "openzl/common/assertion.h"
#include "openzl/common/errors_internal.h"
#include "openzl/shared/varint.h"

#define IMS_DEFINE_DECODE(W, T)                                        \
    static void ims_decode_##W(                                        \
            T* restrict dst,                                           \
            const T* restrict mult,                                    \
            const T* restrict adj,                                     \
            size_t n,                                                  \
            uint64_t base64)                                           \
    {                                                                  \
        T const base = (T)base64;                                      \
        for (size_t i = 0; i < n; i++) {                               \
            dst[i] = (T)((T)(mult[i] * base) + adj[i]);                \
        }                                                              \
    }

IMS_DEFINE_DECODE(8, uint8_t)
IMS_DEFINE_DECODE(16, uint16_t)
IMS_DEFINE_DECODE(32, uint32_t)
IMS_DEFINE_DECODE(64, uint64_t)

ZL_Report DI_int_mult_split(ZL_Decoder* dictx, const ZL_Input* ins[])
{
    ZL_RESULT_DECLARE_SCOPE_REPORT(dictx);
    const ZL_Input* const multIn = ins[0];
    const ZL_Input* const adjIn  = ins[1];
    ZL_ASSERT_EQ(ZL_Input_type(multIn), ZL_Type_numeric);
    ZL_ASSERT_EQ(ZL_Input_type(adjIn), ZL_Type_numeric);

    ZL_RBuffer const hdr = ZL_Decoder_getCodecHeader(dictx);
    ZL_ERR_IF_LT(hdr.size, 2, corruption);
    const uint8_t* ptr       = (const uint8_t*)hdr.start;
    const uint8_t* const end = ptr + hdr.size;

    uint8_t const flags  = *ptr++;
    unsigned const wLog2 = flags & 3u;
    ZL_ERR_IF_NE(flags >> 3, 0, corruption);

    size_t const eltWidth = (size_t)1u << wLog2;
    ZL_ERR_IF_NE(ZL_Input_eltWidth(multIn), eltWidth, corruption);
    ZL_ERR_IF_NE(ZL_Input_eltWidth(adjIn), eltWidth, corruption);

    ZL_TRY_LET(uint64_t, baseMinus1, ZL_varintDecode(&ptr, end));
    ZL_ERR_IF_NE(ptr, end, corruption);
    ZL_ERR_IF_EQ(baseMinus1, UINT64_MAX, corruption);
    uint64_t const base = baseMinus1 + 1u;

    size_t const n   = ZL_Input_numElts(multIn);
    size_t const adjN = ZL_Input_numElts(adjIn);

    ZL_Output* const out = ZL_Decoder_create1OutStream(dictx, n, eltWidth);
    ZL_ERR_IF_NULL(out, allocation);

    if (base == 1u) {
        ZL_ERR_IF_NE(adjN, 0, corruption);
        if (n > 0) {
            memcpy(ZL_Output_ptr(out), ZL_Input_ptr(multIn), n * eltWidth);
        }
        ZL_ERR_IF_ERR(ZL_Output_commit(out, n));
        return ZL_returnValue(1);
    }

    ZL_ERR_IF_NE(adjN, n, corruption);
    if (n > 0) {
        void* const dstPtr       = ZL_Output_ptr(out);
        const void* const multPtr = ZL_Input_ptr(multIn);
        const void* const adjPtr  = ZL_Input_ptr(adjIn);
        switch (eltWidth) {
            case 1:
                ims_decode_8(
                        (uint8_t*)dstPtr,
                        (const uint8_t*)multPtr,
                        (const uint8_t*)adjPtr,
                        n,
                        base);
                break;
            case 2:
                ims_decode_16(
                        (uint16_t*)dstPtr,
                        (const uint16_t*)multPtr,
                        (const uint16_t*)adjPtr,
                        n,
                        base);
                break;
            case 4:
                ims_decode_32(
                        (uint32_t*)dstPtr,
                        (const uint32_t*)multPtr,
                        (const uint32_t*)adjPtr,
                        n,
                        base);
                break;
            case 8:
                ims_decode_64(
                        (uint64_t*)dstPtr,
                        (const uint64_t*)multPtr,
                        (const uint64_t*)adjPtr,
                        n,
                        base);
                break;
            default:
                ZL_ERR(corruption);
        }
    }

    ZL_ERR_IF_ERR(ZL_Output_commit(out, n));
    return ZL_returnValue(1);
}
