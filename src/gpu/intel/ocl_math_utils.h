/*******************************************************************************
* Copyright 2019-2025 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef GPU_INTEL_OCL_MATH_UTILS_H
#define GPU_INTEL_OCL_MATH_UTILS_H

#include "gpu/intel/ocl_custom_types.h"
#include "gpu/intel/ocl_utils.h"

// Due to JIT compilation and a lack of bitwise operations in implementations,
// this warning has a high false-positive rate.
#pragma clang diagnostic ignored "-Wconstant-logical-operand"

// Due to JIT compilation this feature is generally desired.
#pragma clang diagnostic ignored "-Wtautological-compare"

int __attribute__((overloadable)) div_up(int a, unsigned int b) {
    return (a / b) + (a % b != 0);
}

int __attribute__((overloadable)) div_up(unsigned int a, unsigned int b) {
    return (a / b) + (a % b != 0);
}

long __attribute__((overloadable)) div_up(long a, unsigned int b) {
    return (a / b) + (a % b != 0);
}

int __attribute__((overloadable)) rnd_up(int a, unsigned int b) {
    return div_up(a, b) * b;
}

int __attribute__((overloadable)) rnd_up(unsigned int a, unsigned int b) {
    return div_up(a, b) * b;
}

int __attribute__((overloadable)) rnd_up(long a, unsigned int b) {
    return div_up(a, b) * b;
}

int __attribute__((overloadable)) rnd_down(int a, unsigned int b) {
    return (a / b) * b;
}

int __attribute__((overloadable)) rnd_down(unsigned int a, unsigned int b) {
    return (a / b) * b;
}

int __attribute__((overloadable)) rnd_down(long a, unsigned int b) {
    return (a / b) * b;
}

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#if DT_BF8 || SRC_DT_BF8 || WEI_DT_BF8 || DST_DT_BF8 || BIA_DT_BF8 || A_DT_BF8 \
        || B_DT_BF8 || C_DT_BF8 || DATA_DT_BF8 || POST_OP_USING_BF8 \
        || SRC_SCALES_DT_BF8 || WEI_SCALES_DT_BF8 || DST_SCALES_DT_BF8
#define MATH_UTILS_DECLARE_BF8 1
#endif

#if DT_HF8 || SRC_DT_HF8 || WEI_DT_HF8 || DST_DT_HF8 || BIA_DT_HF8 || A_DT_HF8 \
        || A_DT_HF8 || B_DT_HF8 || C_DT_HF8 || DATA_DT_HF8 \
        || POST_OP_USING_HF8 || SRC_SCALES_DT_HF8 || WEI_SCALES_DT_HF8 \
        || DST_SCALES_DT_HF8
#define MATH_UTILS_DECLARE_HF8 1
#endif

#if DT_F4_E2M1 || SRC_DT_F4_E2M1 || WEI_DT_F4_E2M1 || DST_DT_F4_E2M1 \
        || BIA_DT_F4_E2M1 || A_DT_F4_E2M1 || A_DT_F4_E2M1 || B_DT_F4_E2M1 \
        || C_DT_F4_E2M1 || DATA_DT_F4_E2M1 || POST_OP_USING_F4_E2M1
#define MATH_UTILS_DECLARE_F4_E2M1 1
#endif

#if DT_F4_E3M0 || SRC_DT_F4_E3M0 || WEI_DT_F4_E3M0 || DST_DT_F4_E3M0 \
        || BIA_DT_F4_E3M0 || A_DT_F4_E3M0 || A_DT_F4_E3M0 || B_DT_F4_E3M0 \
        || C_DT_F4_E3M0 || DATA_DT_F4_E3M0 || POST_OP_USING_F4_E3M0
#define MATH_UTILS_DECLARE_F4_E3M0 1
#endif

#if DT_S4 || SRC_DT_S4 || WEI_DT_S4 || DST_DT_S4 || BIA_DT_S4 || A_DT_S4 \
        || B_DT_S4 || C_DT_S4 || DATA_DT_S4 || WEI_ZP_DT_S4 || SRC_ZP_DT_S4
#define MATH_UTILS_DECLARE_S4 1
#endif

#if DT_U4 || SRC_DT_U4 || WEI_DT_U4 || DST_DT_U4 || BIA_DT_U4 || A_DT_U4 \
        || A_DT_U4 || B_DT_U4 || C_DT_U4 || DATA_DT_U4 || WEI_ZP_DT_U4 \
        || SRC_ZP_DT_U4
#define MATH_UTILS_DECLARE_U4 1
#endif

#if DT_BF16 || SRC_DT_BF16 || WEI_DT_BF16 || DST_DT_BF16 || BIA_DT_BF16 \
        || A_DT_BF16 || B_DT_BF16 || C_DT_BF16 || SUM_DT_BF16 || DATA_DT_BF16 \
        || POST_OP_USING_BF16 || SRC_SCALES_DT_BF16 || WEI_SCALES_DT_BF16 \
        || DST_SCALES_DT_BF16
#define MATH_UTILS_DECLARE_BF16 1
#endif

#if DST_SCALES_DT_E8M0
#define MATH_UTILS_DECLARE_E8M0 1
#endif

ulong8 __builtin_IB_simd_block_read_8_global_l(const __global ulong *);
ushort16 __builtin_IB_simd_block_read_16_global_h(const __global ushort *);

void __builtin_IB_simd_block_write_8_global_l(__global ulong *, ulong8);
void __builtin_IB_simd_block_write_16_global_h(__global ushort *, ushort16);
float __attribute__((overloadable)) cvt_e8m0_to_f32(uchar f) {
    if (f == (uchar)0xff) return as_float(0xffc00000);
    uint bits = f << 23;
    return as_float(bits);
}

#if MATH_UTILS_DECLARE_HF8
// Emulation functions for f8_e4m3 <-> f16 conversion.
uchar __attribute__((overloadable)) cvt_hf_to_f8_e4m3(half f) {
    // Here the idea is to add a large constant to the float16_t to force the
    // proper rounding to f8_e4m3 accuracy.
    uchar raw_bits = 0;
    ushort fraw = as_ushort(f);

    // first we extract the sign and make the input positive
    uint s8 = (fraw & 0x8000) >> 8;
    fraw = fraw & 0x7fff;

    // we filter out overlow, nan
    if (fraw >= 0x5f40) {
        raw_bits = s8 | 0x7f;
        return raw_bits;
    }
    // we filter out underflow when f <= 2^-10
    if (fraw <= 0x1400) {
        raw_bits = s8;
        return raw_bits;
    }

    // compute the rounding shifter by taking its exponent + 0x1p7
    // Lucky us, it does not overflow as fraw <= 448.
    ushort a = 0x7c00, b = 0x1c00;
    ushort shifter = (fraw & a) + b;
    // e8 = e16 - e16_bias + e8_bias = e16 - 15 + 7
    // e8 will be denorm if e8 <= 0 or e16 + 7 < 16
    const int exp_threshold = 0x4000; // raw bits of exponent = 16
    ushort is_denorm = shifter < exp_threshold;
    if (is_denorm) shifter = exp_threshold;

    ushort rounded
            = as_ushort((as_half(fraw) + as_half(shifter)) - as_half(shifter));

    int e8 = ((rounded & 0x7c00) >> 10) - 8;
    uchar m8 = (rounded & 0x03ff) >> 7;

    // we need to make the implicit f32 mantissa bit explicit for
    // denorm f8_e4m3
    if (is_denorm) {
        m8 = (m8 | 0x08) >> (-e8 + 1);
        e8 = 0;
    }

    raw_bits = s8 | (e8 << 3) | m8;
    return raw_bits;
}

uchar2 __attribute__((overloadable)) cvt_hf_to_f8_e4m3(half2 f) {
    uchar2 r;
    for (int i = 0; i < 2; i++) {
        r[i] = cvt_hf_to_f8_e4m3(f[i]);
    }
    return r;
}

uchar4 __attribute__((overloadable)) cvt_hf_to_f8_e4m3(half4 f) {
    uchar4 r;
    for (int i = 0; i < 4; i++) {
        r[i] = cvt_hf_to_f8_e4m3(f[i]);
    }
    return r;
}

uchar8 __attribute__((overloadable)) cvt_hf_to_f8_e4m3(half8 f) {
    uchar8 r;
    for (int i = 0; i < 8; i++) {
        r[i] = cvt_hf_to_f8_e4m3(f[i]);
    }
    return r;
}

uchar16 __attribute__((overloadable)) cvt_hf_to_f8_e4m3(half16 f) {
    uchar16 r;
    for (int i = 0; i < 16; i++) {
        r[i] = cvt_hf_to_f8_e4m3(f[i]);
    }
    return r;
}

half __attribute__((overloadable)) cvt_f8_e4m3_to_hf(uchar b) {
    uchar raw_bits_ = b;
    ushort s8 = (raw_bits_ & 0x80) >> 7;
    ushort e8 = (raw_bits_ & 0x78) >> 3;
    ushort m8 = (raw_bits_ & 0x7);
    ushort s16 = s8;
    ushort e16 = e8 + 8; /* 15 - 7 = e16_bias - e8_bias */
    ushort m16 = m8;

    // Need to convert f8_e4m3 denormal into f16 normal.
    if (e8 == 0 && m8 != 0) {
        ushort count = 2;
        count = m8 > 0x1 ? 1 : count;
        count = m8 > 0x3 ? 0 : count;
        e16 -= count;
        m16 = (m16 << (count + 1)) & 0x7;
    } else if (e8 == 0 && m8 == 0) {
        e16 = 0;
    } else if (e8 == 0xf && m8 == 0x7) {
        e16 = 0x1f;
        m16 = 0x4; // Real Indefinite (a qNaN)
    }
    s16 <<= 15;
    e16 <<= 10;
    m16 <<= 7;

    ushort u16 = s16 | e16 | m16;
    return as_half(u16);
}

half2 __attribute__((overloadable)) cvt_f8_e4m3_to_hf(uchar2 b) {
    half2 f;
    for (int i = 0; i < 2; i++) {
        f[i] = cvt_f8_e4m3_to_hf(b[i]);
    }
    return f;
}

half4 __attribute__((overloadable)) cvt_f8_e4m3_to_hf(uchar4 b) {
    half4 f;
    for (int i = 0; i < 4; i++) {
        f[i] = cvt_f8_e4m3_to_hf(b[i]);
    }
    return f;
}

half8 __attribute__((overloadable)) cvt_f8_e4m3_to_hf(uchar8 b) {
    half8 f;
    for (int i = 0; i < 8; i++) {
        f[i] = cvt_f8_e4m3_to_hf(b[i]);
    }
    return f;
}

half16 __attribute__((overloadable)) cvt_f8_e4m3_to_hf(uchar16 b) {
    half16 f;
    for (int i = 0; i < 16; i++) {
        f[i] = cvt_f8_e4m3_to_hf(b[i]);
    }
    return f;
}

#endif
// clang-format on

#if MATH_UTILS_DECLARE_BF8
// Emulation functions for f8_e5m2 <-> f16 conversion.
uchar __attribute__((overloadable)) cvt_hf_to_f8_e5m2(half f) {
    // we just need to apply rounding
    ushort fraw = as_ushort(f);
    ushort naninf_mask = 0x7c00;

    bool is_special = (fraw & naninf_mask) == naninf_mask;
    bool is_nan = is_special && (fraw & 0x03ff); // one of the lsb is non zero

    // we always return R ind for Nan input as there is no good
    // conversion of payload
    if (is_nan) { return (fraw >> 8) | 0x02; }

    // if infinity, we just return it as is
    if (is_special) {
        uchar raw_bits = fraw >> 8;
        return raw_bits;
    }

    // otherwise we just round and return
    ushort rounding_nudge = 0x007f + ((fraw & 0x0100) >> 8);
    fraw = fraw + rounding_nudge;
    uchar raw_bits = fraw >> 8;
    return raw_bits;
}

uchar2 __attribute__((overloadable)) cvt_hf_to_f8_e5m2(half2 f) {
    uchar2 r;
    for (int i = 0; i < 2; i++) {
        r[i] = cvt_hf_to_f8_e5m2(f[i]);
    }
    return r;
}

uchar4 __attribute__((overloadable)) cvt_hf_to_f8_e5m2(half4 f) {
    uchar4 r;
    for (int i = 0; i < 4; i++) {
        r[i] = cvt_hf_to_f8_e5m2(f[i]);
    }
    return r;
}

uchar8 __attribute__((overloadable)) cvt_hf_to_f8_e5m2(half8 f) {
    uchar8 r;
    for (int i = 0; i < 8; i++) {
        r[i] = cvt_hf_to_f8_e5m2(f[i]);
    }
    return r;
}

uchar16 __attribute__((overloadable)) cvt_hf_to_f8_e5m2(half16 f) {
    uchar16 r;
    for (int i = 0; i < 16; i++) {
        r[i] = cvt_hf_to_f8_e5m2(f[i]);
    }
    return r;
}

half __attribute__((overloadable)) cvt_f8_e5m2_to_hf(uchar b) {
    uchar2 iraw = {0, b};
    return as_half(iraw);
}

half2 __attribute__((overloadable)) cvt_f8_e5m2_to_hf(uchar2 b) {
    half2 f;
    for (int i = 0; i < 2; i++) {
        f[i] = cvt_f8_e5m2_to_hf(b[i]);
    }
    return f;
}

half4 __attribute__((overloadable)) cvt_f8_e5m2_to_hf(uchar4 b) {
    half4 f;
    for (int i = 0; i < 4; i++) {
        f[i] = cvt_f8_e5m2_to_hf(b[i]);
    }
    return f;
}

half8 __attribute__((overloadable)) cvt_f8_e5m2_to_hf(uchar8 b) {
    half8 f;
    for (int i = 0; i < 8; i++) {
        f[i] = cvt_f8_e5m2_to_hf(b[i]);
    }
    return f;
}

half16 __attribute__((overloadable)) cvt_f8_e5m2_to_hf(uchar16 b) {
    half16 f;
    for (int i = 0; i < 16; i++) {
        f[i] = cvt_f8_e5m2_to_hf(b[i]);
    }
    return f;
}
#endif

#if MATH_UTILS_DECLARE_BF16
#ifdef cl_future_bf16_cvt
// f32 -> bf16 conversion builtins (rte rounding mode)
short __builtin_IB_ftobf_1(float a) __attribute__((const));
short2 __builtin_IB_ftobf_2(float2 a) __attribute__((const));
short4 __builtin_IB_ftobf_4(float4 a) __attribute__((const));
short8 __builtin_IB_ftobf_8(float8 a) __attribute__((const));
short16 __builtin_IB_ftobf_16(float16 a) __attribute__((const));

// bf16 -> f32 conversion builtins (precise conversion)
float __builtin_IB_bftof_1(short a) __attribute__((const));
float2 __builtin_IB_bftof_2(short2 a) __attribute__((const));
float4 __builtin_IB_bftof_4(short4 a) __attribute__((const));
float8 __builtin_IB_bftof_8(short8 a) __attribute__((const));
float16 __builtin_IB_bftof_16(short16 a) __attribute__((const));

// clang-format off
ushort   __attribute__((overloadable)) cvt_f32_to_bf16(float   a) { return as_ushort  (__builtin_IB_ftobf_1 (a)); }
ushort2  __attribute__((overloadable)) cvt_f32_to_bf16(float2  a) { return as_ushort2 (__builtin_IB_ftobf_2 (a)); }
ushort4  __attribute__((overloadable)) cvt_f32_to_bf16(float4  a) { return as_ushort4 (__builtin_IB_ftobf_4 (a)); }
ushort8  __attribute__((overloadable)) cvt_f32_to_bf16(float8  a) { return as_ushort8 (__builtin_IB_ftobf_8 (a)); }
ushort16 __attribute__((overloadable)) cvt_f32_to_bf16(float16 a) { return as_ushort16(__builtin_IB_ftobf_16(a)); }

float   __attribute__((overloadable)) cvt_bf16_to_f32(ushort   a) { return __builtin_IB_bftof_1 (as_short  (a)); }
float2  __attribute__((overloadable)) cvt_bf16_to_f32(ushort2  a) { return __builtin_IB_bftof_2 (as_short2 (a)); }
float4  __attribute__((overloadable)) cvt_bf16_to_f32(ushort4  a) { return __builtin_IB_bftof_4 (as_short4 (a)); }
float8  __attribute__((overloadable)) cvt_bf16_to_f32(ushort8  a) { return __builtin_IB_bftof_8 (as_short8 (a)); }
float16 __attribute__((overloadable)) cvt_bf16_to_f32(ushort16 a) { return __builtin_IB_bftof_16(as_short16(a)); }

#ifdef cl_khr_fp64
double   __attribute__((overloadable)) cvt_bf16_to_f64(ushort   a) { return convert_double(__builtin_IB_bftof_1 (as_short  (a))); }
double2  __attribute__((overloadable)) cvt_bf16_to_f64(ushort2  a) { return convert_double2(__builtin_IB_bftof_2 (as_short2 (a))); }
double4  __attribute__((overloadable)) cvt_bf16_to_f64(ushort4  a) { return convert_double4(__builtin_IB_bftof_4 (as_short4 (a))); }
double8  __attribute__((overloadable)) cvt_bf16_to_f64(ushort8  a) { return convert_double8(__builtin_IB_bftof_8 (as_short8 (a))); }
double16 __attribute__((overloadable)) cvt_bf16_to_f64(ushort16 a) { return convert_double16(__builtin_IB_bftof_16(as_short16(a))); }
#endif
// clang-format on

#else

// Emulation functions for bf16 <-> f32 conversion.
ushort __attribute__((overloadable)) cvt_f32_to_bf16(float f) {
    uint i = as_uint(f);
    i += 0x00007FFF + ((i & 0x10000) >> 16);
    ushort2 r = as_ushort2(i);
    return r[1];
}

ushort2 __attribute__((overloadable)) cvt_f32_to_bf16(float2 f) {
    ushort2 r;
    for (int i = 0; i < 2; i++) {
        r[i] = cvt_f32_to_bf16(f[i]);
    }
    return r;
}

ushort4 __attribute__((overloadable)) cvt_f32_to_bf16(float4 f) {
    ushort4 r;
    for (int i = 0; i < 4; i++) {
        r[i] = cvt_f32_to_bf16(f[i]);
    }
    return r;
}

ushort8 __attribute__((overloadable)) cvt_f32_to_bf16(float8 f) {
    ushort8 r;
    for (int i = 0; i < 8; i++) {
        r[i] = cvt_f32_to_bf16(f[i]);
    }
    return r;
}

ushort16 __attribute__((overloadable)) cvt_f32_to_bf16(float16 f) {
    ushort16 r;
    for (int i = 0; i < 16; i++) {
        r[i] = cvt_f32_to_bf16(f[i]);
    }
    return r;
}

float __attribute__((overloadable)) cvt_bf16_to_f32(ushort b) {
    ushort2 r = {0, b};
    float f = as_float(r);
    return f;
}

float2 __attribute__((overloadable)) cvt_bf16_to_f32(ushort2 b) {
    float2 f;
    for (int i = 0; i < 2; i++) {
        f[i] = cvt_bf16_to_f32(b[i]);
    }
    return f;
}

float4 __attribute__((overloadable)) cvt_bf16_to_f32(ushort4 b) {
    float4 f;
    for (int i = 0; i < 4; i++) {
        f[i] = cvt_bf16_to_f32(b[i]);
    }
    return f;
}

float8 __attribute__((overloadable)) cvt_bf16_to_f32(ushort8 b) {
    float8 f;
    for (int i = 0; i < 8; i++) {
        f[i] = cvt_bf16_to_f32(b[i]);
    }
    return f;
}

float16 __attribute__((overloadable)) cvt_bf16_to_f32(ushort16 b) {
    float16 f;
    for (int i = 0; i < 16; i++) {
        f[i] = cvt_bf16_to_f32(b[i]);
    }
    return f;
}

#ifdef cl_khr_fp64
// Emulation functions for bf16 -> f64 conversion.
double __attribute__((overloadable)) cvt_bf16_to_f64(ushort b) {
    ushort2 r = {0, b};
    float f = as_float(r);
    return convert_double(f);
}
double2 __attribute__((overloadable)) cvt_bf16_to_f64(ushort2 b) {
    double2 f;
    for (int i = 0; i < 2; i++) {
        f[i] = cvt_bf16_to_f64(b[i]);
    }
    return f;
}
double4 __attribute__((overloadable)) cvt_bf16_to_f64(ushort4 b) {
    double4 f;
    for (int i = 0; i < 4; i++) {
        f[i] = cvt_bf16_to_f64(b[i]);
    }
    return f;
}
double8 __attribute__((overloadable)) cvt_bf16_to_f64(ushort8 b) {
    double8 f;
    for (int i = 0; i < 8; i++) {
        f[i] = cvt_bf16_to_f64(b[i]);
    }
    return f;
}
double16 __attribute__((overloadable)) cvt_bf16_to_f64(ushort16 b) {
    double16 f;
    for (int i = 0; i < 16; i++) {
        f[i] = cvt_bf16_to_f64(b[i]);
    }
    return f;
}
#endif

#endif
#endif

#define DECLARE_BLOCK_READ(suffix, func, data_type, addr_space, p_type) \
    data_type __attribute__((overloadable)) \
            block_read##suffix(const addr_space p_type *p) { \
        return func(p); \
    }

#define DECLARE_BLOCK_READ_EMU(suffix, data_type, addr_space, p_type) \
    data_type __attribute__((overloadable)) \
            block_read##suffix##_emu(const addr_space p_type *p) { \
        data_type ret; \
        uint idx = get_sub_group_local_id(); \
        for (int i = 0; i < sizeof(data_type) / sizeof(p_type); i++) { \
            ((p_type *)&ret)[i] = p[idx]; \
            idx += get_max_sub_group_size(); \
        } \
        return ret; \
    }

#define DECLARE_BLOCK_WRITE(suffix, func, data_type, addr_space, p_type) \
    void __attribute__((overloadable)) \
            block_write##suffix(addr_space p_type *p, data_type data) { \
        func(p, data); \
    }

#define DECLARE_BLOCK_WRITE_EMU(suffix, data_type, addr_space, p_type) \
    void __attribute__((overloadable)) \
            block_write##suffix##_emu(addr_space p_type *p, data_type data) { \
        uint idx = get_sub_group_local_id(); \
        for (int i = 0; i < sizeof(data_type) / sizeof(p_type); i++) { \
            p[idx] = ((p_type *)&data)[i]; \
            p += get_max_sub_group_size(); \
        } \
    }

DECLARE_BLOCK_READ(, intel_sub_group_block_read, uint, __global, uint)
DECLARE_BLOCK_READ(2, intel_sub_group_block_read2, uint2, __global, uint)
DECLARE_BLOCK_READ(4, intel_sub_group_block_read4, uint4, __global, uint)
DECLARE_BLOCK_READ(8, intel_sub_group_block_read8, uint8, __global, uint)

DECLARE_BLOCK_WRITE(, intel_sub_group_block_write, uint, __global, uint)
DECLARE_BLOCK_WRITE(2, intel_sub_group_block_write2, uint2, __global, uint)
DECLARE_BLOCK_WRITE(4, intel_sub_group_block_write4, uint4, __global, uint)
DECLARE_BLOCK_WRITE(8, intel_sub_group_block_write8, uint8, __global, uint)

#ifdef cl_intel_subgroups_char
void __attribute__((overloadable))
intel_sub_group_block_write_uc16(__global uchar *p, uchar16 data);

uchar16 __attribute__((overloadable))
intel_sub_group_block_read_uc16(const __global uchar *p);
#endif

// Emulation for cl_intel_subgroup_local_block_io. These functions are not
// defined under ifndef/endif because some kernels rely on the emulation
// functions in case when pointers are not properly aligned for the native
// extensions.
DECLARE_BLOCK_READ_EMU(, uint, __local, uint)
DECLARE_BLOCK_READ_EMU(2, uint2, __local, uint)
DECLARE_BLOCK_READ_EMU(4, uint4, __local, uint)
DECLARE_BLOCK_READ_EMU(8, uint8, __local, uint)

DECLARE_BLOCK_WRITE_EMU(, uint, __local, uint)
DECLARE_BLOCK_WRITE_EMU(2, uint2, __local, uint)
DECLARE_BLOCK_WRITE_EMU(4, uint4, __local, uint)
DECLARE_BLOCK_WRITE_EMU(8, uint8, __local, uint)

DECLARE_BLOCK_WRITE_EMU(_us, ushort, __local, ushort)
DECLARE_BLOCK_WRITE_EMU(_us2, ushort2, __local, ushort)
DECLARE_BLOCK_WRITE_EMU(_us4, ushort4, __local, ushort)
DECLARE_BLOCK_WRITE_EMU(_us8, ushort8, __local, ushort)
#ifdef cl_intel_subgroup_local_block_io

DECLARE_BLOCK_READ(, intel_sub_group_block_read, uint, __local, uint)
DECLARE_BLOCK_READ(2, intel_sub_group_block_read2, uint2, __local, uint)
DECLARE_BLOCK_READ(4, intel_sub_group_block_read4, uint4, __local, uint)
DECLARE_BLOCK_READ(8, intel_sub_group_block_read8, uint8, __local, uint)

DECLARE_BLOCK_WRITE(, intel_sub_group_block_write, uint, __local, uint)
DECLARE_BLOCK_WRITE(2, intel_sub_group_block_write2, uint2, __local, uint)
DECLARE_BLOCK_WRITE(4, intel_sub_group_block_write4, uint4, __local, uint)
DECLARE_BLOCK_WRITE(8, intel_sub_group_block_write8, uint8, __local, uint)

DECLARE_BLOCK_WRITE(
        _us, intel_sub_group_block_write_us, ushort, __local, ushort)

#else

DECLARE_BLOCK_READ(, block_read_emu, uint, __local, uint)
DECLARE_BLOCK_READ(2, block_read2_emu, uint2, __local, uint)
DECLARE_BLOCK_READ(4, block_read4_emu, uint4, __local, uint)
DECLARE_BLOCK_READ(8, block_read8_emu, uint8, __local, uint)

DECLARE_BLOCK_WRITE(, block_write_emu, uint, __local, uint)
DECLARE_BLOCK_WRITE(2, block_write2_emu, uint2, __local, uint)
DECLARE_BLOCK_WRITE(4, block_write4_emu, uint4, __local, uint)
DECLARE_BLOCK_WRITE(8, block_write8_emu, uint8, __local, uint)

DECLARE_BLOCK_WRITE(_us, block_write_us_emu, ushort, __local, ushort)

#endif

// Atomics
#if !DETERMINISTIC && ATOMICS_SUPPORTED
#define ATOMIC(x) CONCAT2(atomic_, x)
#define DECLARE_ATOMIC_OP(op, type) \
    type __attribute__((overloadable)) CONCAT3(atomic_, op, _global)( \
            volatile global ATOMIC(type) * source, type operand) { \
        return CONCAT3(atomic_fetch_, op, _explicit)( \
                source, operand, memory_order_relaxed); \
    }

DECLARE_ATOMIC_OP(min, int)
DECLARE_ATOMIC_OP(max, int)
DECLARE_ATOMIC_OP(add, int)
DECLARE_ATOMIC_OP(sub, int)

#if ATOMIC_FLOAT_SUPPORTED
#ifdef __opencl_c_ext_fp32_global_atomic_add
#define HAS_FLOAT_ATOMIC_ADD
DECLARE_ATOMIC_OP(add, float)
DECLARE_ATOMIC_OP(sub, float)
#endif // __opencl_c_ext_fp32_global_atomic_add

#ifdef __opencl_c_ext_fp32_global_atomic_min_max
DECLARE_ATOMIC_OP(min, float)
DECLARE_ATOMIC_OP(max, float)
#endif // __opencl_c_ext_fp32_global_atomic_min_max
#endif // ATOMIC_FLOATS_SUPPORTED

#ifndef HAS_FLOAT_ATOMIC_ADD
// Fallback atomic implementations
inline float atomic_add_global(
        volatile __global atomic_float *source, float operand) {
    float old_val = atomic_load_explicit(
            source, memory_order_relaxed, memory_scope_device);
    bool success = false;
    do {
        float new_val = old_val + operand;
        success = atomic_compare_exchange_strong_explicit(source, &old_val,
                new_val, memory_order_acq_rel, memory_order_relaxed,
                memory_scope_device);
    } while (!success);
    return old_val;
}
#endif
#endif

#if MATH_UTILS_DECLARE_S4 || MATH_UTILS_DECLARE_U4

uchar __attribute__((overloadable)) cvt_f32_to_u4(float a) {
    uchar i = convert_uchar_sat_rte(a);
    return (i & 0xf0) ? 0x0f : i & 0x0f;
}

char __attribute__((overloadable)) cvt_f32_to_s4(float a) {
    return convert_char_sat_rte(min(max(a, -8.0f), 7.0f)) & 0x0F;
}

float __attribute__((overloadable)) cvt_s4_to_f32(char a) {
    char sign = (a & 0x08) ? 0xf0 : 0x0;
    char val = a | sign;
    return convert_float(val);
}

float __attribute__((overloadable)) cvt_s4_to_s32(char a) {
    char sign = (a & 0x08) ? 0xf0 : 0x0;
    char val = a | sign;
    return convert_int_sat_rte(val);
}

#endif

#if MATH_UTILS_DECLARE_F4_E2M1

uchar __attribute__((overloadable)) cvt_f32_to_f4_e2m1(float a) {
    const float f4_e2m1_max = as_float(0x40c00000);
    const float exp_shift = as_float(0x00800000);

    // clamp
    // sel (lt)f0.0 t0:f (abs)x:f 0x40c00000:f
    float intermediate = fmin(fabs(a), f4_e2m1_max);
    if (isnan(intermediate)) intermediate = f4_e2m1_max;

    // shift high exp bit down
    // mul t0:f t0:f 0x00800000:f
    intermediate *= exp_shift;

    // rtne logic
    // add t0:ud to:ud -0x00200000:ud
    // and (nz)f0.0 null t0:ud 0x007fffff:ud
    uint bits = as_uint(intermediate);
    bits -= 0x00200000;
    uint round_up = bits & 0x007fffff;

    // shr t0:ud t0:ud 22
    bits >>= 22;

    // round
    // (f0.0) add t0:ud t0:ud 1
    if (round_up) bits += 1;

    // copy sign
    // shr y:ud x:ud 28
    // bfn.0xCA y:ud y:ud t0:ud 0x07
    uint dst = as_uint(a) >> 28;
    return ((dst & ~0x07) | (bits & 0x07)) & 0xf;
}

float __attribute__((overloadable)) cvt_f4_e2m1_to_f32(uchar a) {
    uint sign = a & 0x08;
    uint em = a & 0x07;
    uint exp = em >> 1;
    uint mant = exp ? a & 0x01 : 0x0;
    if (em) exp += 126; // No f4 values are subnormal in f32
    return as_float((sign << 28) | (exp << 23) | (mant << 22));
}

#endif
#if MATH_UTILS_DECLARE_F4_E3M0

// OCL translation of common fp4 methods.
uchar __attribute__((overloadable)) cvt_f32_to_f4_e3m0(float a) {
    const float f4_e3m0_max = as_float(0x41800000);
    const float exp_shift = as_float(0x01800000);

    // clamp
    // sel (lt)f0.0 t0:f (abs)x:f 0x41800000:f
    float intermediate = fmin(fabs(a), f4_e3m0_max);
    if (isnan(intermediate)) intermediate = f4_e3m0_max;

    // shift high exp bit down
    // mul t0:f t0:f 0x01800000:f
    intermediate *= exp_shift;

    // rtne logic
    // add t0:ud to:ud -0x00400000:ud
    // and (nz)f0.0 null t0:ud 0x00ffffff:ud
    uint bits = as_uint(intermediate);
    bits -= 0x00400000;
    uint round_up = bits & 0x00ffffff;

    // shr t0:ud t0:ud 23
    bits >>= 23;

    // round
    // (f0.0) add t0:ud t0:ud 1
    if (round_up) bits += 1;

    // copy sign
    // shr y:ud x:ud 28
    // bfn.0xCA y:ud y:ud t0:ud 0x07
    uint dst = as_uint(a) >> 28;
    return ((dst & ~0x07) | (bits & 0x07)) & 0xf;
}

float __attribute__((overloadable)) cvt_f4_e3m0_to_f32(uchar a) {
    // List of e3m0 values. The index of each value maps to its encoding.
    const float e3m0_table[16] = {0.0f, .25f, .5f, 1.0f, 2.0f, 4.0f, 8.0f,
            16.0f, -0.0f, -.25f, -.5f, -1.0f, -2.0f, -4.0f, -8.0f, -16.0f};
    return e3m0_table[a];
}

#endif

#if MATH_UTILS_DECLARE_S4 || MATH_UTILS_DECLARE_U4 \
        || MATH_UTILS_DECLARE_F4_E2M1 || MATH_UTILS_DECLARE_F4_E3M0
#define GET_HALF_BYTE(x, y) get_half_byte(x, y)

uchar __attribute__((overloadable))
get_half_byte(const __global uchar *x, off_t y) {
    uchar ret = 0;
    if (y % 2) {
        ret = (uchar)((uchar)(x[y / 2] & 0xf0) >> 4);
    } else {
        ret = (uchar)(x[y / 2] & 0x0f);
    }
    return ret;
}

char __attribute__((overloadable))
get_half_byte(const __global char *x, off_t y) {
    if (y % 2) {
        return (x[y / 2] & 0xf0) >> 4;
    } else {
        return x[y / 2] & 0x0f;
    }
}

void __attribute__((overloadable))
set_double_half_byte(__global uchar *x, off_t y, uchar z) {
    x[y / 2] = z;
}

void __attribute__((overloadable))
set_double_half_byte(__global char *x, off_t y, uchar z) {
    x[y / 2] = z;
}

#endif
#endif
