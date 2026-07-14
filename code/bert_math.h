#ifndef BERT_MATH_H
#define BERT_MATH_H

#include "bert_kernel_interfaces.h"

static float uint32_to_float(uint32_t bits)
{
#pragma HLS INLINE
    union { uint32_t u; float f; } value;
    value.u = bits;
    return value.f;
}

static uint32_t float_to_uint32(float input)
{
#pragma HLS INLINE
    union { uint32_t u; float f; } value;
    value.f = input;
    return value.u;
}

static void unpack_bus16(bus_t word, float lane[PACK_SIZE])
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=lane complete dim=1
    lane[0]  = uint32_to_float((uint32_t)word.range(31, 0));
    lane[1]  = uint32_to_float((uint32_t)word.range(63, 32));
    lane[2]  = uint32_to_float((uint32_t)word.range(95, 64));
    lane[3]  = uint32_to_float((uint32_t)word.range(127, 96));
    lane[4]  = uint32_to_float((uint32_t)word.range(159, 128));
    lane[5]  = uint32_to_float((uint32_t)word.range(191, 160));
    lane[6]  = uint32_to_float((uint32_t)word.range(223, 192));
    lane[7]  = uint32_to_float((uint32_t)word.range(255, 224));
    lane[8]  = uint32_to_float((uint32_t)word.range(287, 256));
    lane[9]  = uint32_to_float((uint32_t)word.range(319, 288));
    lane[10] = uint32_to_float((uint32_t)word.range(351, 320));
    lane[11] = uint32_to_float((uint32_t)word.range(383, 352));
    lane[12] = uint32_to_float((uint32_t)word.range(415, 384));
    lane[13] = uint32_to_float((uint32_t)word.range(447, 416));
    lane[14] = uint32_to_float((uint32_t)word.range(479, 448));
    lane[15] = uint32_to_float((uint32_t)word.range(511, 480));
}

static bus_t pack_bus16(const float lane[PACK_SIZE])
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=lane complete dim=1
    bus_t word = 0;
    word.range(31, 0)    = float_to_uint32(lane[0]);
    word.range(63, 32)   = float_to_uint32(lane[1]);
    word.range(95, 64)   = float_to_uint32(lane[2]);
    word.range(127, 96)  = float_to_uint32(lane[3]);
    word.range(159, 128) = float_to_uint32(lane[4]);
    word.range(191, 160) = float_to_uint32(lane[5]);
    word.range(223, 192) = float_to_uint32(lane[6]);
    word.range(255, 224) = float_to_uint32(lane[7]);
    word.range(287, 256) = float_to_uint32(lane[8]);
    word.range(319, 288) = float_to_uint32(lane[9]);
    word.range(351, 320) = float_to_uint32(lane[10]);
    word.range(383, 352) = float_to_uint32(lane[11]);
    word.range(415, 384) = float_to_uint32(lane[12]);
    word.range(447, 416) = float_to_uint32(lane[13]);
    word.range(479, 448) = float_to_uint32(lane[14]);
    word.range(511, 480) = float_to_uint32(lane[15]);
    return word;
}

static float fp32_sum8_tree(const float value[8])
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=value complete dim=1
    const float s0 = value[0] + value[1];
    const float s1 = value[2] + value[3];
    const float s2 = value[4] + value[5];
    const float s3 = value[6] + value[7];
    return (s0 + s1) + (s2 + s3);
}

#endif
