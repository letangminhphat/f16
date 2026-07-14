#include "bert_ffn_kernel_lab.h"
#include "bert_encoder_protocol.h"

#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>

#define PACK_BITS 512
#define PACK_FLOATS 16
#define HIDDEN_PACKS (HIDDEN_SIZE / PACK_FLOATS)
#define V10_TILE_K 64
#define V10_K_PAR 16
#define V10_MAC_GROUP 16
#define V13_TILE_I 16
#define V13_TILE_O 32
#define V15_W1_TILE_PACKS (V13_TILE_I / PACK_FLOATS)
#define V15_W2_TILE_PACKS (V13_TILE_O / PACK_FLOATS)
#define V15_HIDDEN_TILES (HIDDEN_SIZE / V13_TILE_O)
#define FFN_UP_WEIGHT_WORDS (HIDDEN_SIZE * INTERMEDIATE_SIZE / PACK_FLOATS)
#define FFN_UP_BIAS_WORDS (INTERMEDIATE_SIZE / PACK_FLOATS)
#define FFN_DOWN_WEIGHT_WORDS (INTERMEDIATE_SIZE * HIDDEN_SIZE / PACK_FLOATS)
#define FFN_DOWN_BIAS_WORDS HIDDEN_PACKS

#if ((HIDDEN_SIZE % V10_TILE_K) != 0)
#error "HIDDEN_SIZE must be divisible by V10_TILE_K"
#endif

#if ((INTERMEDIATE_SIZE % V13_TILE_I) != 0)
#error "INTERMEDIATE_SIZE must be divisible by V13_TILE_I"
#endif

#if ((V13_TILE_I % PACK_FLOATS) != 0)
#error "V13_TILE_I must be divisible by PACK_FLOATS"
#endif

#if ((V13_TILE_I % V10_K_PAR) != 0)
#error "V13_TILE_I must be divisible by V10_K_PAR"
#endif

#if ((HIDDEN_SIZE % V13_TILE_O) != 0)
#error "HIDDEN_SIZE must be divisible by V13_TILE_O"
#endif

#if ((V13_TILE_O % V10_MAC_GROUP) != 0)
#error "V13_TILE_O must be divisible by V10_MAC_GROUP"
#endif

#if ((V13_TILE_O % PACK_FLOATS) != 0)
#error "V13_TILE_O must be divisible by PACK_FLOATS"
#endif

#if (V15_W2_TILE_PACKS != 2)
#error "V15 W2 load timing path assumes two packed words per output tile"
#endif

typedef ap_uint<PACK_BITS> packed_float16_t;
typedef hls::stream<packed_float16_t> packed_float_stream_t;

static float gelu_pwl_packed(float x) {
#pragma HLS INLINE
    float gate;

    if (x <= -3.0f) {
        return 0.0f;
    }
    if (x >= 3.0f) {
        return x;
    }

    gate = 0.5f + x * (1.0f / 6.0f);
    return x * gate;
}

static float bits_to_float(ap_uint<32> bits) {
#pragma HLS INLINE
    union {
        unsigned int u;
        float f;
    } conv;

    conv.u = (unsigned int)bits;
    return conv.f;
}

static ap_uint<32> float_to_bits(float value) {
#pragma HLS INLINE
    union {
        unsigned int u;
        float f;
    } conv;

    conv.f = value;
    return conv.u;
}

static void unpack_packed_float16(packed_float16_t word, float values[PACK_FLOATS])
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=values complete dim=1
    values[0] = bits_to_float(word.range(31, 0));
    values[1] = bits_to_float(word.range(63, 32));
    values[2] = bits_to_float(word.range(95, 64));
    values[3] = bits_to_float(word.range(127, 96));
    values[4] = bits_to_float(word.range(159, 128));
    values[5] = bits_to_float(word.range(191, 160));
    values[6] = bits_to_float(word.range(223, 192));
    values[7] = bits_to_float(word.range(255, 224));
    values[8] = bits_to_float(word.range(287, 256));
    values[9] = bits_to_float(word.range(319, 288));
    values[10] = bits_to_float(word.range(351, 320));
    values[11] = bits_to_float(word.range(383, 352));
    values[12] = bits_to_float(word.range(415, 384));
    values[13] = bits_to_float(word.range(447, 416));
    values[14] = bits_to_float(word.range(479, 448));
    values[15] = bits_to_float(word.range(511, 480));
}

static packed_float16_t pack_packed_float16(const float values[PACK_FLOATS])
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=values complete dim=1
    packed_float16_t word = 0;
    word.range(31, 0) = float_to_bits(values[0]);
    word.range(63, 32) = float_to_bits(values[1]);
    word.range(95, 64) = float_to_bits(values[2]);
    word.range(127, 96) = float_to_bits(values[3]);
    word.range(159, 128) = float_to_bits(values[4]);
    word.range(191, 160) = float_to_bits(values[5]);
    word.range(223, 192) = float_to_bits(values[6]);
    word.range(255, 224) = float_to_bits(values[7]);
    word.range(287, 256) = float_to_bits(values[8]);
    word.range(319, 288) = float_to_bits(values[9]);
    word.range(351, 320) = float_to_bits(values[10]);
    word.range(383, 352) = float_to_bits(values[11]);
    word.range(415, 384) = float_to_bits(values[12]);
    word.range(447, 416) = float_to_bits(values[13]);
    word.range(479, 448) = float_to_bits(values[14]);
    word.range(511, 480) = float_to_bits(values[15]);
    return word;
}

static void v10_load_input_matrix(
    packed_float16_t *input_hidden,
    float input_buf[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off

v10_load_input_s:
    for (int s = 0; s < SEQ_LEN; s++) {
        int input_index = s * HIDDEN_PACKS;
    v10_load_input_pack:
        for (int h_pack = 0; h_pack < HIDDEN_PACKS; h_pack++) {
#pragma HLS PIPELINE II=1
            float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
            packed_float16_t word = input_hidden[input_index + h_pack];
            unpack_packed_float16(word, lanes);
        v10_load_input_lane:
            for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                input_buf[s][h_pack * PACK_FLOATS + lane] = lanes[lane];
            }
        }
    }
}

static void v10_store_output_matrix(
    float output_buf[SEQ_LEN][HIDDEN_SIZE],
    packed_float16_t *output_hidden)
{
#pragma HLS INLINE off

v10_store_output_s:
    for (int s = 0; s < SEQ_LEN; s++) {
        int output_index = s * HIDDEN_PACKS;
    v10_store_output_pack:
        for (int h_pack = 0; h_pack < HIDDEN_PACKS; h_pack++) {
#pragma HLS PIPELINE II=1
            float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
            int h_base = h_pack << 4;
        v10_store_output_lane:
            for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                lanes[lane] = output_buf[s][h_base + lane];
            }
            output_hidden[output_index + h_pack] = pack_packed_float16(lanes);
        }
    }
}

static void v11_load_input_stream_matrix(
    packed_float_stream_t &input_stream,
    float input_buf[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off

v11_load_input_stream_s:
    for (int s = 0; s < SEQ_LEN; s++) {
    v11_load_input_stream_pack:
        for (int h_pack = 0; h_pack < HIDDEN_PACKS; h_pack++) {
#pragma HLS PIPELINE II=1
            float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
            packed_float16_t word = input_stream.read();
            unpack_packed_float16(word, lanes);
        v11_load_input_stream_lane:
            for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                input_buf[s][h_pack * PACK_FLOATS + lane] = lanes[lane];
            }
        }
    }
}

static float v13_dot16_tree_margin(
    const float lhs[V10_K_PAR],
    const float rhs[V10_K_PAR])
{
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
#pragma HLS ARRAY_PARTITION variable=lhs complete dim=1
#pragma HLS ARRAY_PARTITION variable=rhs complete dim=1
    float p0, p1, p2, p3, p4, p5, p6, p7;
    float p8, p9, p10, p11, p12, p13, p14, p15;
    float s8[8];
    float s4[4];
    float s2[2];
    float sum;
#pragma HLS ARRAY_PARTITION variable=s8 complete dim=1
#pragma HLS ARRAY_PARTITION variable=s4 complete dim=1
#pragma HLS ARRAY_PARTITION variable=s2 complete dim=1
#pragma HLS BIND_OP variable=p0 op=fmul impl=fulldsp latency=7
#pragma HLS BIND_OP variable=p1 op=fmul impl=fulldsp latency=7
#pragma HLS BIND_OP variable=p2 op=fmul impl=fulldsp latency=7
#pragma HLS BIND_OP variable=p3 op=fmul impl=fulldsp latency=7
#pragma HLS BIND_OP variable=p4 op=fmul impl=fulldsp latency=7
#pragma HLS BIND_OP variable=p5 op=fmul impl=fulldsp latency=7
#pragma HLS BIND_OP variable=p6 op=fmul impl=fulldsp latency=7
#pragma HLS BIND_OP variable=p7 op=fmul impl=fulldsp latency=7
#pragma HLS BIND_OP variable=p8 op=fmul impl=fulldsp latency=7
#pragma HLS BIND_OP variable=p9 op=fmul impl=fulldsp latency=7
#pragma HLS BIND_OP variable=p10 op=fmul impl=fulldsp latency=7
#pragma HLS BIND_OP variable=p11 op=fmul impl=fulldsp latency=7
#pragma HLS BIND_OP variable=p12 op=fmul impl=fulldsp latency=7
#pragma HLS BIND_OP variable=p13 op=fmul impl=fulldsp latency=7
#pragma HLS BIND_OP variable=p14 op=fmul impl=fulldsp latency=7
#pragma HLS BIND_OP variable=p15 op=fmul impl=fulldsp latency=7
#pragma HLS BIND_OP variable=s8 op=fadd impl=fulldsp latency=6
#pragma HLS BIND_OP variable=s4 op=fadd impl=fulldsp latency=6
#pragma HLS BIND_OP variable=s2 op=fadd impl=fulldsp latency=6
#pragma HLS BIND_OP variable=sum op=fadd impl=fulldsp latency=6

    p0 = lhs[0] * rhs[0];
    p1 = lhs[1] * rhs[1];
    p2 = lhs[2] * rhs[2];
    p3 = lhs[3] * rhs[3];
    p4 = lhs[4] * rhs[4];
    p5 = lhs[5] * rhs[5];
    p6 = lhs[6] * rhs[6];
    p7 = lhs[7] * rhs[7];
    p8 = lhs[8] * rhs[8];
    p9 = lhs[9] * rhs[9];
    p10 = lhs[10] * rhs[10];
    p11 = lhs[11] * rhs[11];
    p12 = lhs[12] * rhs[12];
    p13 = lhs[13] * rhs[13];
    p14 = lhs[14] * rhs[14];
    p15 = lhs[15] * rhs[15];

    s8[0] = p0 + p1;
    s8[1] = p2 + p3;
    s8[2] = p4 + p5;
    s8[3] = p6 + p7;
    s8[4] = p8 + p9;
    s8[5] = p10 + p11;
    s8[6] = p12 + p13;
    s8[7] = p14 + p15;

v13_dot_l2:
    for (int i = 0; i < 4; i++) {
#pragma HLS UNROLL
        s4[i] = s8[2 * i] + s8[2 * i + 1];
    }

v13_dot_l3:
    for (int i = 0; i < 2; i++) {
#pragma HLS UNROLL
        s2[i] = s4[2 * i] + s4[2 * i + 1];
    }

    sum = s2[0] + s2[1];
    return sum;
}

static void v15_ffn_up_tilemajor_producer(
    float input_buf[SEQ_LEN][HIDDEN_SIZE],
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float_stream_t &gelu_stream)
{
#pragma HLS INLINE off
    float w_tile[V13_TILE_I][V10_TILE_K];
    float acc[SEQ_LEN][V13_TILE_I];
#pragma HLS BIND_STORAGE variable=w_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=acc type=RAM_2P impl=BRAM latency=2
#pragma HLS ARRAY_PARTITION variable=w_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_tile cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=acc complete dim=2

v15_up_output_tile:
    for (int i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += V13_TILE_I) {
        int i0_pack = i0 / PACK_FLOATS;
        int i_tile = i0 / V13_TILE_I;
    v15_up_init_s:
        for (int s = 0; s < SEQ_LEN; s++) {
        v15_up_init_pack:
            for (int i_pack = 0; i_pack < V15_W1_TILE_PACKS; i_pack++) {
#pragma HLS PIPELINE II=1
                float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
                packed_float16_t word = b1[i0_pack + i_pack];
                unpack_packed_float16(word, lanes);
            v15_up_init_lane:
                for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    acc[s][i_pack * PACK_FLOATS + lane] = lanes[lane];
                }
            }
        }

    v15_up_hidden_tile:
        for (int h0 = 0; h0 < HIDDEN_SIZE; h0 += V10_TILE_K) {
            int packed_base = i_tile * HIDDEN_SIZE * V15_W1_TILE_PACKS + h0 * V15_W1_TILE_PACKS;
        v15_up_load_w_k:
            for (int tk = 0; tk < V10_TILE_K; tk++) {
            v15_up_load_w_pack:
                for (int i_pack = 0; i_pack < V15_W1_TILE_PACKS; i_pack++) {
#pragma HLS PIPELINE II=1
                    float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
                    packed_float16_t word = w1[packed_base + i_pack];
                    unpack_packed_float16(word, lanes);
                v15_up_load_w_lane:
                    for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                        w_tile[i_pack * PACK_FLOATS + lane][tk] = lanes[lane];
                    }
                }
                packed_base += V15_W1_TILE_PACKS;
            }

        v15_up_k_chunk:
            for (int tk = 0; tk < V10_TILE_K; tk += V10_K_PAR) {
            v15_up_seq:
                for (int s = 0; s < SEQ_LEN; s++) {
                v15_up_mac_group:
                    for (int io = 0; io < V13_TILE_I; io += V10_MAC_GROUP) {
#pragma HLS PIPELINE II=1
                        float x_vec[V10_K_PAR];
                        float acc_vec[V10_MAC_GROUP];
#pragma HLS ARRAY_PARTITION variable=x_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_vec complete dim=1
#pragma HLS BIND_OP variable=acc_vec op=fadd impl=fulldsp latency=5

                    v15_up_load_x_vec:
                        for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                            x_vec[kk] = input_buf[s][h0 + tk + kk];
                        }

                    v15_up_load_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc_vec[jj] = acc[s][io + jj];
                        }

                    v15_up_mac_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            float w_vec[V10_K_PAR];
#pragma HLS ARRAY_PARTITION variable=w_vec complete dim=1
                        v15_up_load_w_vec:
                            for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                                w_vec[kk] = w_tile[io + jj][tk + kk];
                            }
                            acc_vec[jj] += v13_dot16_tree_margin(x_vec, w_vec);
                        }

                    v15_up_store_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc[s][io + jj] = acc_vec[jj];
                        }
                    }
                }
            }
        }

    v15_up_write_gelu_s:
        for (int s = 0; s < SEQ_LEN; s++) {
        v15_up_write_gelu_pack:
            for (int i_pack = 0; i_pack < V15_W1_TILE_PACKS; i_pack++) {
#pragma HLS PIPELINE II=1
                float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
            v15_up_write_gelu_lane:
                for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    lanes[lane] = gelu_pwl_packed(acc[s][i_pack * PACK_FLOATS + lane]);
                }
                packed_float16_t word = pack_packed_float16(lanes);
                gelu_stream.write(word);
            }
        }
    }
}

static void v15_initialize_down_bias(
    packed_float16_t *b2,
    float output_buf[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
v15_down_init_s:
    for (int s = 0; s < SEQ_LEN; s++) {
    v15_down_init_pack:
        for (int h_pack = 0; h_pack < HIDDEN_PACKS; h_pack++) {
#pragma HLS PIPELINE II=1
            float bias_lane[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=bias_lane complete dim=1
            unpack_packed_float16(b2[h_pack], bias_lane);
        v15_down_init_lane:
            for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                output_buf[s][h_pack * PACK_FLOATS + lane] = bias_lane[lane];
            }
        }
    }
}

static void v21_initialize_down_bias_residual(
    packed_float_stream_t &residual_stream,
    packed_float16_t *b2,
    float output_buf[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
v21_down_residual_s:
    for (int s = 0; s < SEQ_LEN; s++) {
    v21_down_residual_pack:
        for (int h_pack = 0; h_pack < HIDDEN_PACKS; h_pack++) {
#pragma HLS PIPELINE II=1
            float bias_lane[PACK_FLOATS];
            float residual_lane[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=bias_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=residual_lane complete dim=1
            unpack_packed_float16(b2[h_pack], bias_lane);
            unpack_packed_float16(residual_stream.read(), residual_lane);
        v21_down_residual_lane:
            for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                output_buf[s][h_pack * PACK_FLOATS + lane] =
                    bias_lane[lane] + residual_lane[lane];
            }
        }
    }
}

static void v15_ffn_down_tilemajor_consumer(
    packed_float_stream_t &gelu_stream,
    packed_float16_t *w2,
    float output_buf[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
    float activation[SEQ_LEN][V13_TILE_I];
    float w_tile[V13_TILE_O][V13_TILE_I];
#pragma HLS BIND_STORAGE variable=activation type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=w_tile type=RAM_2P impl=BRAM latency=2
#pragma HLS ARRAY_PARTITION variable=activation complete dim=2
#pragma HLS ARRAY_PARTITION variable=w_tile complete dim=1
#pragma HLS ARRAY_PARTITION variable=w_tile complete dim=2

v15_down_ffn_tile:
    for (int i0 = 0; i0 < INTERMEDIATE_SIZE; i0 += V13_TILE_I) {
        int i_tile = i0 / V13_TILE_I;
    v15_down_read_gelu_s:
        for (int s = 0; s < SEQ_LEN; s++) {
        v15_down_read_gelu_pack:
            for (int i_pack = 0; i_pack < V15_W1_TILE_PACKS; i_pack++) {
#pragma HLS PIPELINE II=1
                float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
                packed_float16_t word = gelu_stream.read();
                unpack_packed_float16(word, lanes);
            v15_down_read_gelu_lane:
                for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    activation[s][i_pack * PACK_FLOATS + lane] = lanes[lane];
                }
            }
        }

        v15_down_output_tile:
        for (int o0 = 0; o0 < HIDDEN_SIZE; o0 += V13_TILE_O) {
            int o_tile = o0 / V13_TILE_O;
            int packed_base = ((i_tile * V15_HIDDEN_TILES + o_tile) * V13_TILE_I) * V15_W2_TILE_PACKS;
        v15_down_load_w_k:
            for (int packed_offset = 0; packed_offset < V13_TILE_I * V15_W2_TILE_PACKS; packed_offset++) {
#pragma HLS PIPELINE II=1
                int tk = packed_offset >> 1;
                int o_pack = packed_offset & 1;
                float lanes[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lanes complete dim=1
                packed_float16_t word = w2[packed_base + packed_offset];
                unpack_packed_float16(word, lanes);
            v15_down_load_w_lane:
                for (int lane = 0; lane < PACK_FLOATS; lane++) {
#pragma HLS UNROLL
                    w_tile[o_pack * PACK_FLOATS + lane][tk] = lanes[lane];
                }
            }

        v15_down_k_chunk:
            for (int tk = 0; tk < V13_TILE_I; tk += V10_K_PAR) {
            v15_down_seq:
                for (int s = 0; s < SEQ_LEN; s++) {
                v15_down_mac_group:
                    for (int oo = 0; oo < V13_TILE_O; oo += V10_MAC_GROUP) {
#pragma HLS PIPELINE II=1
                        float x_vec[V10_K_PAR];
                        float acc_vec[V10_MAC_GROUP];
#pragma HLS ARRAY_PARTITION variable=x_vec complete dim=1
#pragma HLS ARRAY_PARTITION variable=acc_vec complete dim=1
#pragma HLS BIND_OP variable=acc_vec op=fadd impl=fulldsp latency=5

                    v15_down_load_x_vec:
                        for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                            x_vec[kk] = activation[s][tk + kk];
                        }

                    v15_down_load_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            acc_vec[jj] = output_buf[s][o0 + oo + jj];
                        }

                    v15_down_mac_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            float w_vec[V10_K_PAR];
#pragma HLS ARRAY_PARTITION variable=w_vec complete dim=1
                        v15_down_load_w_vec:
                            for (int kk = 0; kk < V10_K_PAR; kk++) {
#pragma HLS UNROLL
                                w_vec[kk] = w_tile[oo + jj][tk + kk];
                            }
                            acc_vec[jj] += v13_dot16_tree_margin(x_vec, w_vec);
                        }

                    v15_down_store_acc_vec:
                        for (int jj = 0; jj < V10_MAC_GROUP; jj++) {
#pragma HLS UNROLL
                            output_buf[s][o0 + oo + jj] = acc_vec[jj];
                        }
                    }
                }
            }
        }
    }
}

static void v21_layernorm_and_store(
    float output_buf[SEQ_LEN][HIDDEN_SIZE],
    packed_float16_t *gamma,
    packed_float16_t *beta,
    packed_float16_t *output_hidden)
{
#pragma HLS INLINE off
    float gamma_cache[HIDDEN_SIZE];
    float beta_cache[HIDDEN_SIZE];
#pragma HLS BIND_STORAGE variable=gamma_cache type=RAM_2P impl=BRAM latency=2
#pragma HLS BIND_STORAGE variable=beta_cache type=RAM_2P impl=BRAM latency=2
#pragma HLS ARRAY_PARTITION variable=gamma_cache cyclic factor=PACK_FLOATS dim=1
#pragma HLS ARRAY_PARTITION variable=beta_cache cyclic factor=PACK_FLOATS dim=1

v21_norm_load_pack:
    for (int h_pack = 0; h_pack < HIDDEN_PACKS; ++h_pack) {
#pragma HLS PIPELINE II=1
        float gamma_lane[PACK_FLOATS];
        float beta_lane[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=gamma_lane complete dim=1
#pragma HLS ARRAY_PARTITION variable=beta_lane complete dim=1
        unpack_packed_float16(gamma[h_pack], gamma_lane);
        unpack_packed_float16(beta[h_pack], beta_lane);
    v21_norm_load_lane:
        for (int lane = 0; lane < PACK_FLOATS; ++lane) {
#pragma HLS UNROLL
            const int h = h_pack * PACK_FLOATS + lane;
            gamma_cache[h] = gamma_lane[lane];
            beta_cache[h] = beta_lane[lane];
        }
    }

v21_norm_row:
    for (int s = 0; s < SEQ_LEN; ++s) {
        float sum_bank[PACK_FLOATS];
        float square_bank[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=sum_bank complete dim=1
#pragma HLS ARRAY_PARTITION variable=square_bank complete dim=1
    v21_norm_bank_init:
        for (int lane = 0; lane < PACK_FLOATS; ++lane) {
#pragma HLS UNROLL
            sum_bank[lane] = 0.0f;
            square_bank[lane] = 0.0f;
        }

    v21_norm_reduce_pack:
        for (int h_pack = 0; h_pack < HIDDEN_PACKS; ++h_pack) {
#pragma HLS PIPELINE II=1
        v21_norm_reduce_lane:
            for (int lane = 0; lane < PACK_FLOATS; ++lane) {
#pragma HLS UNROLL
                const float value =
                    output_buf[s][h_pack * PACK_FLOATS + lane];
                sum_bank[lane] += value;
                square_bank[lane] += value * value;
            }
        }

        float sum = 0.0f;
        float square_sum = 0.0f;
    v21_norm_bank_reduce:
        for (int lane = 0; lane < PACK_FLOATS; ++lane) {
            sum += sum_bank[lane];
            square_sum += square_bank[lane];
        }
        const float mean = sum * (1.0f / 768.0f);
        const float variance =
            square_sum * (1.0f / 768.0f) - mean * mean;
        const float inv_std = hls::rsqrtf(variance + 1.0e-12f);

    v21_norm_write_pack:
        for (int h_pack = 0; h_pack < HIDDEN_PACKS; ++h_pack) {
#pragma HLS PIPELINE II=1
            float lane_value[PACK_FLOATS];
#pragma HLS ARRAY_PARTITION variable=lane_value complete dim=1
        v21_norm_write_lane:
            for (int lane = 0; lane < PACK_FLOATS; ++lane) {
#pragma HLS UNROLL
                const int h = h_pack * PACK_FLOATS + lane;
                lane_value[lane] =
                    (output_buf[s][h] - mean) * inv_std * gamma_cache[h]
                    + beta_cache[h];
            }
            output_hidden[s * HIDDEN_PACKS + h_pack] =
                pack_packed_float16(lane_value);
        }
    }
}

static void v15_projection_dataflow_tilemajor(
    float input_buf[SEQ_LEN][HIDDEN_SIZE],
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float16_t *w2,
    packed_float16_t *b2,
    float output_buf[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
    packed_float_stream_t gelu_stream("v15_gelu_stream");
#pragma HLS STREAM variable=gelu_stream depth=512
#pragma HLS DATAFLOW
    v15_ffn_up_tilemajor_producer(input_buf, w1, b1, gelu_stream);
    v15_initialize_down_bias(b2, output_buf);
    v15_ffn_down_tilemajor_consumer(gelu_stream, w2, output_buf);
}

extern "C" {

void bert_ffn_up_gelu_v21_dotpipe_kernel(
    packed_float_stream_t &attn_mid_stream,
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float_stream_t &gelu_stream
) {
#pragma HLS INTERFACE axis port=attn_mid_stream
#pragma HLS INTERFACE axis port=gelu_stream
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float input_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=input_buf type=RAM_1P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=input_buf cyclic factor=16 dim=2

    v11_load_input_stream_matrix(attn_mid_stream, input_buf);
    v15_ffn_up_tilemajor_producer(input_buf, w1, b1, gelu_stream);
}

void bert_ffn_down_v21_dotpipe_kernel(
    packed_float_stream_t &gelu_stream,
    packed_float16_t *w2,
    packed_float16_t *b2,
    packed_float16_t *output_hidden
) {
#pragma HLS INTERFACE axis port=gelu_stream
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float output_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=output_buf type=RAM_2P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=output_buf cyclic factor=16 dim=2

    v15_initialize_down_bias(b2, output_buf);
    v15_ffn_down_tilemajor_consumer(gelu_stream, w2, output_buf);
    v10_store_output_matrix(output_buf, output_hidden);
}

void bert_ffn_kernel_v21_dotpipe_estimate(
    packed_float16_t *input_hidden,
    packed_float16_t *w1,
    packed_float16_t *b1,
    packed_float16_t *w2,
    packed_float16_t *b2,
    packed_float16_t *output_hidden
) {
#pragma HLS INTERFACE m_axi port=input_hidden offset=slave bundle=gmem_in
#pragma HLS INTERFACE m_axi port=w1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=b1 offset=slave bundle=gmem_w1
#pragma HLS INTERFACE m_axi port=w2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=b2 offset=slave bundle=gmem_w2
#pragma HLS INTERFACE m_axi port=output_hidden offset=slave bundle=gmem_out
#pragma HLS INTERFACE s_axilite port=input_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=w1 bundle=control
#pragma HLS INTERFACE s_axilite port=b1 bundle=control
#pragma HLS INTERFACE s_axilite port=w2 bundle=control
#pragma HLS INTERFACE s_axilite port=b2 bundle=control
#pragma HLS INTERFACE s_axilite port=output_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float input_buf[SEQ_LEN][HIDDEN_SIZE];
    static float output_buf[SEQ_LEN][HIDDEN_SIZE];

#pragma HLS BIND_STORAGE variable=input_buf type=RAM_1P impl=URAM latency=2
#pragma HLS BIND_STORAGE variable=output_buf type=RAM_2P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=input_buf cyclic factor=16 dim=2
#pragma HLS ARRAY_PARTITION variable=output_buf cyclic factor=16 dim=2

    v10_load_input_matrix(input_hidden, input_buf);
    v15_projection_dataflow_tilemajor(input_buf, w1, b1, w2, b2, output_buf);
    v10_store_output_matrix(output_buf, output_hidden);
}

void bert_ffn_up_gelu_kernel(
    bert_cmd_stream_t &layer_cmd,
    packed_float_stream_t &attn_mid_stream,
    packed_float16_t *w1_all,
    packed_float16_t *b1_all,
    packed_float_stream_t &gelu_stream)
{
#pragma HLS INTERFACE axis port=layer_cmd
#pragma HLS INTERFACE axis port=attn_mid_stream
#pragma HLS INTERFACE axis port=gelu_stream
#pragma HLS INTERFACE m_axi port=w1_all offset=slave bundle=gmem_w1 depth=1769472
#pragma HLS INTERFACE m_axi port=b1_all offset=slave bundle=gmem_w1 depth=2304
#pragma HLS INTERFACE s_axilite port=w1_all bundle=control
#pragma HLS INTERFACE s_axilite port=b1_all bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float input_buf[SEQ_LEN][HIDDEN_SIZE];
#pragma HLS BIND_STORAGE variable=input_buf type=RAM_1P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=input_buf cyclic factor=16 dim=2

v21_up_layer:
    for (int command_index = 0; command_index < BERT_NUM_LAYERS; ++command_index) {
#pragma HLS LOOP_TRIPCOUNT min=12 max=12
#pragma HLS UNROLL factor=1
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE off
        const bert_layer_cmd_t cmd = layer_cmd.read();
        const int layer_id = (int)bert_cmd_layer(cmd);
        v11_load_input_stream_matrix(attn_mid_stream, input_buf);
        v15_ffn_up_tilemajor_producer(
            input_buf,
            w1_all + layer_id * FFN_UP_WEIGHT_WORDS,
            b1_all + layer_id * FFN_UP_BIAS_WORDS,
            gelu_stream);
    }
}

void bert_ffn_down_residual_norm_kernel(
    bert_cmd_stream_t &layer_cmd,
    packed_float_stream_t &gelu_stream,
    packed_float_stream_t &ffn_residual_stream,
    packed_float16_t *w2_all,
    packed_float16_t *b2_all,
    packed_float16_t *final_norm_gamma_all,
    packed_float16_t *final_norm_beta_all,
    packed_float16_t *hidden_slots,
    bert_cmd_stream_t &layer_done)
{
#pragma HLS INTERFACE axis port=layer_cmd
#pragma HLS INTERFACE axis port=gelu_stream
#pragma HLS INTERFACE axis port=ffn_residual_stream
#pragma HLS INTERFACE axis port=layer_done
#pragma HLS INTERFACE m_axi port=w2_all offset=slave bundle=gmem_w2 depth=1769472
#pragma HLS INTERFACE m_axi port=b2_all offset=slave bundle=gmem_w2 depth=576
#pragma HLS INTERFACE m_axi port=final_norm_gamma_all offset=slave bundle=gmem_w2 depth=576
#pragma HLS INTERFACE m_axi port=final_norm_beta_all offset=slave bundle=gmem_w2 depth=576
#pragma HLS INTERFACE m_axi port=hidden_slots offset=slave bundle=gmem_out depth=12288
#pragma HLS INTERFACE s_axilite port=w2_all bundle=control
#pragma HLS INTERFACE s_axilite port=b2_all bundle=control
#pragma HLS INTERFACE s_axilite port=final_norm_gamma_all bundle=control
#pragma HLS INTERFACE s_axilite port=final_norm_beta_all bundle=control
#pragma HLS INTERFACE s_axilite port=hidden_slots bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    static float output_buf[SEQ_LEN][HIDDEN_SIZE];
#pragma HLS BIND_STORAGE variable=output_buf type=RAM_2P impl=URAM latency=2
#pragma HLS ARRAY_PARTITION variable=output_buf cyclic factor=16 dim=2

v21_down_layer:
    for (int command_index = 0; command_index < BERT_NUM_LAYERS; ++command_index) {
#pragma HLS LOOP_TRIPCOUNT min=12 max=12
#pragma HLS UNROLL factor=1
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE off
        const bert_layer_cmd_t cmd = layer_cmd.read();
        const int layer_id = (int)bert_cmd_layer(cmd);
        const int write_slot = (int)bert_cmd_write_slot(cmd);
        packed_float16_t *layer_output =
            hidden_slots + write_slot * BERT_HIDDEN_WORDS;

        // Drain the residual stream before projection.  Bias + residual is
        // written directly into the existing DOWN accumulator URAM, so no
        // second full hidden tensor buffer is introduced on SLR3.
        v21_initialize_down_bias_residual(
            ffn_residual_stream,
            b2_all + layer_id * FFN_DOWN_BIAS_WORDS,
            output_buf);
        v15_ffn_down_tilemajor_consumer(
            gelu_stream,
            w2_all + layer_id * FFN_DOWN_WEIGHT_WORDS,
            output_buf);
        v21_layernorm_and_store(
            output_buf,
            final_norm_gamma_all + layer_id * HIDDEN_PACKS,
            final_norm_beta_all + layer_id * HIDDEN_PACKS,
            layer_output);

        // Read-after-write on the same AXI master is the visibility fence for
        // the DDR ping-pong slot before the narrow completion token returns.
        volatile packed_float16_t visible =
            layer_output[BERT_HIDDEN_WORDS - 1];
        (void)visible;
        layer_done.write((unsigned int)(layer_id + 1));
    }
}

}
