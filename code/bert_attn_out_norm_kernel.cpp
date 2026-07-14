#include "bert_kernel_interfaces.h"
#include "bert_math.h"
#include <hls_math.h>

static void load_residual(
    const bus_t *residual_in,
    float residual[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
LOAD_RESIDUAL_ROW:
    for (int s = 0; s < SEQ_LEN; ++s) {
    LOAD_RESIDUAL_PACK:
        for (int p = 0; p < PACKS; ++p) {
#pragma HLS PIPELINE II=1
            float rlane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=rlane complete dim=1
            unpack_bus16(residual_in[s * PACKS + p], rlane);
        LOAD_RESIDUAL_LANE:
            for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
                residual[s][p * PACK_SIZE + i] = rlane[i];
            }
        }
    }
}

static void read_context_group(
    hidden_stream_t &context_stream,
    float context_group[SEQ_LEN][GROUP_DIM])
{
#pragma HLS INLINE off
READ_CONTEXT_GROUP_ROW:
    for (int s = 0; s < SEQ_LEN; ++s) {
    READ_CONTEXT_GROUP_PACK:
        for (int p = 0; p < GROUP_PACKS; ++p) {
#pragma HLS PIPELINE II=1
            float lane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=lane complete dim=1
            unpack_bus16(context_stream.read(), lane);
        READ_CONTEXT_GROUP_LANE:
            for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
                context_group[s][p * PACK_SIZE + i] = lane[i];
            }
        }
    }
}

static void load_output_bias32(
    const bus_t *bias,
    int out_base,
    float bias_tile[OUT_TILE_O])
{
#pragma HLS INLINE off
LOAD_OUT_BIAS_PACK:
    for (int p = 0; p < OUT_TILE_O / PACK_SIZE; ++p) {
#pragma HLS PIPELINE II=1
        float lane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=lane complete dim=1
        unpack_bus16(bias[out_base / PACK_SIZE + p], lane);
    LOAD_OUT_BIAS_LANE:
        for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
            bias_tile[p * PACK_SIZE + i] = lane[i];
        }
    }
}

static void initialize_projection(
    const bus_t *bias,
    float projected[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
    static const int OUTPUT_TILES = HIDDEN_SIZE / OUT_TILE_O;
    float bias_tile[OUT_TILE_O];
#pragma HLS ARRAY_PARTITION variable=bias_tile cyclic factor=OUT_O_PAR dim=1

INIT_PROJECTION_TILE:
    for (int tile = 0; tile < OUTPUT_TILES; ++tile) {
        const int out_base = tile * OUT_TILE_O;
        load_output_bias32(bias, out_base, bias_tile);
    INIT_PROJECTION_ROW:
        for (int s = 0; s < SEQ_LEN; ++s) {
        INIT_PROJECTION_OUT:
            for (int o = 0; o < OUT_TILE_O; o += OUT_O_PAR) {
#pragma HLS PIPELINE II=1
            INIT_PROJECTION_LANE:
                for (int j = 0; j < OUT_O_PAR; ++j) {
#pragma HLS UNROLL
                    projected[s][out_base + o + j] = bias_tile[o + j];
                }
            }
        }
    }
}

static void accumulate_projection_group(
    float context_group[SEQ_LEN][GROUP_DIM],
    const bus_t *weight,
    int group,
    float projected[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
    static const int OUTPUT_TILES = HIDDEN_SIZE / OUT_TILE_O;
    float wtile[OUT_TILE_O][PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=wtile cyclic factor=OUT_O_PAR dim=1
#pragma HLS ARRAY_PARTITION variable=wtile cyclic factor=OUT_K_PAR dim=2

ACCUMULATE_OUTPUT_TILE:
    for (int tile = 0; tile < OUTPUT_TILES; ++tile) {
        const int out_base = tile * OUT_TILE_O;
    ACCUMULATE_INPUT_PACK:
        for (int ip = 0; ip < GROUP_PACKS; ++ip) {
            LOAD_OUT_WEIGHT:
            for (int o = 0; o < OUT_TILE_O; ++o) {
#pragma HLS PIPELINE II=1
                const int word =
                    ((group * OUTPUT_TILES + tile) * GROUP_PACKS + ip) *
                    OUT_TILE_O + o;
                const bus_t packed_weight = weight[word];
            LOAD_OUT_WEIGHT_LANE:
                for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
                    wtile[o][i] = uint32_to_float(
                        (uint32_t)packed_weight.range(31 + i * 32, i * 32));
                }
            }
        ACCUMULATE_ROW_BLOCK:
            for (int rb = 0; rb < SEQ_LEN * (OUT_TILE_O / OUT_O_PAR); ++rb) {
#pragma HLS PIPELINE II=2
#pragma HLS DEPENDENCE variable=projected inter false
                const int s = rb / (OUT_TILE_O / OUT_O_PAR);
                const int out_block = (rb % (OUT_TILE_O / OUT_O_PAR)) * OUT_O_PAR;
                float x[PACK_SIZE];
                float product[OUT_O_PAR][OUT_K_PAR];
                float partial[OUT_O_PAR][OUT_K_PAR];
#pragma HLS ARRAY_PARTITION variable=x complete dim=1
#pragma HLS ARRAY_PARTITION variable=product complete dim=1
#pragma HLS ARRAY_PARTITION variable=product complete dim=2
#pragma HLS ARRAY_PARTITION variable=partial complete dim=1
#pragma HLS ARRAY_PARTITION variable=partial complete dim=2
#pragma HLS BIND_OP variable=product op=fmul impl=maxdsp
            LOAD_OUT_X:
                for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
                    x[i] = context_group[s][ip * PACK_SIZE + i];
                }
            INIT_OUT_PARTIAL:
                for (int o = 0; o < OUT_O_PAR; ++o) {
#pragma HLS UNROLL
                INIT_OUT_PARTIAL_LANE:
                    for (int k = 0; k < OUT_K_PAR; ++k) {
#pragma HLS UNROLL
                        partial[o][k] = 0.0f;
                    }
                }
            OUT_DOT_HALF:
                for (int base = 0; base < PACK_SIZE; base += OUT_K_PAR) {
                OUT_DOT_OUTPUT:
                    for (int o = 0; o < OUT_O_PAR; ++o) {
#pragma HLS UNROLL
                    OUT_DOT_LANE:
                        for (int k = 0; k < OUT_K_PAR; ++k) {
#pragma HLS UNROLL
                            product[o][k] =
                                x[base + k] * wtile[out_block + o][base + k];
                            partial[o][k] += product[o][k];
                        }
                    }
                }
            ACCUMULATE_ALL_OUTPUTS:
                for (int o = 0; o < OUT_O_PAR; ++o) {
#pragma HLS UNROLL
                    const float s0 = partial[o][0] + partial[o][1];
                    const float s1 = partial[o][2] + partial[o][3];
                    const float s2 = partial[o][4] + partial[o][5];
                    const float s3 = partial[o][6] + partial[o][7];
                    const float dot = (s0 + s1) + (s2 + s3);
                    projected[s][out_base + out_block + o] += dot;
                }
            }
        }
    }
}

static void load_norm_parameters(
    const bus_t *gamma,
    const bus_t *beta,
    float gamma_cache[HIDDEN_SIZE],
    float beta_cache[HIDDEN_SIZE])
{
#pragma HLS INLINE off
LOAD_NORM_PACK:
    for (int p = 0; p < PACKS; ++p) {
#pragma HLS PIPELINE II=1
        float glane[PACK_SIZE];
        float blane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=glane complete dim=1
#pragma HLS ARRAY_PARTITION variable=blane complete dim=1
        unpack_bus16(gamma[p], glane);
        unpack_bus16(beta[p], blane);
    LOAD_NORM_LANE:
        for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
            gamma_cache[p * PACK_SIZE + i] = glane[i];
            beta_cache[p * PACK_SIZE + i] = blane[i];
        }
    }
}

static void residual_norm_and_write(
    float residual[SEQ_LEN][HIDDEN_SIZE],
    float projected[SEQ_LEN][HIDDEN_SIZE],
    const bus_t *gamma,
    const bus_t *beta,
    bus_t *attn_mid_ddr,
    hidden_stream_t &attn_mid_stream,
    hidden_stream_t &ffn_residual_stream)
{
#pragma HLS INLINE off
    float gamma_cache[HIDDEN_SIZE];
    float beta_cache[HIDDEN_SIZE];
#pragma HLS BIND_STORAGE variable=gamma_cache type=RAM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=beta_cache type=RAM_2P impl=BRAM
#pragma HLS ARRAY_RESHAPE variable=gamma_cache cyclic factor=PACK_SIZE dim=1
#pragma HLS ARRAY_RESHAPE variable=beta_cache cyclic factor=PACK_SIZE dim=1
    load_norm_parameters(gamma, beta, gamma_cache, beta_cache);

NORM_ROW:
    for (int s = 0; s < SEQ_LEN; ++s) {
        float sum_bank[PACK_SIZE];
        float square_bank[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=sum_bank complete dim=1
#pragma HLS ARRAY_PARTITION variable=square_bank complete dim=1
    NORM_BANK_INIT:
        for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
            sum_bank[i] = 0.0f;
            square_bank[i] = 0.0f;
        }
    NORM_REDUCE_PACK:
        for (int p = 0; p < PACKS; ++p) {
#pragma HLS PIPELINE II=1
        NORM_REDUCE_LANE:
            for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
                const int h = p * PACK_SIZE + i;
                const float x = residual[s][h] + projected[s][h];
                sum_bank[i] += x;
                square_bank[i] += x * x;
            }
        }
        float sum = 0.0f;
        float square_sum = 0.0f;
    NORM_BANK_REDUCE:
        for (int i = 0; i < PACK_SIZE; ++i) {
            sum += sum_bank[i];
            square_sum += square_bank[i];
        }
        const float mean = sum * (1.0f / 768.0f);
        const float variance = square_sum * (1.0f / 768.0f) - mean * mean;
        const float inv_std = hls::rsqrtf(variance + 1.0e-12f);

    NORM_WRITE_PACK:
        for (int p = 0; p < PACKS; ++p) {
#pragma HLS PIPELINE II=1
            float lane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=lane complete dim=1
        NORM_WRITE_LANE:
            for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
                const int h = p * PACK_SIZE + i;
                const float x = residual[s][h] + projected[s][h];
                lane[i] = (x - mean) * inv_std * gamma_cache[h] + beta_cache[h];
            }
            const bus_t word = pack_bus16(lane);
            attn_mid_ddr[s * PACKS + p] = word;
            attn_mid_stream.write(word);
            ffn_residual_stream.write(word);
        }
    }
}

extern "C" void bert_attn_out_norm_kernel(
    bert_cmd_stream_t &layer_cmd,
    hidden_stream_t &context_stream,
    const bus_t *hidden_slots,
    const bus_t *attn_o_w_all,
    const bus_t *attn_o_b_all,
    const bus_t *attn_norm_gamma_all,
    const bus_t *attn_norm_beta_all,
    bus_t *attn_mid_ddr,
    unsigned int *attn_mid_done,
    hidden_stream_t &attn_mid_stream,
    hidden_stream_t &ffn_residual_stream)
{
#pragma HLS INTERFACE axis port=layer_cmd
#pragma HLS INTERFACE axis port=context_stream
#pragma HLS INTERFACE axis port=attn_mid_stream
#pragma HLS INTERFACE axis port=ffn_residual_stream
#pragma HLS INTERFACE m_axi port=hidden_slots offset=slave bundle=gmem_hidden depth=12288
#pragma HLS INTERFACE m_axi port=attn_o_w_all offset=slave bundle=gmem_o depth=442368
#pragma HLS INTERFACE m_axi port=attn_o_b_all offset=slave bundle=gmem_o depth=576
#pragma HLS INTERFACE m_axi port=attn_norm_gamma_all offset=slave bundle=gmem_norm depth=576
#pragma HLS INTERFACE m_axi port=attn_norm_beta_all offset=slave bundle=gmem_norm depth=576
#pragma HLS INTERFACE m_axi port=attn_mid_ddr offset=slave bundle=gmem_attn_mid depth=6144
#pragma HLS INTERFACE m_axi port=attn_mid_done offset=slave bundle=gmem_attn_mid depth=1
#pragma HLS INTERFACE s_axilite port=hidden_slots bundle=control
#pragma HLS INTERFACE s_axilite port=attn_o_w_all bundle=control
#pragma HLS INTERFACE s_axilite port=attn_o_b_all bundle=control
#pragma HLS INTERFACE s_axilite port=attn_norm_gamma_all bundle=control
#pragma HLS INTERFACE s_axilite port=attn_norm_beta_all bundle=control
#pragma HLS INTERFACE s_axilite port=attn_mid_ddr bundle=control
#pragma HLS INTERFACE s_axilite port=attn_mid_done bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    float context_group[SEQ_LEN][GROUP_DIM];
    float residual[SEQ_LEN][HIDDEN_SIZE];
    float projected[SEQ_LEN][HIDDEN_SIZE];
#pragma HLS BIND_STORAGE variable=context_group type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=residual type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=projected type=RAM_2P impl=URAM
#pragma HLS ARRAY_RESHAPE variable=context_group cyclic factor=OUT_K_PAR dim=2
#pragma HLS ARRAY_RESHAPE variable=residual cyclic factor=PACK_SIZE dim=2
#pragma HLS ARRAY_RESHAPE variable=projected cyclic factor=OUT_O_PAR dim=2

OUT_NORM_LAYER:
    for (int command_index = 0; command_index < BERT_NUM_LAYERS; ++command_index) {
#pragma HLS LOOP_TRIPCOUNT min=12 max=12
#pragma HLS UNROLL factor=1
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE off
        const bert_layer_cmd_t cmd = layer_cmd.read();
        const int layer_id = (int)bert_cmd_layer(cmd);
        const int read_slot = (int)bert_cmd_read_slot(cmd);
        const int weight_offset = layer_id * ATTN_W_PACKS;
        const int bias_offset = layer_id * ATTN_B_PACKS;
        const int norm_offset = layer_id * PACKS;

        load_residual(
            hidden_slots + read_slot * BERT_HIDDEN_WORDS, residual);
        initialize_projection(attn_o_b_all + bias_offset, projected);
    PROJECT_CONTEXT_GROUP:
        for (int group = 0; group < HEAD_GROUPS; ++group) {
            read_context_group(context_stream, context_group);
            accumulate_projection_group(
                context_group, attn_o_w_all + weight_offset, group, projected);
        }
        residual_norm_and_write(
            residual, projected,
            attn_norm_gamma_all + norm_offset,
            attn_norm_beta_all + norm_offset,
            attn_mid_ddr, attn_mid_stream, ffn_residual_stream);
        attn_mid_done[0] = (unsigned int)(layer_id + 1);
    }
}
