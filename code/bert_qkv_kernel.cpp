#include "bert_kernel_interfaces.h"
#include "bert_math.h"

// Keep the eight recurrence lanes as named fields. Vitis HLS 2022.1 does not
// reliably disambiguate a fully-partitioned array accessed with a dynamic bank
// index; it scalarizes the array and retains a false distance-one dependence.
struct qkv_acc8_t {
    float b0;
    float b1;
    float b2;
    float b3;
    float b4;
    float b5;
    float b6;
    float b7;
};

static inline void qkv_acc8_clear(qkv_acc8_t &acc)
{
#pragma HLS INLINE
    acc.b0 = 0.0f;
    acc.b1 = 0.0f;
    acc.b2 = 0.0f;
    acc.b3 = 0.0f;
    acc.b4 = 0.0f;
    acc.b5 = 0.0f;
    acc.b6 = 0.0f;
    acc.b7 = 0.0f;
}

static inline void qkv_acc8_rotate_add(qkv_acc8_t &acc, float value)
{
#pragma HLS INLINE
    const float updated = acc.b0 + value;
    acc.b0 = acc.b1;
    acc.b1 = acc.b2;
    acc.b2 = acc.b3;
    acc.b3 = acc.b4;
    acc.b4 = acc.b5;
    acc.b5 = acc.b6;
    acc.b6 = acc.b7;
    acc.b7 = updated;
}

// Keep the one accumulator that Vitis HLS 2022.1 otherwise maps to a
// latency-four fabric adder in its own scheduled hierarchy.  The seven-cycle
// full-DSP adder is shorter than the true eight-iteration recurrence distance,
// so PROJECT_INPUT_STEP can retain II=1.  This boundary is intentionally not
// inline: earlier inline/direct BIND_OP variants were recorded and then
// discarded during operator sharing.
static float qkv_last_v_accumulate(float current, float value)
{
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    float updated;
#pragma HLS BIND_OP variable=updated op=fadd impl=fulldsp latency=7
    updated = current + value;
    return updated;
}

static inline void qkv_acc8_read(const qkv_acc8_t &acc, float lane[8])
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=lane complete dim=1
    lane[0] = acc.b0;
    lane[1] = acc.b1;
    lane[2] = acc.b2;
    lane[3] = acc.b3;
    lane[4] = acc.b4;
    lane[5] = acc.b5;
    lane[6] = acc.b6;
    lane[7] = acc.b7;
}

static void load_hidden_matrix(
    const bus_t *hidden_in,
    float hidden[SEQ_LEN][HIDDEN_SIZE])
{
#pragma HLS INLINE off
LOAD_HIDDEN_ROW:
    for (int s = 0; s < SEQ_LEN; ++s) {
    LOAD_HIDDEN_PACK:
        for (int p = 0; p < PACKS; ++p) {
#pragma HLS PIPELINE II=1
            float lane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=lane complete dim=1
            unpack_bus16(hidden_in[s * PACKS + p], lane);
        LOAD_HIDDEN_LANE:
            for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
                hidden[s][p * PACK_SIZE + i] = lane[i];
            }
        }
    }
}

static void load_qkv_bias64(
    const bus_t *q_bias,
    const bus_t *k_bias,
    const bus_t *v_bias,
    int out_base,
    float qb[QKV_TILE_O],
    float kb[QKV_TILE_O],
    float vb[QKV_TILE_O])
{
#pragma HLS INLINE off
LOAD_BIAS_PACK:
    for (int p = 0; p < QKV_TILE_O / PACK_SIZE; ++p) {
#pragma HLS PIPELINE II=1
        float qlane[PACK_SIZE];
        float klane[PACK_SIZE];
        float vlane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=qlane complete dim=1
#pragma HLS ARRAY_PARTITION variable=klane complete dim=1
#pragma HLS ARRAY_PARTITION variable=vlane complete dim=1
        const int word = out_base / PACK_SIZE + p;
        unpack_bus16(q_bias[word], qlane);
        unpack_bus16(k_bias[word], klane);
        unpack_bus16(v_bias[word], vlane);
    LOAD_BIAS_LANE:
        for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
            qb[p * PACK_SIZE + i] = qlane[i];
            kb[p * PACK_SIZE + i] = klane[i];
            vb[p * PACK_SIZE + i] = vlane[i];
        }
    }
}

static void load_qkv_weight_tile(
    const bus_t *q_weight,
    const bus_t *k_weight,
    const bus_t *v_weight,
    int out_tile,
    float wq[QKV_TILE_O][HIDDEN_SIZE],
    float wk[QKV_TILE_O][HIDDEN_SIZE],
    float wv[QKV_TILE_O][HIDDEN_SIZE])
{
#pragma HLS INLINE off
LOAD_WEIGHT_PACK:
    for (int ip = 0; ip < PACKS; ++ip) {
    LOAD_WEIGHT_OUTPUT:
        for (int o = 0; o < QKV_TILE_O; ++o) {
#pragma HLS PIPELINE II=1
            const int word = (out_tile * PACKS + ip) * QKV_TILE_O + o;
            const bus_t qword = q_weight[word];
            const bus_t kword = k_weight[word];
            const bus_t vword = v_weight[word];
        LOAD_WEIGHT_LANE:
            for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
                const int h = ip * PACK_SIZE + i;
                wq[o][h] = uint32_to_float(
                    (uint32_t)qword.range(31 + i * 32, i * 32));
                wk[o][h] = uint32_to_float(
                    (uint32_t)kword.range(31 + i * 32, i * 32));
                wv[o][h] = uint32_to_float(
                    (uint32_t)vword.range(31 + i * 32, i * 32));
            }
        }
    }
}

static void project_qkv_group(
    float hidden[SEQ_LEN][HIDDEN_SIZE],
    const bus_t *q_weight,
    const bus_t *q_bias,
    const bus_t *k_weight,
    const bus_t *k_bias,
    const bus_t *v_weight,
    const bus_t *v_bias,
    int group,
    float q_group[SEQ_LEN][GROUP_DIM],
    float k_group[SEQ_LEN][GROUP_DIM],
    float v_group[SEQ_LEN][GROUP_DIM])
{
#pragma HLS INLINE off
    float wq[QKV_TILE_O][HIDDEN_SIZE];
    float wk[QKV_TILE_O][HIDDEN_SIZE];
    float wv[QKV_TILE_O][HIDDEN_SIZE];
    float qb[QKV_TILE_O];
    float kb[QKV_TILE_O];
    float vb[QKV_TILE_O];
#pragma HLS BIND_STORAGE variable=wq type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=wk type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=wv type=RAM_2P impl=URAM
#pragma HLS ARRAY_PARTITION variable=wq cyclic factor=QKV_O_PAR dim=1
#pragma HLS ARRAY_PARTITION variable=wk cyclic factor=QKV_O_PAR dim=1
#pragma HLS ARRAY_PARTITION variable=wv cyclic factor=QKV_O_PAR dim=1
#pragma HLS ARRAY_PARTITION variable=wq cyclic factor=QKV_K_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=wk cyclic factor=QKV_K_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=wv cyclic factor=QKV_K_PAR dim=2
#pragma HLS ARRAY_PARTITION variable=qb cyclic factor=QKV_O_PAR dim=1
#pragma HLS ARRAY_PARTITION variable=kb cyclic factor=QKV_O_PAR dim=1
#pragma HLS ARRAY_PARTITION variable=vb cyclic factor=QKV_O_PAR dim=1

GROUP_OUTPUT_TILE:
    for (int local_base = 0; local_base < GROUP_DIM; local_base += QKV_TILE_O) {
        const int out_base = group * GROUP_DIM + local_base;
        const int out_tile = out_base / QKV_TILE_O;
        load_qkv_bias64(q_bias, k_bias, v_bias, out_base, qb, kb, vb);
        load_qkv_weight_tile(q_weight, k_weight, v_weight, out_tile, wq, wk, wv);

    PROJECT_ROW:
        for (int s = 0; s < SEQ_LEN; ++s) {
        PROJECT_OUTPUT_BLOCK:
            for (int ob = 0; ob < QKV_TILE_O; ob += QKV_O_PAR) {
                qkv_acc8_t qpartial[QKV_O_PAR][QKV_K_PAR];
                qkv_acc8_t kpartial[QKV_O_PAR][QKV_K_PAR];
                qkv_acc8_t vpartial[QKV_O_PAR][QKV_K_PAR];
                float qproduct[QKV_O_PAR][QKV_K_PAR];
                float kproduct[QKV_O_PAR][QKV_K_PAR];
                float vproduct[QKV_O_PAR][QKV_K_PAR];
#pragma HLS ARRAY_PARTITION variable=qpartial complete dim=1
#pragma HLS ARRAY_PARTITION variable=qpartial complete dim=2
#pragma HLS ARRAY_PARTITION variable=kpartial complete dim=1
#pragma HLS ARRAY_PARTITION variable=kpartial complete dim=2
#pragma HLS ARRAY_PARTITION variable=vpartial complete dim=1
#pragma HLS ARRAY_PARTITION variable=vpartial complete dim=2
#pragma HLS ARRAY_PARTITION variable=qproduct complete dim=1
#pragma HLS ARRAY_PARTITION variable=qproduct complete dim=2
#pragma HLS ARRAY_PARTITION variable=kproduct complete dim=1
#pragma HLS ARRAY_PARTITION variable=kproduct complete dim=2
#pragma HLS ARRAY_PARTITION variable=vproduct complete dim=1
#pragma HLS ARRAY_PARTITION variable=vproduct complete dim=2
#pragma HLS BIND_OP variable=qproduct op=fmul impl=maxdsp
#pragma HLS BIND_OP variable=kproduct op=fmul impl=maxdsp
#pragma HLS BIND_OP variable=vproduct op=fmul impl=maxdsp

            INIT_PARTIAL_OUTPUT:
                for (int j = 0; j < QKV_O_PAR; ++j) {
#pragma HLS UNROLL
                INIT_PARTIAL_K:
                    for (int k = 0; k < QKV_K_PAR; ++k) {
#pragma HLS UNROLL
                        qkv_acc8_clear(qpartial[j][k]);
                        qkv_acc8_clear(kpartial[j][k]);
                        qkv_acc8_clear(vpartial[j][k]);
                    }
                }

            PROJECT_INPUT_STEP:
                for (int step = 0;
                     step < PACKS * (PACK_SIZE / QKV_K_PAR);
                     ++step) {
#pragma HLS PIPELINE II=1
                    const int ip = step / (PACK_SIZE / QKV_K_PAR);
                    const int half = step % (PACK_SIZE / QKV_K_PAR);
                    const int input_base = ip * PACK_SIZE + half * QKV_K_PAR;
                PROJECT_PRODUCT_OUTPUT:
                    for (int j = 0; j < QKV_O_PAR; ++j) {
#pragma HLS UNROLL
                    PROJECT_PRODUCT_K:
                        for (int k = 0; k < QKV_K_PAR; ++k) {
#pragma HLS UNROLL
                            const float x = hidden[s][input_base + k];
                            qproduct[j][k] = x * wq[ob + j][input_base + k];
                            kproduct[j][k] = x * wk[ob + j][input_base + k];
                            vproduct[j][k] = x * wv[ob + j][input_base + k];
                            qkv_acc8_rotate_add(qpartial[j][k], qproduct[j][k]);
                            qkv_acc8_rotate_add(kpartial[j][k], kproduct[j][k]);
                            if (j != QKV_O_PAR - 1 || k != QKV_K_PAR - 1) {
                                qkv_acc8_rotate_add(
                                    vpartial[j][k], vproduct[j][k]);
                            }
                        }
                    }
                    const float last_v_updated = qkv_last_v_accumulate(
                        vpartial[QKV_O_PAR - 1][QKV_K_PAR - 1].b0,
                        vproduct[QKV_O_PAR - 1][QKV_K_PAR - 1]);
                    vpartial[QKV_O_PAR - 1][QKV_K_PAR - 1].b0 =
                        vpartial[QKV_O_PAR - 1][QKV_K_PAR - 1].b1;
                    vpartial[QKV_O_PAR - 1][QKV_K_PAR - 1].b1 =
                        vpartial[QKV_O_PAR - 1][QKV_K_PAR - 1].b2;
                    vpartial[QKV_O_PAR - 1][QKV_K_PAR - 1].b2 =
                        vpartial[QKV_O_PAR - 1][QKV_K_PAR - 1].b3;
                    vpartial[QKV_O_PAR - 1][QKV_K_PAR - 1].b3 =
                        vpartial[QKV_O_PAR - 1][QKV_K_PAR - 1].b4;
                    vpartial[QKV_O_PAR - 1][QKV_K_PAR - 1].b4 =
                        vpartial[QKV_O_PAR - 1][QKV_K_PAR - 1].b5;
                    vpartial[QKV_O_PAR - 1][QKV_K_PAR - 1].b5 =
                        vpartial[QKV_O_PAR - 1][QKV_K_PAR - 1].b6;
                    vpartial[QKV_O_PAR - 1][QKV_K_PAR - 1].b6 =
                        vpartial[QKV_O_PAR - 1][QKV_K_PAR - 1].b7;
                    vpartial[QKV_O_PAR - 1][QKV_K_PAR - 1].b7 =
                        last_v_updated;
                }

            REDUCE_OUTPUT:
                for (int j = 0; j < QKV_O_PAR; ++j) {
#pragma HLS PIPELINE II=1
                    float qlane[QKV_K_PAR];
                    float klane[QKV_K_PAR];
                    float vlane[QKV_K_PAR];
#pragma HLS ARRAY_PARTITION variable=qlane complete dim=1
#pragma HLS ARRAY_PARTITION variable=klane complete dim=1
#pragma HLS ARRAY_PARTITION variable=vlane complete dim=1
                REDUCE_K:
                    for (int k = 0; k < QKV_K_PAR; ++k) {
#pragma HLS UNROLL
                        float qbank[QKV_ACC_BANKS];
                        float kbank[QKV_ACC_BANKS];
                        float vbank[QKV_ACC_BANKS];
#pragma HLS ARRAY_PARTITION variable=qbank complete dim=1
#pragma HLS ARRAY_PARTITION variable=kbank complete dim=1
#pragma HLS ARRAY_PARTITION variable=vbank complete dim=1
                        qkv_acc8_read(qpartial[j][k], qbank);
                        qkv_acc8_read(kpartial[j][k], kbank);
                        qkv_acc8_read(vpartial[j][k], vbank);
                        qlane[k] = fp32_sum8_tree(qbank);
                        klane[k] = fp32_sum8_tree(kbank);
                        vlane[k] = fp32_sum8_tree(vbank);
                    }
                    q_group[s][local_base + ob + j] =
                        qb[ob + j] + fp32_sum8_tree(qlane);
                    k_group[s][local_base + ob + j] =
                        kb[ob + j] + fp32_sum8_tree(klane);
                    v_group[s][local_base + ob + j] =
                        vb[ob + j] + fp32_sum8_tree(vlane);
                }
            }
        }
    }
}

static void write_qkv_group(
    float q_group[SEQ_LEN][GROUP_DIM],
    float k_group[SEQ_LEN][GROUP_DIM],
    float v_group[SEQ_LEN][GROUP_DIM],
    hidden_stream_t &q_stream,
    hidden_stream_t &k_stream,
    hidden_stream_t &v_stream)
{
#pragma HLS INLINE off
WRITE_GROUP_ROW:
    for (int s = 0; s < SEQ_LEN; ++s) {
    WRITE_GROUP_PACK:
        for (int p = 0; p < GROUP_PACKS; ++p) {
#pragma HLS PIPELINE II=1
            float qlane[PACK_SIZE];
            float klane[PACK_SIZE];
            float vlane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=qlane complete dim=1
#pragma HLS ARRAY_PARTITION variable=klane complete dim=1
#pragma HLS ARRAY_PARTITION variable=vlane complete dim=1
        WRITE_GROUP_LANE:
            for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
                qlane[i] = q_group[s][p * PACK_SIZE + i];
                klane[i] = k_group[s][p * PACK_SIZE + i];
                vlane[i] = v_group[s][p * PACK_SIZE + i];
            }
            q_stream.write(pack_bus16(qlane));
            k_stream.write(pack_bus16(klane));
            v_stream.write(pack_bus16(vlane));
        }
    }
}

extern "C" void bert_qkv_kernel(
    const bus_t *hidden_slots,
    const qkv_weight_bus_t *attn_q_w_all,
    const bus_t *attn_q_b_all,
    const qkv_weight_bus_t *attn_k_w_all,
    const bus_t *attn_k_b_all,
    const qkv_weight_bus_t *attn_v_w_all,
    const bus_t *attn_v_b_all,
    bert_cmd_stream_t &layer_cmd,
    hidden_stream_t &q_stream,
    hidden_stream_t &k_stream,
    hidden_stream_t &v_stream)
{
#pragma HLS INTERFACE m_axi port=hidden_slots offset=slave bundle=gmem_hidden depth=12288
#pragma HLS INTERFACE m_axi port=attn_q_w_all offset=slave bundle=gmem_q depth=442368
#pragma HLS INTERFACE m_axi port=attn_q_b_all offset=slave bundle=gmem_q depth=576
#pragma HLS INTERFACE m_axi port=attn_k_w_all offset=slave bundle=gmem_k depth=442368
#pragma HLS INTERFACE m_axi port=attn_k_b_all offset=slave bundle=gmem_k depth=576
#pragma HLS INTERFACE m_axi port=attn_v_w_all offset=slave bundle=gmem_v depth=442368
#pragma HLS INTERFACE m_axi port=attn_v_b_all offset=slave bundle=gmem_v depth=576
#pragma HLS INTERFACE axis port=layer_cmd
#pragma HLS INTERFACE axis port=q_stream
#pragma HLS INTERFACE axis port=k_stream
#pragma HLS INTERFACE axis port=v_stream
#pragma HLS INTERFACE s_axilite port=hidden_slots bundle=control
#pragma HLS INTERFACE s_axilite port=attn_q_w_all bundle=control
#pragma HLS INTERFACE s_axilite port=attn_q_b_all bundle=control
#pragma HLS INTERFACE s_axilite port=attn_k_w_all bundle=control
#pragma HLS INTERFACE s_axilite port=attn_k_b_all bundle=control
#pragma HLS INTERFACE s_axilite port=attn_v_w_all bundle=control
#pragma HLS INTERFACE s_axilite port=attn_v_b_all bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    float hidden[SEQ_LEN][HIDDEN_SIZE];
    float q_group[SEQ_LEN][GROUP_DIM];
    float k_group[SEQ_LEN][GROUP_DIM];
    float v_group[SEQ_LEN][GROUP_DIM];
#pragma HLS BIND_STORAGE variable=hidden type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=q_group type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=k_group type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=v_group type=RAM_2P impl=URAM
#pragma HLS ARRAY_RESHAPE variable=hidden cyclic factor=QKV_K_PAR dim=2
#pragma HLS ARRAY_RESHAPE variable=q_group cyclic factor=QKV_O_PAR dim=2
#pragma HLS ARRAY_RESHAPE variable=k_group cyclic factor=QKV_O_PAR dim=2
#pragma HLS ARRAY_RESHAPE variable=v_group cyclic factor=QKV_O_PAR dim=2

QKV_LAYER:
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

        load_hidden_matrix(
            hidden_slots + read_slot * BERT_HIDDEN_WORDS, hidden);

    HEAD_GROUP:
        for (int group = 0; group < HEAD_GROUPS; ++group) {
            project_qkv_group(
                hidden,
                attn_q_w_all + weight_offset, attn_q_b_all + bias_offset,
                attn_k_w_all + weight_offset, attn_k_b_all + bias_offset,
                attn_v_w_all + weight_offset, attn_v_b_all + bias_offset,
                group, q_group, k_group, v_group);
            write_qkv_group(
                q_group, k_group, v_group, q_stream, k_stream, v_stream);
        }
    }
}
