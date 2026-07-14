#include "bert_kernel_interfaces.h"
#include "bert_math.h"
#include <hls_math.h>

struct core_acc8_t {
    float b0;
    float b1;
    float b2;
    float b3;
    float b4;
    float b5;
    float b6;
    float b7;
};

static inline void core_acc8_clear(core_acc8_t &acc)
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

static inline void core_acc8_rotate_add(core_acc8_t &acc, float value)
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

// Isolate the second EXP-row accumulator from the parent loop's register
// store.  Vitis HLS 2022.1 leaves one otherwise-identical accumulator in
// fabric and reports it as the -0.30 ns path.  A real non-inline hierarchy is
// required because earlier inline bindings were absorbed during scheduling.
// Seven cycles remain below the eight-iteration rotating-bank recurrence.
static float core_last_sum_accumulate(float current, float value)
{
#pragma HLS INLINE off
#pragma HLS PIPELINE II=1
    float updated;
#pragma HLS BIND_OP variable=updated op=fadd impl=fulldsp latency=7
    updated = current + value;
    return updated;
}

static inline void core_acc8_read_logical(
    const core_acc8_t &acc,
    int phase,
    float lane[8])
{
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable=lane complete dim=1
    float physical[8];
#pragma HLS ARRAY_PARTITION variable=physical complete dim=1
    physical[0] = acc.b0;
    physical[1] = acc.b1;
    physical[2] = acc.b2;
    physical[3] = acc.b3;
    physical[4] = acc.b4;
    physical[5] = acc.b5;
    physical[6] = acc.b6;
    physical[7] = acc.b7;
CORE_ACC_READ_LOGICAL:
    for (int r = 0; r < 8; ++r) {
#pragma HLS UNROLL
        lane[r] = physical[(r - phase) & 7];
    }
}

static void read_qkv_group(
    hidden_stream_t &q_stream,
    hidden_stream_t &k_stream,
    hidden_stream_t &v_stream,
    float q[SEQ_LEN][GROUP_DIM],
    float k[SEQ_LEN][GROUP_DIM],
    float v[SEQ_LEN][GROUP_DIM])
{
#pragma HLS INLINE off
READ_GROUP_ROW:
    for (int s = 0; s < SEQ_LEN; ++s) {
    READ_GROUP_PACK:
        for (int p = 0; p < GROUP_PACKS; ++p) {
#pragma HLS PIPELINE II=1
            float qlane[PACK_SIZE];
            float klane[PACK_SIZE];
            float vlane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=qlane complete dim=1
#pragma HLS ARRAY_PARTITION variable=klane complete dim=1
#pragma HLS ARRAY_PARTITION variable=vlane complete dim=1
            unpack_bus16(q_stream.read(), qlane);
            unpack_bus16(k_stream.read(), klane);
            unpack_bus16(v_stream.read(), vlane);
        READ_GROUP_LANE:
            for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
                q[s][p * PACK_SIZE + i] = qlane[i];
                k[s][p * PACK_SIZE + i] = klane[i];
                v[s][p * PACK_SIZE + i] = vlane[i];
            }
        }
    }
}

static void attention_dot_pair(
    float q[SEQ_LEN][GROUP_DIM],
    float k[SEQ_LEN][GROUP_DIM],
    int query0,
    int query1,
    int key,
    int head_base,
    float &dot0,
    float &dot1)
{
#pragma HLS INLINE
    float lane0[ATTN_D_PAR];
    float lane1[ATTN_D_PAR];
#pragma HLS ARRAY_PARTITION variable=lane0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane1 complete dim=1
DOT_INIT:
    for (int j = 0; j < ATTN_D_PAR; ++j) {
#pragma HLS UNROLL
        lane0[j] = 0.0f;
        lane1[j] = 0.0f;
    }
DOT_DIM:
    for (int d = 0; d < HEAD_DIM; d += ATTN_D_PAR) {
    DOT_LANE:
        for (int j = 0; j < ATTN_D_PAR; ++j) {
#pragma HLS UNROLL
            const float kval = k[key][head_base + d + j];
            lane0[j] += q[query0][head_base + d + j] * kval;
            lane1[j] += q[query1][head_base + d + j] * kval;
        }
    }
    dot0 = fp32_sum8_tree(lane0);
    dot1 = fp32_sum8_tree(lane1);
}

static void attention_group(
    float q[SEQ_LEN][GROUP_DIM],
    float k[SEQ_LEN][GROUP_DIM],
    float v[SEQ_LEN][GROUP_DIM],
    const int active_indices[SEQ_LEN],
    int active_count,
    float context[SEQ_LEN][GROUP_DIM])
{
#pragma HLS INLINE off
    const float scale = 0.125f;

HEAD:
    for (int head = 0; head < HEAD_PAR; ++head) {
#pragma HLS UNROLL
        const int head_base = head * HEAD_DIM;
    QUERY_PAIR:
        for (int query = 0; query < SEQ_LEN; query += 2) {
            float probability0[SEQ_LEN];
            float probability1[SEQ_LEN];
#pragma HLS BIND_STORAGE variable=probability0 type=RAM_2P impl=BRAM
#pragma HLS BIND_STORAGE variable=probability1 type=RAM_2P impl=BRAM
            float row_max0 = -3.402823466e+38F;
            float row_max1 = -3.402823466e+38F;

        SCORE_KEY:
            for (int n = 0; n < active_count; ++n) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
#pragma HLS PIPELINE II=1
                const int key = active_indices[n];
                float dot0;
                float dot1;
                attention_dot_pair(
                    q, k, query, query + 1, key, head_base, dot0, dot1);
                const float score0 = dot0 * scale;
                const float score1 = dot1 * scale;
                probability0[n] = score0;
                probability1[n] = score1;
                if (score0 > row_max0) {
                    row_max0 = score0;
                }
                if (score1 > row_max1) {
                    row_max1 = score1;
                }
            }

            core_acc8_t sum_acc0;
            core_acc8_t sum_acc1;
            core_acc8_clear(sum_acc0);
            core_acc8_clear(sum_acc1);
        EXP_KEY:
            for (int n = 0; n < active_count; ++n) {
#pragma HLS LOOP_TRIPCOUNT min=1 max=128
#pragma HLS PIPELINE II=1
                const float e0 = hls::expf(probability0[n] - row_max0);
                const float e1 = hls::expf(probability1[n] - row_max1);
                probability0[n] = e0;
                probability1[n] = e1;
                core_acc8_rotate_add(sum_acc0, e0);
                const float sum_acc1_updated = core_last_sum_accumulate(
                    sum_acc1.b0, e1);
                sum_acc1.b0 = sum_acc1.b1;
                sum_acc1.b1 = sum_acc1.b2;
                sum_acc1.b2 = sum_acc1.b3;
                sum_acc1.b3 = sum_acc1.b4;
                sum_acc1.b4 = sum_acc1.b5;
                sum_acc1.b5 = sum_acc1.b6;
                sum_acc1.b6 = sum_acc1.b7;
                sum_acc1.b7 = sum_acc1_updated;
            }
            float sum_lane0[ATTN_D_PAR];
            float sum_lane1[ATTN_D_PAR];
#pragma HLS ARRAY_PARTITION variable=sum_lane0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=sum_lane1 complete dim=1
            const int sum_phase = active_count & (ATTN_D_PAR - 1);
            core_acc8_read_logical(sum_acc0, sum_phase, sum_lane0);
            core_acc8_read_logical(sum_acc1, sum_phase, sum_lane1);
            const float sum0 = fp32_sum8_tree(sum_lane0);
            const float sum1 = fp32_sum8_tree(sum_lane1);
            const float inv_sum0 = (sum0 == 0.0f) ? 0.0f : 1.0f / sum0;
            const float inv_sum1 = (sum1 == 0.0f) ? 0.0f : 1.0f / sum1;

            float context_bank0[HEAD_DIM][ATTN_ACC_BANKS];
            float context_bank1[HEAD_DIM][ATTN_ACC_BANKS];
#pragma HLS ARRAY_PARTITION variable=context_bank0 cyclic factor=ATTN_CONTEXT_BANKS dim=1
#pragma HLS ARRAY_PARTITION variable=context_bank0 complete dim=2
#pragma HLS ARRAY_PARTITION variable=context_bank1 cyclic factor=ATTN_CONTEXT_BANKS dim=1
#pragma HLS ARRAY_PARTITION variable=context_bank1 complete dim=2
        CONTEXT_BANK_INIT_DIM:
            for (int d = 0; d < HEAD_DIM; d += ATTN_CONTEXT_BANKS) {
#pragma HLS PIPELINE II=1
            CONTEXT_BANK_INIT_LANE:
                for (int j = 0; j < ATTN_CONTEXT_BANKS; ++j) {
#pragma HLS UNROLL
                CONTEXT_BANK_INIT_BANK:
                    for (int b = 0; b < ATTN_ACC_BANKS; ++b) {
#pragma HLS UNROLL
                        context_bank0[d + j][b] = 0.0f;
                        context_bank1[d + j][b] = 0.0f;
                    }
                }
            }

        CONTEXT_KEY_CHUNK:
            for (int nd = 0;
                 nd < active_count * (HEAD_DIM / ATTN_CONTEXT_BANKS);
                 ++nd) {
#pragma HLS DEPENDENCE variable=context_bank0 inter false
#pragma HLS DEPENDENCE variable=context_bank1 inter false
#pragma HLS LOOP_TRIPCOUNT min=1 max=512
#pragma HLS PIPELINE II=3
                const int chunks = HEAD_DIM / ATTN_CONTEXT_BANKS;
                const int n = nd / chunks;
                const int d = (nd % chunks) * ATTN_CONTEXT_BANKS;
                const int key = active_indices[n];
                const int bank = n & (ATTN_ACC_BANKS - 1);
                const float p0 = probability0[n] * inv_sum0;
                const float p1 = probability1[n] * inv_sum1;
            CONTEXT_KEY_CHUNK_LANE:
                for (int j = 0; j < ATTN_CONTEXT_BANKS; ++j) {
#pragma HLS UNROLL
                    const float value = v[key][head_base + d + j];
                    context_bank0[d + j][bank] += p0 * value;
                    context_bank1[d + j][bank] += p1 * value;
                }
            }

        CONTEXT_BANK_REDUCE_DIM:
            for (int d = 0; d < HEAD_DIM; ++d) {
#pragma HLS PIPELINE II=1
                float lane0[ATTN_ACC_BANKS];
                float lane1[ATTN_ACC_BANKS];
#pragma HLS ARRAY_PARTITION variable=lane0 complete dim=1
#pragma HLS ARRAY_PARTITION variable=lane1 complete dim=1
            CONTEXT_BANK_REDUCE_BANK:
                for (int b = 0; b < ATTN_ACC_BANKS; ++b) {
#pragma HLS UNROLL
                    lane0[b] = context_bank0[d][b];
                    lane1[b] = context_bank1[d][b];
                }
                context[query][head_base + d] = fp32_sum8_tree(lane0);
                context[query + 1][head_base + d] = fp32_sum8_tree(lane1);
            }
        }
    }
}

static void write_context_group_stream(
    float context[SEQ_LEN][GROUP_DIM],
    hidden_stream_t &context_stream)
{
#pragma HLS INLINE off
WRITE_CONTEXT_ROW:
    for (int s = 0; s < SEQ_LEN; ++s) {
    WRITE_CONTEXT_PACK:
        for (int p = 0; p < GROUP_PACKS; ++p) {
#pragma HLS PIPELINE II=1
            float lane[PACK_SIZE];
#pragma HLS ARRAY_PARTITION variable=lane complete dim=1
        WRITE_CONTEXT_LANE:
            for (int i = 0; i < PACK_SIZE; ++i) {
#pragma HLS UNROLL
                lane[i] = context[s][p * PACK_SIZE + i];
            }
            context_stream.write(pack_bus16(lane));
        }
    }
}

extern "C" void bert_attn_core_kernel(
    bert_cmd_stream_t &layer_cmd,
    hidden_stream_t &q_stream,
    hidden_stream_t &k_stream,
    hidden_stream_t &v_stream,
    const int *attention_mask,
    hidden_stream_t &context_stream)
{
#pragma HLS INTERFACE axis port=layer_cmd
#pragma HLS INTERFACE axis port=q_stream
#pragma HLS INTERFACE axis port=k_stream
#pragma HLS INTERFACE axis port=v_stream
#pragma HLS INTERFACE axis port=context_stream
#pragma HLS INTERFACE m_axi port=attention_mask offset=slave bundle=gmem_mask depth=128
#pragma HLS INTERFACE s_axilite port=attention_mask bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    int active_indices[SEQ_LEN];
    float q[SEQ_LEN][GROUP_DIM];
    float k[SEQ_LEN][GROUP_DIM];
    float v[SEQ_LEN][GROUP_DIM];
    float group_context[SEQ_LEN][GROUP_DIM];
#pragma HLS BIND_STORAGE variable=active_indices type=RAM_S2P impl=LUTRAM
#pragma HLS BIND_STORAGE variable=q type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=k type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=v type=RAM_2P impl=URAM
#pragma HLS BIND_STORAGE variable=group_context type=RAM_2P impl=URAM
#pragma HLS ARRAY_RESHAPE variable=q cyclic factor=ATTN_D_PAR dim=2
#pragma HLS ARRAY_RESHAPE variable=k cyclic factor=ATTN_D_PAR dim=2
#pragma HLS ARRAY_RESHAPE variable=v cyclic factor=ATTN_CONTEXT_BANKS dim=2
#pragma HLS ARRAY_RESHAPE variable=group_context cyclic factor=ATTN_CONTEXT_BANKS dim=2

    int active_count = 0;
LOAD_MASK:
    for (int i = 0; i < SEQ_LEN; ++i) {
#pragma HLS PIPELINE II=1
        if (attention_mask[i] != 0) {
            active_indices[active_count] = i;
            ++active_count;
        }
    }

CORE_LAYER:
    for (int command_index = 0; command_index < BERT_NUM_LAYERS; ++command_index) {
#pragma HLS LOOP_TRIPCOUNT min=12 max=12
#pragma HLS UNROLL factor=1
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE off
        const bert_layer_cmd_t cmd = layer_cmd.read();
        (void)cmd;

    ATTENTION_GROUP:
        for (int group = 0; group < HEAD_GROUPS; ++group) {
            read_qkv_group(q_stream, k_stream, v_stream, q, k, v);
            attention_group(
                q, k, v, active_indices, active_count, group_context);
            write_context_group_stream(group_context, context_stream);
        }
    }
}
