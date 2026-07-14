#include "bert_encoder_protocol.h"

extern "C" void bert_encoder_ctrl(
    bert_cmd_stream_t &qkv_cmd,
    bert_cmd_stream_t &core_cmd,
    bert_cmd_stream_t &out_cmd,
    bert_cmd_stream_t &up_cmd,
    bert_cmd_stream_t &down_cmd,
    bert_cmd_stream_t &layer_done,
    unsigned int *model_done)
{
#pragma HLS INTERFACE axis port=qkv_cmd
#pragma HLS INTERFACE axis port=core_cmd
#pragma HLS INTERFACE axis port=out_cmd
#pragma HLS INTERFACE axis port=up_cmd
#pragma HLS INTERFACE axis port=down_cmd
#pragma HLS INTERFACE axis port=layer_done
#pragma HLS INTERFACE m_axi port=model_done offset=slave bundle=gmem_status depth=1
#pragma HLS INTERFACE s_axilite port=model_done bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    unsigned int status = 0;
    model_done[0] = 0;

ENCODER_LAYER:
    for (unsigned int layer = 0; layer < BERT_NUM_LAYERS; ++layer) {
#pragma HLS LOOP_TRIPCOUNT min=12 max=12
#pragma HLS UNROLL factor=1
#pragma HLS LOOP_FLATTEN off
#pragma HLS PIPELINE off
        const unsigned int read_slot = layer & 1U;
        const unsigned int write_slot = read_slot ^ 1U;
        const bert_layer_cmd_t cmd = bert_make_layer_cmd(
            layer, read_slot, write_slot, layer == BERT_NUM_LAYERS - 1);

        // Dedicated command streams avoid a high-fanout control net spanning
        // all four SLRs.
        qkv_cmd.write(cmd);
        core_cmd.write(cmd);
        out_cmd.write(cmd);
        up_cmd.write(cmd);
        down_cmd.write(cmd);

        // DOWN emits this token only after the DDR output slot is visible.
        const bert_layer_cmd_t done = layer_done.read();
        if (done.to_uint() != layer + 1U) {
            status = 0x80000000U | done.to_uint();
        }
    }

    model_done[0] = (status == 0U) ? BERT_NUM_LAYERS : status;
}
