#ifndef BERT_ENCODER_PROTOCOL_H
#define BERT_ENCODER_PROTOCOL_H

#include <ap_int.h>
#include <hls_stream.h>

#define BERT_NUM_LAYERS 12
#define BERT_HIDDEN_WORDS 6144
#define BERT_HIDDEN_SLOTS 2

typedef ap_uint<32> bert_layer_cmd_t;
typedef hls::stream<bert_layer_cmd_t> bert_cmd_stream_t;

static bert_layer_cmd_t bert_make_layer_cmd(
    unsigned int layer_id,
    unsigned int read_slot,
    unsigned int write_slot,
    bool last)
{
#pragma HLS INLINE
    bert_layer_cmd_t cmd = 0;
    cmd.range(3, 0) = layer_id;
    cmd[4] = read_slot & 1U;
    cmd[5] = write_slot & 1U;
    cmd[6] = last ? 1 : 0;
    return cmd;
}

static unsigned int bert_cmd_layer(bert_layer_cmd_t cmd)
{
#pragma HLS INLINE
    return cmd.range(3, 0).to_uint();
}

static unsigned int bert_cmd_read_slot(bert_layer_cmd_t cmd)
{
#pragma HLS INLINE
    return cmd[4] ? 1U : 0U;
}

static unsigned int bert_cmd_write_slot(bert_layer_cmd_t cmd)
{
#pragma HLS INLINE
    return cmd[5] ? 1U : 0U;
}

#endif
