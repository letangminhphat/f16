#include <ap_int.h>
#include <hls_stream.h>

#include "bert_ffn_kernel_lab.h"

#define PACK_BITS 512
#define PACK_FLOATS 16
#define HIDDEN_PACKS (HIDDEN_SIZE / PACK_FLOATS)
#define INPUT_WORDS (SEQ_LEN * HIDDEN_PACKS)

typedef ap_uint<PACK_BITS> packed_float16_t;
typedef hls::stream<packed_float16_t> packed_float_stream_t;

extern "C" {

void bert_ffn_input_v15_feeder_kernel(
    const packed_float16_t *input_hidden,
    packed_float_stream_t &attn_mid_stream)
{
#pragma HLS INTERFACE m_axi port=input_hidden offset=slave bundle=gmem_in \
    max_read_burst_length=16 num_read_outstanding=4 latency=64
#pragma HLS INTERFACE axis port=attn_mid_stream
#pragma HLS INTERFACE s_axilite port=input_hidden bundle=control
#pragma HLS INTERFACE s_axilite port=return bundle=control

    for (int word_idx = 0; word_idx < INPUT_WORDS; word_idx++) {
#pragma HLS PIPELINE II=1
        attn_mid_stream.write(input_hidden[word_idx]);
    }
}

}
