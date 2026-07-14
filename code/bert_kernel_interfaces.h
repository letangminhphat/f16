#ifndef BERT_KERNEL_INTERFACES_H
#define BERT_KERNEL_INTERFACES_H

#include <ap_int.h>
#include <hls_stream.h>
#include <stdint.h>

#include "bert_encoder_protocol.h"

#define SEQ_LEN 128
#define HIDDEN_SIZE 768
#define NUM_HEADS 12
#define HEAD_DIM 64
#define HEAD_PAR 2
#define HEAD_GROUPS (NUM_HEADS / HEAD_PAR)

#define PACK_SIZE 16
#define PACKS (HIDDEN_SIZE / PACK_SIZE)
#define GROUP_DIM (HEAD_PAR * HEAD_DIM)
#define GROUP_PACKS (GROUP_DIM / PACK_SIZE)

#define QKV_TILE_O 64
#define QKV_K_PAR 8
#define QKV_O_PAR 8
#define QKV_ACC_BANKS 8

#define ATTN_D_PAR 8
#define ATTN_CONTEXT_BANKS 16
#define ATTN_ACC_BANKS 8

#define OUT_TILE_O 32
#define OUT_K_PAR 8
#define OUT_O_PAR 8

#define ATTN_W_PACKS (HIDDEN_SIZE * HIDDEN_SIZE / PACK_SIZE)
#define ATTN_B_PACKS PACKS

typedef ap_uint<512> bus_t;
typedef bus_t qkv_weight_bus_t;
typedef hls::stream<bus_t> hidden_stream_t;

#endif
