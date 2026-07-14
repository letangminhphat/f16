# BERT Attention + FFN current split-kernel architecture

Ngày tổng hợp: 2026-07-13

Cập nhật kiến trúc/budget: 2026-07-14

Tài liệu này ghi lại trạng thái hiện tại của các khối Attention và FFN cho
BERT-base trên Alveo U250. Mục đích là chốt kiến trúc split-kernel, latency
budget và ước lượng tài nguyên theo từng SLR trước khi full-link.

Các con số trong tài liệu được gắn mức bằng chứng rõ ràng:

- **Attention csynth:** Vitis HLS 2022.1, clock 300 MHz.
- **Attention RTL synthesis:** cả QKV, Core và Out/Norm đã chạy Vivado logic
  synthesis từ `export_design -flow syn`; chưa full-link/place-and-route.
- **FFN routed baseline:** V15 trong architecture handoff, đã full-link và
  post-route; report gốc dùng tool 2021.2.
- **FFN latency reference:** report V21 hiện có trong workspace; V21 chưa được
  chấp nhận làm route candidate cuối.
- **SLR capacity:** dùng tài nguyên dynamic region còn lại sau khi platform U250
  đã chiếm static region. Không trừ platform lần thứ hai.
- **SLR2 combined estimate:** cộng routed SLR2 baseline với standalone RTL
  synthesis của Attention Out/Norm. Đây là proxy bảo thủ, không thay thế
  utilization sau full-link.

## 1. Target hệ thống

```text
Board                  Alveo U250
Part                   xcu250-figd2104-2L-e
Attention HLS tool     Vitis HLS 2022.1
Clock                  300 MHz (3.333 ns)
Encoder layers         12
Sequence length        128
Hidden size            768
Attention heads        12
Head dimension         64
Intermediate size      3072
External stream width  512 bit = 16 x float32
Model latency target   < 200 ms
Total cycle budget     < 60,000,000 cycles
Per-layer budget       < 5,000,000 cycles
Layer scheduling       thực hiện 12 layer bên trong xclbin; host không lặp 12 lần
Route-friendly target  LUT <= 65%, FF <= 70%, BRAM <= 65%, URAM/DSP <= 70%/SLR
```

Kích thước tensor chính:

```text
hidden/context tensor  128 * 768  / 16 = 6,144 word 512-bit
FFN intermediate       128 * 3072 / 16 = 24,576 word 512-bit
```

### 1.1 Tài nguyên dynamic khả dụng sau platform installation

Đây là denominator phải dùng cho budget. `BRAM tile` trong bảng là tile 36 Kb;
report HLS/RTL standalone ghi `BRAM_18K`, nên phải chia hai trước khi so sánh.

| Resource | SLR0 | SLR1 | SLR2 | SLR3 |
|---|---:|---:|---:|---:|
| CLB LUT | 420,000 | 205,000 | 407,000 | 424,000 |
| CLB register | 840,000 | 411,000 | 815,000 | 849,000 |
| BRAM tile 36 Kb | 668 | 384 | 660 | 672 |
| URAM | 312 | 128 | 308 | 320 |
| DSP | 3,032 | 1,536 | 2,994 | 3,072 |

Planning cap dùng trước route:

| SLR | LUT 65% | FF 70% | BRAM 65% | URAM 70% | DSP 70% |
|---|---:|---:|---:|---:|---:|
| SLR0 | 273,000 | 588,000 | 434.2 | 218.4 | 2,122.4 |
| SLR1 | 133,250 | 287,700 | 249.6 | 89.6 | 1,075.2 |
| SLR2 | 264,550 | 570,500 | 429.0 | 215.6 | 2,095.8 |
| SLR3 | 275,600 | 594,300 | 436.8 | 224.0 | 2,150.4 |

70% không phải chứng nhận routing. LUT được giữ ở target 65% vì thiết kế có
nhiều FP-DSP, URAM banking và stream crossing tại 300 MHz. Chỉ cho phép vượt
target sau khi full-link cho thấy congestion và timing vẫn sạch.

## 2. Kiến trúc split-kernel mục tiêu sau hoàn thiện

Sơ đồ dưới đây gồm cả kernel đã có và khối bắt buộc còn phải implement. Trạng
thái thực tế của từng khối được tách rõ ở mục 8--9.

```text
Host chuẩn bị hidden tensor đầu vào tại DDR[0] slot 0 trước khi launch
          |
          v
SLR1  bert_encoder_ctrl (chạy một lần/model)
      - phát lệnh layer 0..11 và quản lý ping-pong
      - host không launch lại kernel theo từng layer
          |
          v
SLR0  bert_qkv_kernel
      - fused Q, K, V projection
      - xử lý 2 head/group, tổng cộng 6 group
          |
          | q_stream + k_stream + v_stream
          | 512 bit, group order
          v
SLR1  bert_attn_core_kernel
      - QK^T
      - attention mask
      - softmax FP32
      - weighted sum tạo context
          |
          | context_stream
          | 512 bit, row-major [sequence][hidden_pack]
          v
SLR2  bert_attn_out_norm_kernel
      - output projection
      - residual add
      - attention LayerNorm
      - ghi attn_mid_ddr và phát attn_mid_stream
          |
          | attn_mid_stream: 6,144 word, row-major
          v
SLR2  bert_ffn_up_gelu_*_kernel
      - projection 768 -> 3072
      - GELU
          |
          | gelu_stream: 24,576 word
          | order [intermediate_tile][sequence]
          v
SLR3  bert_ffn_down_residual_norm_kernel
      - projection 3072 -> 768
      - residual add + LayerNorm
      - ghi hidden ping-pong slot kế tiếp
          |
          | layer_done token về controller; lặp tới layer 11
          +---------------------------------------------------
```

Phân bố cố định:

| SLR | Kernel/khối | Vai trò |
|---|---|---|
| SLR0 | `bert_qkv_kernel` | Fused Q/K/V projection |
| SLR1 | `bert_encoder_ctrl` | Điều phối 12 layer hoàn toàn trong xclbin |
| SLR1 | `bert_attn_core_kernel` | Score, mask, softmax, weighted sum |
| SLR2 | `bert_attn_out_norm_kernel` | Output projection, residual, Attention LayerNorm |
| SLR2 | FFN UP + GELU | Projection 768 -> 3072 và activation |
| SLR3 | FFN DOWN + residual/LN | Hoàn tất encoder layer và phát `layer_done` |

FFN-only feeder trên SLR2 là test infrastructure. Khi full-link, output
`attn_mid_stream` của Out/Norm thay feeder và nối trực tiếp vào FFN UP.

Đây là kiến trúc hybrid: spatial giữa các operator/SLR, temporal reuse cùng
phần cứng qua 12 encoder layer. Lặp 12 layer không nhân tài nguyên lên 12 lần;
chỉ nhân thời gian xử lý và đổi offset weight theo `layer_id`.

## 3. Cấu hình song song Attention đã chốt

```text
HEAD_PAR             2
HEAD_GROUPS          6
QKV_TILE_O          64
QKV_K_PAR            8
QKV_O_PAR            8
QKV_ACC_BANKS        8
ATTN_D_PAR           8
ATTN_CONTEXT_BANKS  16
ATTN_ACC_BANKS       8
OUT_TILE_O          32
OUT_K_PAR            8
OUT_O_PAR            8
```

Các cấu hình này tạo pipeline sáu head-group:

```text
QKV group g -> Core group g -> Out projection accumulation group g
```

## 4. Tài nguyên và latency từng kernel

### 4.1 Attention — csynth hiện tại

| Kernel | Top latency | Group latency | BRAM | DSP | FF | LUT | URAM | Csynth slack |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| QKV | 2,230,737 | 369,738 | 0 | 960 | 283,348 | 178,060 | 216 | -0.30 ns |
| Attention Core | 1,955,549 | 325,901 | 2 | 185 | 180,508 | 140,172 | 24 | -0.30 ns |
| Attention Out/Norm | 1,365,434 | 215,758 | 16 | 393 | 98,437 | 91,362 | 32 | 0.00 ns |

Tcl hiện tại không gọi `set_clock_uncertainty`. HLS tự dùng scheduling budget
bảo thủ (QKV/Core 2.729 ns, Out/Norm 2.500 ns). Vì vậy không dùng cột HLS slack
để kết luận 300 MHz; dùng RTL logic-synthesis timing ở mục 4.2--4.3.

Tổng standalone csynth của ba kernel Attention là:

| BRAM | DSP | FF | LUT | URAM |
|---:|---:|---:|---:|---:|
| 18 | 1,538 | 562,293 | 409,594 | 272 |

Tổng này chỉ thể hiện lượng logic của Attention trên toàn thiết bị; không dùng
để đánh giá congestion vì ba kernel được đặt ở ba SLR khác nhau.

### 4.2 QKV — RTL logic synthesis tại 300 MHz

| Mục | Kết quả |
|---|---:|
| Requested period | 3.333 ns |
| Achieved post-synthesis period | 2.661 ns |
| Setup slack | **+0.672 ns** |
| Equivalent frequency | khoảng 375.8 MHz |
| LUT | 131,396 |
| FF | 230,591 |
| DSP | 960 |
| BRAM_18K | 109 = 54.5 BRAM tile |
| URAM | 216 |

QKV đạt timing thật ở bước logic synthesis mà không giảm clock uncertainty.
Critical path sau synthesis là control/fanout tới địa chỉ K-weight URAM, với
net delay chiếm phần lớn datapath; nó không còn là critical FP adder của HLS
scheduler. Đây chưa phải post-route timing closure.

### 4.3 Core và Out/Norm — RTL logic synthesis tại 300 MHz

| Kernel | Required period | Achieved period | Setup slack | LUT | FF | DSP | BRAM_18K* | URAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Attention Core | 3.333 ns | 2.987 ns | **+0.346 ns** | 70,459 | 126,926 | 185 | 3 | 24 |
| Attention Out/Norm | 3.333 ns | 2.479 ns | **+0.854 ns** | 54,483 | 80,174 | 393 | 107 | 32 |

`*` Khi cộng với cột `BRAM Tile` của routed SLR report, hai BRAM_18K được
quy đổi thành khoảng một BRAM tile.

Core worst path là FSM/control của `CONTEXT_KEY_CHUNK` tới write-enable của
context-bank LUTRAM. Net delay chiếm khoảng 81.3%; `EXP_KEY` không còn là
critical path sau RTL synthesis.

Out/Norm worst paths là local `projected` URAM read qua một LUT tới register
trong projection/norm. Norm fabric fadd không còn là worst synthesized path.

Cả ba Attention kernel đều đạt 300 MHz ở logic synthesis. Đây chưa phải timing
sau placement/routing; FFN V15 full-link vẫn còn WNS âm nhẹ.

### 4.4 FFN — latency report hiện có

| Kernel | Version/report | Top latency | BRAM_18K | DSP | FF | LUT | URAM | Trạng thái |
|---|---|---:|---:|---:|---:|---:|---:|---|
| FFN UP + GELU | V21 | 1,536,392 | 272 | 864 | 190,406 | 134,244 | 32 | Latency reference; chưa là final route candidate |
| FFN DOWN | V21 | 1,655,260 | 16 | 768 | 209,957 | 142,758 | 32 | Latency reference; chưa là final route candidate |

V15 vẫn là route baseline ổn định của FFN. V21 chứng minh hierarchy/II nhưng
không thay được critical multiplier RTL như dự kiến, nên không được dùng để
tuyên bố timing closure.

### 4.5 FFN V15 routed baseline

Kernel utilization sau route:

| CU | LUT | REG | BRAM | URAM | DSP |
|---|---:|---:|---:|---:|---:|
| FFN-only feeder | 1,716 | 4,432 | 7 | 0 | 0 |
| V15 UP | 71,610 | 134,107 | 143 | 32 | 1,280 |
| V15 DOWN | 79,150 | 163,887 | 23 | 32 | 1,280 |

Full-SLR utilization của routed FFN-only design:

| SLR | LUT | REG | BRAM tile | URAM | DSP |
|---|---:|---:|---:|---:|---:|
| SLR2 | 113,545 (27.90%) | 194,519 (23.87%) | 247 (37.42%) | 32 (10.39%) | 1,283 (42.85%) |
| SLR3 | 108,725 (25.64%) | 206,822 (24.36%) | 87 (12.95%) | 32 (10.00%) | 1,283 (41.76%) |

Timing routed baseline:

```text
WNS  = -0.033 ns
TNS  = -5.784 ns
WHS  =  0.005 ns
setup failing endpoints = 336
```

Critical paths là FP multiplier/DSP nội bộ UP và DOWN, không phải AXIS/SLL.

## 5. Tài nguyên gộp theo từng SLR

### 5.1 Bảng đánh giá hiện tại

| SLR | Thành phần | LUT | FF/REG | BRAM | URAM | DSP | Đánh giá |
|---|---|---:|---:|---:|---:|---:|---|
| SLR0 | QKV RTL synthesis | 131,396 | 230,591 | 54.5 BRAM tile | 216 | 960 | URAM 69.23%, không nhận thêm URAM |
| SLR1 | Core RTL synthesis | 70,459 | 126,926 | 1.5 BRAM tile | 24 | 185 | Còn nhiều headroom |
| SLR2 | Routed FFN SLR2 baseline + Out/Norm RTL synthesis | 168,028 | 274,693 | 300.5 BRAM tile | 64 | 1,676 | DSP 55.98%, BRAM 45.53% |
| SLR3 | Routed FFN SLR3 baseline | 108,725 | 206,822 | 87 | 32 | 1,283 | Còn headroom cho FFN residual/LN |

Ước lượng tỷ lệ SLR2 sau khi cộng Out/Norm:

| Resource | Combined estimate | Tỷ lệ SLR2 |
|---|---:|---:|
| LUT | 168,028 | 41.28% |
| FF/REG | 274,693 | 33.70% |
| BRAM tile | 247 + 107/2 = 300.5 | 45.53% |
| URAM | 64 | 20.78% |
| DSP | 1,676 | 55.98% |

Đây là phép cộng khác mức báo cáo: routed full-SLR baseline cộng standalone
HLS estimate. Nó thích hợp để kiểm tra headroom ban đầu nhưng không phải con số
utilization cuối cùng. Chỉ full-link mới loại bỏ double-counting/static overhead
và cho biết congestion thật.

SLR0 QKV so với dynamic resource đã trừ static platform:

```text
DSP   31.66%
FF    27.45%
LUT   31.28%
BRAM   8.16%
URAM  69.23%
```

URAM chỉ còn 2.4 block tới planning cap 70%. Vì vậy không tăng QKV parallelism
và không thêm full-tensor URAM buffer tại đây.

## 6. Latency budget dưới 200 ms

Gọi:

```text
Q = 369,738 cycles/group
C = 325,901 cycles/group
O = 215,758 cycles/group
```

Sáu group đi qua ba Attention kernel theo pipeline. Công thức estimate:

```text
attention_group_pipeline = Q + C + O + 5 * max(Q, C, O)
norm_tail                 = Out top - 6 * O
attention_cycles          = attention_group_pipeline + norm_tail
ffn_cycles                = max(FFN_UP, FFN_DOWN)
layer_cycles              = attention_cycles + ffn_cycles
model_cycles              = 12 * layer_cycles
model_ms                  = model_cycles / 300,000
```

Kết quả hiện tại:

| Thành phần | Cycles |
|---|---:|
| Attention group pipeline | 2,760,087 |
| Out/Norm tail | 70,886 |
| Attention total/layer | 2,830,973 |
| FFN critical stage/layer | 1,655,260 |
| **Tổng một encoder layer** | **4,486,233** |
| Per-layer budget | 5,000,000 |
| **12-layer estimate** | **53,834,796** |
| Total budget | 60,000,000 |
| **Estimated latency @ 300 MHz** | **179.45 ms** |
| **Budget margin** | **6,165,204 cycles = 20.55 ms** |

Model estimate đạt mục tiêu dưới 200 ms với khoảng 10.3% margin trước các khối
còn thiếu. Đây là compute/pipeline estimate, chưa bao gồm one-time XRT launch,
DDR contention, full-link backpressure hoặc task head.

QKV là bottleneck của Attention vì có group latency lớn nhất. Core và Out/Norm
đã nhanh hơn QKV, nên giảm II hoặc tăng parallelism ở hai kernel này không làm
tăng throughput Attention hiện tại.

## 7. Stream contract và crossing SLR

| Crossing | Stream | Width | Word count | Logical order |
|---|---|---:|---:|---|
| SLR0 -> SLR1 | `q_stream` | 512 bit | 6,144 | six head-groups, packed FP32 |
| SLR0 -> SLR1 | `k_stream` | 512 bit | 6,144 | six head-groups, packed FP32 |
| SLR0 -> SLR1 | `v_stream` | 512 bit | 6,144 | six head-groups, packed FP32 |
| SLR1 -> SLR2 | `context_stream` | 512 bit | 6,144 | row-major `[sequence][hidden_pack]` |
| SLR2 Attention -> FFN UP | `attn_mid_stream` | 512 bit | 6,144 | row-major `[sequence][hidden_pack]` |
| SLR2 -> SLR3 | `ffn_residual_stream` | 512 bit | 6,144 | row-major `[sequence][hidden_pack]` |
| SLR2 -> SLR3 | `gelu_stream` | 512 bit | 24,576 | `[intermediate_tile][sequence]` |
| SLR1 -> worker CUs | `layer_cmd_*` | 32/64 bit | 1/layer | `{layer_id, read_slot, write_slot, last}` |
| SLR3 -> SLR1 | `layer_done` | 32 bit | 1/layer | layer hoàn tất, DDR output đã visible |

V15 đã route crossing SLR2 -> SLR3 với SLL utilization
`2,789 / 23,040 = 12.11%`. Không thay đổi order của `gelu_stream` nếu không sửa
đồng bộ FFN DOWN consumer.

## 8. Trạng thái chốt và phần còn thiếu

### Đã có

- Fused QKV projection trên SLR0.
- Attention score, mask, softmax và weighted sum trên SLR1.
- Attention output projection, residual và LayerNorm trên SLR2.
- FFN UP + GELU trên SLR2.
- FFN DOWN projection trên SLR3.
- ABI stream 512-bit giữa các kernel Attention.
- Latency estimate dưới 200 ms.
- QKV đạt logic-synthesis timing tại 300 MHz.
- Core và Out/Norm đạt logic-synthesis timing tại 300 MHz.
- Routed FFN baseline và SLR2-SLR3 stream crossing.

### Cần xác nhận trước full-link

1. Chốt FFN route candidate: V15 vẫn là baseline; V21 chưa được chấp nhận.
2. Implement các khối bắt buộc còn thiếu và loop 12 layer trong xclbin.
3. Full-link cả bốn SLR và lấy per-SLR utilization/congestion report.
4. Xác nhận FIFO depth, DDR ping-pong visibility và không có deadlock.
5. Chạy accuracy regression cho Attention, softmax, GELU và cả hai LayerNorm.

### Chưa có trong datapath hoàn chỉnh

- FFN residual add và LayerNorm sau DOWN. Architecture handoff đề xuất fork
  Attention residual output từ SLR2 sang SLR3 rồi tích hợp vào DOWN buffer.
- `bert_encoder_ctrl` và worker ABI để điều phối 12 layer trong xclbin.
- Pooler hoặc task-specific output head.
- End-to-end hardware runtime bao gồm DDR và host overhead.

Lưu ý trạng thái source: ba Attention kernel hiện vẫn nhận/chạy một `layer_id`
mỗi invocation; FFN source hoàn chỉnh và controller không có trong workspace
này. Do đó yêu cầu device-managed 12 layer là kiến trúc bắt buộc tiếp theo,
không phải chức năng đã implement.

## 9. Các kernel còn phải làm

Phần Attention đã có đủ ba kernel compute. FFN đã có UP và DOWN projection,
nhưng DOWN chưa có residual add và LayerNorm cuối encoder layer. Vì vậy roadmap
được chia thành kernel bắt buộc cho encoder và kernel tùy chọn theo task.

### 9.1 Danh sách tổng hợp

| Ưu tiên | Kernel/công việc | SLR | Bắt buộc | Latency target | Resource tăng tối đa dự kiến |
|---:|---|---|---|---:|---|
| 1 | Fuse residual + LayerNorm vào FFN DOWN | SLR3 | Có | tail <= 70,000 cycles/layer | 16 BRAM, 16 DSP, 25k FF, 15k LUT, không thêm URAM nếu reuse buffer |
| 2 | Thêm `ffn_residual_stream` vào Out/Norm và DOWN | SLR2 -> SLR3 | Có | II=1, overlap; không cộng tail tuần tự | FIFO/crossing nhỏ, không thêm compute DSP |
| 3 | `bert_encoder_ctrl` + worker loop 12 layer | SLR1 + mọi worker | Có | <= 50,000 cycles/model control overhead | <= 4 BRAM tile, 0 DSP/URAM, 8k FF, 5k LUT cho controller |
| 4 | `bert_pooler_classifier_kernel` | SLR3 | Chỉ classification | <= 50,000 cycles/model | 128 DSP, 60k FF, 40k LUT, 16 URAM |
| 5 | Full-link/connectivity build | Cả 4 SLR | Có | không làm giảm throughput group | Không phải kernel compute mới |

Các resource trên là **trần thiết kế ban đầu**, chưa phải kết quả csynth. Kernel
mới chỉ được chấp nhận nếu không vượt trần và tổng SLR vẫn giữ dưới khoảng 70%
khi có thể.

### 9.2 Bắt buộc: FFN DOWN + residual + LayerNorm trên SLR3

Không nên tạo một LayerNorm kernel riêng vì sẽ cần thêm buffer/stream boundary
và đọc lại toàn bộ hidden tensor. Hướng ưu tiên là đổi kernel DOWN hiện tại
thành một kernel fused, ví dụ:

```text
bert_ffn_down_residual_norm_kernel
```

Datapath:

```text
gelu_stream ----------------------+
                                  v
W2/B2 -> FFN DOWN projection -> residual add -> LayerNorm -> output
                                  ^
                                  |
ffn_residual_stream --------------+
```

Interface dự kiến:

```text
input  gelu_stream          512 bit, 24,576 word, tile-major
input  ffn_residual_stream  512 bit,  6,144 word, row-major
input  W2/B2                DDR[3]
input  final LN gamma/beta  DDR[3]
output hidden              DDR[0] hoặc 512-bit stream cho layer kế tiếp
```

Implementation:

- Reuse `output_buf` URAM hiện có của DOWN.
- Không tạo full residual buffer thứ hai nếu residual stream có thể được drain
  vào đúng lịch khởi tạo/output buffer.
- Giữ hot projection loop II=1.
- LayerNorm có thể dùng cấu trúc reduction-bank tương tự Attention Out/Norm.
- Không thay đổi GELU stream order.

Latency budget:

| Thành phần | Target |
|---|---:|
| FFN DOWN projection reference | 1,655,260 cycles |
| Residual + LayerNorm tail | <= 70,000 cycles |
| **Fused DOWN total** | **<= 1,725,260 cycles** |

Resource tăng cho phần residual/LN được lấy bảo thủ từ khối
`residual_norm_and_write` đang có ở Attention Out/Norm:

| Resource | Increment budget | SLR3 combined estimate sau khi thêm |
|---|---:|---:|
| LUT | <= 15,000 | <= 123,725, khoảng 28.6% SLR3 |
| FF | <= 25,000 | <= 231,822, khoảng 26.8% SLR3 |
| BRAM | <= 16 | <= 103, khoảng 15.3% SLR3 |
| URAM | 0 nếu reuse buffer | 32, khoảng 10.0% SLR3 |
| DSP | <= 16 | <= 1,299, khoảng 42.3% SLR3 |

SLR3 còn đủ headroom. Acceptance gate quan trọng hơn resource là timing của FP
multiplier và rsqrt/normalization tại 300 MHz.

### 9.3 Bắt buộc: residual stream từ SLR2 sang SLR3

Attention Out/Norm đã tạo hidden tensor sau Attention residual/LN. Tensor này
vừa là input FFN UP, vừa là residual phải cộng lại sau FFN DOWN.

Hướng ưu tiên là cho `bert_attn_out_norm_kernel` phát hai stream đồng thời:

```text
attn_mid_stream       -> FFN UP trên SLR2
ffn_residual_stream   -> fused FFN DOWN trên SLR3
```

Contract:

```text
width       512 bit
word count  6,144
order       [sequence][hidden_pack]
producer II 1
consumer    fused DOWN phải drain stream thật
```

Không tạo standalone fork kernel nếu Out/Norm có thể duplicate mỗi word khi
ghi. Việc duplicate phải overlap với output loop, nên latency tăng kỳ vọng gần
0 cycle; không được cộng thêm 6,144 cycles tuần tự vào critical layer path.

Chỉ thêm stream sau khi DOWN đã có consumer. Stream không được drain sẽ gây
deadlock toàn pipeline.

### 9.4 Bắt buộc: điều phối 12 encoder layer bên trong xclbin

Host chỉ thực hiện setup buffer/argument, nạp xclbin và launch mỗi CU **một
lần**. Host không có vòng `for layer=0..11`. Một controller nhỏ trong xclbin
điều phối tiến độ; controller không launch kernel khác mà trao đổi token AXIS
với các worker đã chạy.

Kernel đề xuất:

```text
bert_encoder_ctrl                     SLR1
bert_qkv_kernel                       SLR0, one launch / 12 commands
bert_attn_core_kernel                 SLR1, one launch / 12 layer payloads
bert_attn_out_norm_kernel             SLR2, one launch / 12 commands
bert_ffn_up_gelu_*_kernel             SLR2, one launch / 12 commands
bert_ffn_down_residual_norm_kernel    SLR3, one launch / 12 commands
```

Protocol:

1. Trước khi launch, host ghi hidden tensor đã chuẩn bị sẵn vào `slot[0]` tại
   DDR[0]. Đây là DMA input một lần, không phải điều phối encoder layer.
2. Sau khi được launch một lần, controller phát
   `{layer_id, read_slot, write_slot, last}` qua các FIFO lệnh
   riêng; không dùng một control net fanout xuyên bốn SLR.
3. Các worker tính offset weight từ `layer_id` và xử lý đúng một layer. Core
   đọc đúng sáu head-group mỗi layer; mask có thể load một lần ngoài loop.
4. Fused DOWN đọc residual sớm để drain `ffn_residual_stream`, hoàn tất
   projection + residual + LayerNorm, rồi ghi `slot[write_slot]`.
5. DOWN chỉ phát `layer_done` sau khi AXI write response/status marker bảo đảm
   dữ liệu DDR đã visible. Controller đổi ping-pong và phát layer kế tiếp.
6. Sau layer 11, controller phát `model_done`; output cuối ở slot đã xác định
   hoặc kích hoạt head tùy chọn một lần.

```text
layer 0: read slot 0 -> write slot 1
layer 1: read slot 1 -> write slot 0
...
layer 11: read slot 1 -> write slot 0
```

DDR ping-pong được chọn thay cho feedback stream 512-bit SLR3 -> SLR0 để tránh
một bus rộng đi xuyên ba biên SLR. Chỉ token điều khiển hẹp quay ngược về SLR1.
Hai slot dùng DDR, không tạo thêm full-hidden URAM buffer.

Budget controller ban đầu:

| Mục | Trần |
|---|---:|
| LUT | <= 5,000 |
| FF | <= 8,000 |
| BRAM tile/FIFO | <= 4 |
| DSP/URAM | 0 |
| Control overhead | <= 50,000 cycles/model = 0.167 ms |

Lặp trong mỗi CU tái sử dụng cùng datapath, nên tài nguyên không nhân 12. ABI
hiện tại vẫn là one-layer (`layer_id` do host truyền) và phải được đổi; kiến
trúc device-managed này chưa tồn tại trong source hiện tại.

### 9.5 Tùy chọn: pooler và classification head trên SLR3

Chỉ cần nếu mục tiêu là sequence classification. Encoder-only benchmark không
cần kernel này.

Kernel đề xuất:

```text
bert_pooler_classifier_kernel
```

Chức năng:

```text
đọc hidden vector của token [CLS]
-> dense 768 x 768
-> tanh
-> task classifier 768 x num_labels
-> logits
```

Placement: **SLR3**, ngay sau output cuối của fused DOWN. Kernel chạy một lần
cho toàn model, nên có thể chia sẻ bandwidth DDR[0]/DDR[3] theo lịch.

Budget cho classification head nhỏ:

| Mục | Target |
|---|---:|
| Latency | <= 50,000 cycles = 0.167 ms/model |
| LUT | <= 40,000 |
| FF | <= 60,000 |
| DSP | <= 128 |
| BRAM | <= 32 |
| URAM | <= 16 |

Với NER/token classification hoặc question answering, head phải xử lý toàn bộ
128 token và cần một budget riêng; không dùng trực tiếp con số classification
head ở trên.

### 9.6 Tài nguyên SLR dự kiến sau khi hoàn thiện

Bảng dưới cộng upper-bound của các kernel còn thiếu vào trạng thái hiện tại.
Nó dùng standalone HLS estimate cho SLR0/SLR1, routed-plus-csynth proxy cho
SLR2 và routed baseline cho SLR3. Vì vậy đây là bảng budget, không phải
full-link utilization report.

| SLR | Cấu hình hoàn thiện được tính | LUT | FF/REG | BRAM tile | URAM | DSP | Tỷ lệ LUT/FF/BRAM/URAM/DSP |
|---|---|---:|---:|---:|---:|---:|---|
| SLR0 | QKV RTL synthesis | 131,396 | 230,591 | 54.5 | 216 | 960 | 31.3% / 27.5% / 8.2% / **69.2%** / 31.7% |
| SLR1 | Core + controller upper-bound | 75,459 | 134,926 | 5.5 | 24 | 185 | 36.8% / 32.8% / 1.4% / 18.8% / 12.0% |
| SLR2 | Routed FFN SLR2 + Out/Norm RTL synthesis + residual fork nhỏ | khoảng 168,100 | khoảng 275,200 | khoảng 300.5 + FIFO | 64 | 1,676 | 41.3% / 33.8% / 45.5% / 20.8% / **56.0%** |
| SLR3 | Routed FFN SLR3 + fused residual/LN | 123,725 | 231,822 | 103 | 32 | 1,299 | 29.2% / 27.3% / 15.3% / 10.0% / 42.3% |
| SLR3 optional | Dòng trên + classification head | 163,725 | 291,822 | 135 | 48 | 1,427 | 38.6% / 34.4% / 20.1% / 15.0% / 46.5% |

Nhận xét:

- SLR0 đã sát planning cap URAM: 216/312 = 69.23%. Không thêm URAM tại đây.
- SLR1 có Core + controller vẫn rất an toàn dù đây là SLR nhỏ nhất.
- SLR2 có DSP cao nhất do Out projection và FFN UP cùng tồn tại: 55.98%. Còn
  khoảng 420 DSP tới cap 70%; không đặt compute lớn mới tại đây.
- SLR3 còn đủ chỗ cho residual/LN và một classification head nhỏ.
- FIFO/AXIS infrastructure chỉ được chốt sau full-link; không giả định bằng 0
  trong utilization cuối.

Headroom tới planning cap sau các khối bắt buộc:

| SLR | LUT còn lại | FF còn lại | BRAM tile còn lại | URAM còn lại | DSP còn lại |
|---|---:|---:|---:|---:|---:|
| SLR0 | 141,604 | 357,409 | 379.7 | **2.4** | 1,162.4 |
| SLR1 | 57,791 | 152,774 | 244.1 | 65.6 | 890.2 |
| SLR2 | khoảng 96,450 | khoảng 295,300 | khoảng 128.5 trừ FIFO | 151.6 | 419.8 |
| SLR3 | 151,875 | 362,478 | 333.8 | 192.0 | 851.4 |

Kết luận budget: các khối bắt buộc fit theo estimate. Chưa được gọi là
"routing-safe" cho đến khi full-link vì tổng SLR thấp vẫn có thể nghẽn cục bộ
ở cột URAM/DSP hoặc tại clock region/SLL.

### 9.7 Latency budget sau khi hoàn thiện các khối bắt buộc

Budget hiện tại `179.45 ms` chưa tính FFN residual/LayerNorm cuối mỗi layer và
controller. Hidden input đã được chuẩn bị ngoài xclbin; accelerator budget bắt
đầu tại QKV:

```text
current model estimate               53,834,796 cycles
FFN residual/LN, 12 * 70,000            840,000 cycles
device controller upper bound             50,000 cycles
---------------------------------------------------------
required encoder total               54,724,796 cycles = 182.42 ms
optional classification head             50,000 cycles
complete classification total        54,774,796 cycles = 182.58 ms
```

Margin:

```text
encoder-only remaining                5,275,204 cycles = 17.58 ms
with classifier remaining             5,225,204 cycles = 17.42 ms
```

Đây là compute estimate; DDR contention, full-link backpressure và one-time
XRT launch vẫn phải đo end-to-end.

### 9.8 Những việc không nên tách thành kernel mới

- Không tách Q, K, V thành ba kernel mới: fused QKV hiện đã đạt latency,
  resource và logic-synthesis timing.
- Không tạo standalone Attention LayerNorm: đã có trong Out/Norm.
- Không tạo standalone FFN LayerNorm: fuse vào DOWN để reuse output buffer.
- Không tạo stream-fork kernel riêng nếu Out/Norm duplicate được hai output.
- Không tạo DDR reorder kernel giữa Core và Out/Norm; stream order đã khớp.
- Không tạo wait-stage compute kernel. `bert_encoder_ctrl` là điều phối chức
  năng bắt buộc cho recurrence 12 layer; nó dùng token hẹp và không được tính
  như một stage latency của datapath Attention/FFN.

## 10. Quyết định tối ưu hiện tại

```text
QKV       khóa datapath hiện tại; không tăng parallelism do URAM/routing.
Core      giữ source; chỉ sửa nếu RTL synthesis thật sự âm tại 3.333 ns.
Out/Norm  giữ projection II=2; không đổi sang II=1 vì không phải bottleneck và
          cùng chia sẻ SLR2 với FFN UP.
FFN       giữ V15 làm routed baseline cho integration; V21 chỉ dùng làm report
          latency cho đến khi có bằng chứng route tốt hơn.
```

Không dùng custom clock uncertainty để làm đẹp report. Timing cuối cùng phải
được đánh giá bằng RTL synthesis và sau đó bằng full-link/post-route ở 300 MHz.

## 11. Report nguồn

```text
reports/csynth_qkv.rpt
reports/csynth_attn_core.rpt
reports/csynth_attn_out_norm.rpt
reports/bert_qkv_kernel_export.rpt
reports/export_syn.rpt
reports/bert_attn_core_kernel/verilog/export_syn.rpt
reports/bert_attn_out_norm_kernel/verilog/export_syn.rpt
csynth_up_v21.rpt
csynth_down_v21.rpt
config/system.cfg
docs/u250_ds962_resource_information.md
ATTENTION_200MS_LATENCY_BUDGET.md
ATTENTION_FFN_SLR_ARCHITECTURE_HANDOFF.md
```
