from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"


def read(name: str) -> str:
    return (CODE / name).read_text(encoding="utf-8")


tops = {
    "bert_encoder_ctrl": "bert_encoder_ctrl.cpp",
    "bert_qkv_kernel": "bert_qkv_kernel.cpp",
    "bert_attn_core_kernel": "bert_attn_core_kernel.cpp",
    "bert_attn_out_norm_kernel": "bert_attn_out_norm_kernel.cpp",
    "bert_ffn_up_gelu_kernel": "bert_ffn_kernel_v21.cpp",
    "bert_ffn_down_residual_norm_kernel": "bert_ffn_kernel_v21.cpp",
}

for top, source in tops.items():
    text = read(source)
    assert f"void {top}(" in text, f"missing top {top} in {source}"

controller = read("bert_encoder_ctrl.cpp")
for port in ("qkv_cmd", "core_cmd", "out_cmd", "up_cmd", "down_cmd"):
    assert f"{port}.write(cmd)" in controller, f"controller does not drive {port}"
assert "layer_done.read()" in controller
assert "BERT_NUM_LAYERS" in controller

for source in (
    "bert_qkv_kernel.cpp",
    "bert_attn_core_kernel.cpp",
    "bert_attn_out_norm_kernel.cpp",
):
    assert "layer_cmd.read()" in read(source), f"missing worker command loop in {source}"

ffn = read("bert_ffn_kernel_v21.cpp")
assert ffn.count("layer_cmd.read()") >= 2
assert "v21_initialize_down_bias_residual(" in ffn
assert "v21_layernorm_and_store(" in ffn
assert "layer_output[BERT_HIDDEN_WORDS - 1]" in ffn
assert ffn.index("layer_output[BERT_HIDDEN_WORDS - 1]") < ffn.index("layer_done.write(")

out_norm = read("bert_attn_out_norm_kernel.cpp")
mid_write = out_norm.index("attn_mid_stream.write(word)")
residual_write = out_norm.index("ffn_residual_stream.write(word)")
assert mid_write < residual_write

config = (ROOT / "config" / "system.cfg").read_text(encoding="utf-8")
required_connections = {
    "ctrl1.qkv_cmd:qkv1.layer_cmd",
    "ctrl1.core_cmd:core1.layer_cmd",
    "ctrl1.out_cmd:out1.layer_cmd",
    "ctrl1.up_cmd:up1.layer_cmd",
    "ctrl1.down_cmd:down1.layer_cmd",
    "down1.layer_done:ctrl1.layer_done",
    "qkv1.q_stream:core1.q_stream",
    "qkv1.k_stream:core1.k_stream",
    "qkv1.v_stream:core1.v_stream",
    "core1.context_stream:out1.context_stream",
    "out1.attn_mid_stream:up1.attn_mid_stream",
    "out1.ffn_residual_stream:down1.ffn_residual_stream",
    "up1.gelu_stream:down1.gelu_stream",
}
for connection in required_connections:
    assert connection in config, f"missing stream connection {connection}"

for placement in (
    "slr=qkv1:SLR0",
    "slr=ctrl1:SLR1",
    "slr=core1:SLR1",
    "slr=out1:SLR2",
    "slr=up1:SLR2",
    "slr=down1:SLR3",
):
    assert placement in config, f"missing placement {placement}"

for path in CODE.glob("*.[ch]pp"):
    source = path.read_text(encoding="utf-8")
    assert '../../' not in source.replace('\\', '/'), f"external include in {path.name}"

print("PASS: device-managed 12-layer ABI, residual fork, fused DOWN/LN and SLR links are present.")
