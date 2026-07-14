#!/usr/bin/env python3
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "report"

TOPS = {
    "qkv": ("csynth_qkv.rpt", "bert_qkv_kernel"),
    "core": ("csynth_attn_core.rpt", "bert_attn_core_kernel"),
    "out": ("csynth_attn_out_norm.rpt", "bert_attn_out_norm_kernel"),
    "up": ("csynth_up_v21.rpt", "bert_ffn_up_gelu_v21_dotpipe_kernel"),
    "down": ("csynth_down_v21.rpt", "bert_ffn_down_v21_dotpipe_kernel"),
}

def top_latency(path: Path, top: str) -> int:
    text = path.read_text(encoding="utf-8", errors="replace")
    pattern = re.compile(r"^\s*\|\+\s*" + re.escape(top) + r"\s*\|.*$", re.M)
    match = pattern.search(text)
    if not match:
        raise RuntimeError(f"cannot find top row for {top} in {path}")
    columns = [part.strip() for part in match.group(0).split("|") if part.strip()]
    # Module, issue type, slack, latency cycles, latency ns, ...
    try:
        return int(columns[3])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"cannot parse top latency for {top}: {columns}") from exc

latency = {name: top_latency(REPORT / filename, top)
           for name, (filename, top) in TOPS.items()}

q_group = 369_738
c_group = 325_901
o_group = 215_758
attention_pipeline = q_group + c_group + o_group + 5 * max(q_group, c_group, o_group)
out_tail = latency["out"] - 6 * o_group
attention_layer = attention_pipeline + out_tail
ffn_projection = max(latency["up"], latency["down"])
final_norm_tail = 70_000
controller = 50_000
layer_cycles = attention_layer + ffn_projection + final_norm_tail
model_cycles = 12 * layer_cycles + controller
clock_hz = 300_000_000
model_ms = model_cycles / clock_hz * 1000.0

capacity = {
    "SLR0": {"lut": 420000, "ff": 840000, "bram": 668, "uram": 312, "dsp": 3032},
    "SLR1": {"lut": 205000, "ff": 411000, "bram": 384, "uram": 128, "dsp": 1536},
    "SLR2": {"lut": 407000, "ff": 815000, "bram": 660, "uram": 308, "dsp": 2994},
    "SLR3": {"lut": 424000, "ff": 849000, "bram": 672, "uram": 320, "dsp": 3072},
}
estimate = {
    "SLR0": {"lut": 131396, "ff": 230591, "bram": 54.5, "uram": 216, "dsp": 960},
    "SLR1": {"lut": 75459, "ff": 134926, "bram": 5.5, "uram": 24, "dsp": 185},
    "SLR2": {"lut": 168100, "ff": 275200, "bram": 302.5, "uram": 64, "dsp": 1676},
    "SLR3": {"lut": 123725, "ff": 231822, "bram": 103, "uram": 32, "dsp": 1299},
}
caps = {"lut": 0.65, "ff": 0.70, "bram": 0.65, "uram": 0.70, "dsp": 0.70}
ratios = {slr: {res: estimate[slr][res] / capacity[slr][res]
                for res in caps} for slr in capacity}
violations = [f"{slr}.{res}={ratio:.1%}>{caps[res]:.0%}"
              for slr, values in ratios.items() for res, ratio in values.items()
              if ratio > caps[res]]
if model_ms >= 200.0:
    violations.append(f"latency={model_ms:.3f}ms>=200ms")

result = {
    "source_report_latency_cycles": latency,
    "attention_cycles_per_layer": attention_layer,
    "ffn_projection_cycles_per_layer": ffn_projection,
    "final_norm_tail_budget_per_layer": final_norm_tail,
    "controller_budget_cycles_per_model": controller,
    "model_cycles": model_cycles,
    "model_ms_at_300mhz": round(model_ms, 3),
    "margin_ms": round(200.0 - model_ms, 3),
    "slr_ratio": ratios,
    "violations": violations,
    "status": "PASS_ESTIMATE" if not violations else "FAIL",
    "qualification": "Estimate only; post-route and hardware E2E measurement remain mandatory.",
}
(ROOT / "build").mkdir(exist_ok=True)
(ROOT / "build" / "budget_check.json").write_text(
    json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))
raise SystemExit(1 if violations else 0)
