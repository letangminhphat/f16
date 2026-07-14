import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CURRENT_REPORT_DIR = ROOT / "report" / "csynth_current"

TOPS = {
    "ctrl": "bert_encoder_ctrl",
    "qkv": "bert_qkv_kernel",
    "core": "bert_attn_core_kernel",
    "out": "bert_attn_out_norm_kernel",
    "up": "bert_ffn_up_gelu_kernel",
    "down": "bert_ffn_down_residual_norm_kernel",
}

# Dynamic resources remaining after the U250 platform static region, copied
# from section 1.1 of the architecture MD. BRAM capacity is in 36-Kb tiles.
CAPACITY = {
    "SLR0": {"LUT": 420000, "FF": 840000, "BRAM": 668, "URAM": 312, "DSP": 3032},
    "SLR1": {"LUT": 205000, "FF": 411000, "BRAM": 384, "URAM": 128, "DSP": 1536},
    "SLR2": {"LUT": 407000, "FF": 815000, "BRAM": 660, "URAM": 308, "DSP": 2994},
    "SLR3": {"LUT": 424000, "FF": 849000, "BRAM": 672, "URAM": 320, "DSP": 3072},
}
CAP_RATIO = {"LUT": 0.65, "FF": 0.70, "BRAM": 0.65, "URAM": 0.70, "DSP": 0.70}
SLR_KERNELS = {
    "SLR0": ("qkv",),
    "SLR1": ("ctrl", "core"),
    "SLR2": ("out", "up"),
    "SLR3": ("down",),
}


def first_number(cell: str, *, integer: bool = True):
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", cell)
    if not match:
        return None
    text = match.group(0).replace(",", "")
    return int(float(text)) if integer else float(text)


def table_rows(text: str):
    for line in text.splitlines():
        line = line.lstrip()
        if not line.startswith("|"):
            continue
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) < 16:
            continue
        raw_name = cells[1]
        name = re.sub(r"^[+o ]+", "", raw_name).strip()
        yield {
            "name": name,
            "raw_name": raw_name,
            "type": cells[2],
            "slack_ns": first_number(cells[3], integer=False),
            "cycles": first_number(cells[4]),
            "iteration_latency": first_number(cells[6]),
            "interval": first_number(cells[7]),
            "count": first_number(cells[8]),
            "bram18k": first_number(cells[10]) or 0,
            "dsp": first_number(cells[11]) or 0,
            "ff": first_number(cells[12]) or 0,
            "lut": first_number(cells[13]) or 0,
            "uram": first_number(cells[14]) or 0,
        }


def find_row(rows, name: str):
    exact = [row for row in rows if row["name"] == name]
    if not exact:
        raise ValueError(f"row '{name}' not found")
    return exact[0]


def parse_report(path: Path, top: str):
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = list(table_rows(text))
    top_row = find_row(rows, top)
    required = ("cycles", "bram18k", "dsp", "ff", "lut", "uram")
    missing = [field for field in required if top_row[field] is None]
    if missing:
        raise ValueError(f"{path.name}: top row missing {', '.join(missing)}")
    return {"path": str(path), "text": text, "rows": rows, "top": top_row}


def current_reports(report_dir: Path):
    parsed = {}
    for key, top in TOPS.items():
        path = report_dir / f"csynth_{top}.rpt"
        parsed[key] = parse_report(path, top)
    return parsed


def baseline_reports():
    specs = {
        "qkv": ("csynth_qkv.rpt", "bert_qkv_kernel"),
        "core": ("csynth_attn_core.rpt", "bert_attn_core_kernel"),
        "out": ("csynth_attn_out_norm.rpt", "bert_attn_out_norm_kernel"),
        "up": ("csynth_up_v21.rpt", "bert_ffn_up_gelu_v21_dotpipe_kernel"),
        "down": ("csynth_down_v21.rpt", "bert_ffn_down_v21_dotpipe_kernel"),
    }
    parsed = {}
    for key, (filename, top) in specs.items():
        parsed[key] = parse_report(ROOT / "report" / filename, top)
    # Controller did not exist in the baseline reports. Zero here is only for
    # the legacy planning view; current acceptance requires its real report.
    parsed["ctrl"] = {
        "path": None,
        "rows": [],
        "top": {"cycles": 0, "bram18k": 0, "dsp": 0, "ff": 0, "lut": 0, "uram": 0, "slack_ns": None},
    }
    return parsed


def resource_result(parsed):
    kernel_resources = {}
    for key, report in parsed.items():
        top = report["top"]
        kernel_resources[key] = {
            "top": TOPS.get(key, key),
            "report": report["path"],
            "BRAM_18K": top["bram18k"],
            "BRAM_36K_equivalent": top["bram18k"] / 2.0,
            "DSP": top["dsp"],
            "FF": top["ff"],
            "LUT": top["lut"],
            "URAM": top["uram"],
            "slack_ns": top.get("slack_ns"),
        }

    slrs = {}
    all_fit = True
    for slr, keys in SLR_KERNELS.items():
        usage = {
            "LUT": sum(kernel_resources[key]["LUT"] for key in keys),
            "FF": sum(kernel_resources[key]["FF"] for key in keys),
            "BRAM": sum(kernel_resources[key]["BRAM_36K_equivalent"] for key in keys),
            "URAM": sum(kernel_resources[key]["URAM"] for key in keys),
            "DSP": sum(kernel_resources[key]["DSP"] for key in keys),
        }
        checks = {}
        for resource, value in usage.items():
            capacity = CAPACITY[slr][resource]
            ratio = value / capacity
            fit = ratio <= CAP_RATIO[resource]
            checks[resource] = {
                "used": value,
                "capacity_dynamic": capacity,
                "ratio": ratio,
                "planning_cap_ratio": CAP_RATIO[resource],
                "fit": fit,
            }
            all_fit &= fit
        slrs[slr] = {"kernels": list(keys), "resources": checks, "fit": all(v["fit"] for v in checks.values())}
    return kernel_resources, slrs, all_fit


def loop_latency(report, loop_name: str):
    row = find_row(report["rows"], loop_name)
    latency = row["iteration_latency"]
    if latency is None:
        raise ValueError(f"loop {loop_name} does not have a finite iteration latency")
    return latency


def latency_result(parsed, baseline: bool):
    q_group = loop_latency(parsed["qkv"], "HEAD_GROUP")
    c_group = loop_latency(parsed["core"], "ATTENTION_GROUP")
    o_group = loop_latency(parsed["out"], "PROJECT_CONTEXT_GROUP")

    if baseline:
        out_layer = parsed["out"]["top"]["cycles"]
        up_layer = parsed["up"]["top"]["cycles"]
        # Baseline DOWN has no final residual/LayerNorm; apply the MD's
        # conservative 70k-cycle tail budget.
        down_layer = parsed["down"]["top"]["cycles"] + 70000
        source = "legacy reports plus the MD final-LN/controller budgets"
    else:
        out_layer = loop_latency(parsed["out"], "OUT_NORM_LAYER")
        up_layer = loop_latency(parsed["up"], "v21_up_layer")
        down_layer = loop_latency(parsed["down"], "v21_down_layer")
        source = "current device-managed csynth loop latencies"

    attention_group_pipeline = q_group + c_group + o_group + 5 * max(q_group, c_group, o_group)
    out_non_group = max(0, out_layer - 6 * o_group)
    attention_layer = attention_group_pipeline + out_non_group
    ffn_layer = max(up_layer, down_layer)
    layer_cycles = attention_layer + ffn_layer
    controller_budget = 50000
    model_cycles = 12 * layer_cycles + controller_budget
    model_ms = model_cycles / 300000.0
    target_cycles = 60000000
    return {
        "qualification": "CSYNTH_PIPELINE_ESTIMATE; split-kernel full-model latency requires link/hardware validation",
        "source": source,
        "q_group_cycles": q_group,
        "core_group_cycles": c_group,
        "out_group_cycles": o_group,
        "attention_group_pipeline_cycles": attention_group_pipeline,
        "out_non_group_cycles": out_non_group,
        "attention_layer_cycles": attention_layer,
        "ffn_up_layer_cycles": up_layer,
        "ffn_down_fused_layer_cycles": down_layer,
        "ffn_critical_layer_cycles": ffn_layer,
        "encoder_layer_cycles": layer_cycles,
        "controller_budget_cycles": controller_budget,
        "model_cycles": model_cycles,
        "model_ms_at_300mhz": model_ms,
        "target_cycles": target_cycles,
        "target_ms": 200.0,
        "margin_cycles": target_cycles - model_cycles,
        "margin_ms": 200.0 - model_ms,
        "fit": model_cycles < target_cycles,
    }


def print_summary(result):
    print("\nPer-SLR planning check (standalone csynth resources):")
    print("SLR   FIT   LUT%    FF%   BRAM%  URAM%   DSP%")
    for slr, data in result["slrs"].items():
        r = data["resources"]
        ratios = [100 * r[name]["ratio"] for name in ("LUT", "FF", "BRAM", "URAM", "DSP")]
        print(f"{slr:<5} {'PASS' if data['fit'] else 'FAIL':<5} " + " ".join(f"{value:6.2f}" for value in ratios))

    latency = result["latency"]
    print("\n12-layer encoder latency model:")
    print(f"  attention/layer : {latency['attention_layer_cycles']:,} cycles")
    print(f"  FFN/layer       : {latency['ffn_critical_layer_cycles']:,} cycles")
    print(f"  model           : {latency['model_cycles']:,} cycles")
    print(f"  latency @300MHz : {latency['model_ms_at_300mhz']:.3f} ms")
    print(f"  margin          : {latency['margin_ms']:.3f} ms")
    print(f"  result          : {'PASS' if latency['fit'] else 'FAIL'} (<200 ms csynth estimate)")
    print("  NOTE: this is not linked/post-route/hardware E2E latency.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=CURRENT_REPORT_DIR)
    parser.add_argument("--baseline", action="store_true", help="analyze supplied one-layer legacy reports")
    parser.add_argument("--output", type=Path, default=ROOT / "build" / "csynth_fit_report.json")
    args = parser.parse_args()

    try:
        parsed = baseline_reports() if args.baseline else current_reports(args.report_dir)
        kernels, slrs, resource_fit = resource_result(parsed)
        latency = latency_result(parsed, args.baseline)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(
            f"ERROR: {exc}\nRun: vitis_hls -f scripts/csynth_each_kernel.tcl"
        )

    result = {
        "mode": "legacy_baseline" if args.baseline else "current_device_managed",
        "resource_qualification": "Standalone HLS estimates combined by assigned SLR; full-link utilization is still mandatory.",
        "kernels": kernels,
        "slrs": slrs,
        "resource_fit": resource_fit,
        "latency": latency,
        "status": "PASS_ESTIMATE" if resource_fit and latency["fit"] else "FAIL_ESTIMATE",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print_summary(result)
    print(f"\nJSON: {args.output}")
    raise SystemExit(0 if result["status"] == "PASS_ESTIMATE" else 1)


if __name__ == "__main__":
    main()
