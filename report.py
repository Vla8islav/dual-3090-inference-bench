#!/usr/bin/env python3
"""Aggregate evidence and compare only matched measured inference cases."""
import argparse
import json
from pathlib import Path
import re


def read(path):
    return json.loads(path.read_text())


def summarize(directory, exit_code):
    reports = {p.parent.name: read(p) for p in directory.glob("*/report.json")}
    failures, warnings = [], []
    if exit_code:
        failures.append(f"orchestrator exited {exit_code}; inspect stage/startup/build logs")
    config = read(directory / "configuration.json")
    expected = {"preflight"}
    if config["mode"] in ("hardware", "quick", "extended", "soak"):
        expected.add("hardware")
    if config["mode"] in ("inference", "quick", "extended", "soak"):
        expected.update(["tp1-gpu0", "tp1-gpu1", "tp2"])
    for stage in sorted(expected):
        if stage not in reports:
            failures.append(f"missing selected stage: {stage}")
    for stage, report in reports.items():
        if report["status"] != "PASS_SELECTED_CHECKS":
            failures.append(f"{stage}: {report['status']}")
    error_log = directory / "host-error.txt"
    if error_log.exists():
        failures.append(error_log.read_text().strip())
    kernel = directory / "kernel.log"
    permission = directory / "kernel.err"
    start_permission = directory / "journal-start.err"
    if (not kernel.exists() or (permission.exists() and permission.read_text().strip())
            or (start_permission.exists() and start_permission.read_text().strip())):
        warnings.append("Kernel journal coverage unavailable or restricted; Xid/modeset/OOM absence is NOT established.")
    if kernel.exists():
        pattern = re.compile(r"NVRM:\s*Xid|Error while waiting for GPU progress|fallen off the bus|oom-kill|Out of memory: Killed", re.I)
        failures.extend(line for line in kernel.read_text().splitlines() if pattern.search(line))
    warnings.append("No universal speed floor: use a matched baseline comparison or configure rate floors.")
    if config["mode"] == "preflight":
        warnings.append("Inventory only; inference and memory correctness were not tested.")
    status = "FAIL" if failures else "PASS_SELECTED_CHECKS"
    summary = {"status": status, "failures": failures, "warnings": warnings,
               "stages": {name: report["status"] for name, report in reports.items()}}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2))
    (directory / "STATUS").write_text(status + "\n")
    lines = ["# Dual RTX 3090 qualification", "", f"**{status}**", "",
             "| Stage | Status |", "|---|---|"]
    lines += [f"| {name} | {value} |" for name, value in sorted(summary["stages"].items())]
    lines += ["", "## Memory", "", "| Stage | GPU UUID | Observed peak used MiB | Peak core C |",
              "|---|---|---:|---:|"]
    for stage, report in sorted(reports.items()):
        for gpu, peaks in report.get("telemetry_summary", {}).items():
            lines.append(f"| {stage} | {gpu} | {peaks['peak_used_MiB']:.0f} | {peaks['peak_core_C']:.0f} |")
    lines += ["", "Model-stage VRAM includes weights, reserved KV blocks and runtime overhead; see cache metrics for occupancy.",
              "", "## Hardware speed", "", "| Check | GPU | Precision | GEMM TFLOP/s | Copy read+write GB/s |",
              "|---|---:|---|---:|---:|"]
    for check in reports.get("hardware", {}).get("checks", []):
        result = check.get("result")
        results = result if isinstance(result, list) else [result]
        for result in results:
            if isinstance(result, dict) and "median_tflops" in result:
                lines.append(f"| {check['name']} | {result['device']} | {result['dtype']} | {result['median_tflops']:.2f} | {result['copy_read_plus_write_GB_s']:.2f} |")
    lines += ["", "## Inference speed", "", "| Stage | Input tokens | Concurrency | Output tokens/s | p95 TTFT (s) |",
              "|---|---:|---:|---:|---:|"]
    for stage, report in sorted(reports.items()):
        for check in report["checks"]:
            result = check.get("result")
            if check["name"].startswith("inference_") and isinstance(result, dict) and "aggregate_output_tokens_s" in result:
                lines.append(f"| {stage} | {result['input_tokens']} | {result['concurrency']} | {result['aggregate_output_tokens_s']:.2f} | {result['ttft_p95_s']:.3f} |")
    lines += ["", "## Failures"] + [f"- {f}" for f in failures]
    lines += ["", "## Caveats"] + [f"- {w}" for w in warnings]
    (directory / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(f"{status}: {directory / 'SUMMARY.md'}")
    return 1 if failures else 0


def fingerprint(directory):
    config = read(directory / "configuration.json")
    # Ignore workload selection/floors; case matching below handles token/concurrency sizes.
    keys = ("MODEL", "MODEL_REVISION", "VLLM_IMAGE", "EXPECTED_CPU", "GPU0", "GPU1",
            "GPU_MEMORY_UTILIZATION", "MAX_MODEL_LEN", "MAX_NUM_SEQS", "OUTPUT_TOKENS")
    preflight = read(directory / "preflight" / "report.json")
    gpus = preflight["inventory"]["gpus"]
    images = read(directory / "server-image.json")
    return {"configuration": {k: config[k] for k in keys},
            "suite_version": preflight["suite_version"],
            "suite_sha256": preflight.get("suite_sha256"),
            "driver": preflight["inventory"]["driver"]["stdout"],
            "gpus": [{k: gpu[k] for k in ("uuid", "name", "memory.total", "power.limit")} for gpu in gpus],
            "cpu": preflight["inventory"]["cpu"]["stdout"].split("Model name:")[-1].splitlines()[0].strip(),
            "image": [{"Id": i["Id"], "RepoDigests": i.get("RepoDigests", [])} for i in images]}


def case_metrics(directory):
    metrics = {}
    for report_path in directory.glob("tp*/report.json"):
        report = read(report_path)
        for check in report["checks"]:
            result = check.get("result")
            if check["status"] != "PASS" or not isinstance(result, dict) or "aggregate_output_tokens_s" not in result:
                continue
            key = (report_path.parent.name, result["input_tokens"], result["concurrency"])
            metrics.setdefault(key, []).append(result)
    return metrics


def compare(baseline, current, max_regression):
    import statistics
    for directory in (baseline, current):
        if read(directory / "summary.json")["status"] != "PASS_SELECTED_CHECKS":
            raise ValueError(f"cannot compare failed/incomplete run {directory}")
    if fingerprint(baseline) != fingerprint(current):
        raise ValueError("configuration/hardware/image fingerprint differs; rerun a matched baseline")
    old, new = case_metrics(baseline), case_metrics(current)
    if not old or old.keys() != new.keys():
        raise ValueError("inference case sets differ or are empty")
    rows, failed = [], False
    for key in sorted(old):
        a = statistics.median(v["aggregate_output_tokens_s"] for v in old[key])
        b = statistics.median(v["aggregate_output_tokens_s"] for v in new[key])
        ta = statistics.median(v["ttft_p95_s"] for v in old[key])
        tb = statistics.median(v["ttft_p95_s"] for v in new[key])
        regression = (a - b) / a
        ttft_regression = (tb - ta) / ta
        bad = regression > max_regression or ttft_regression > max_regression
        failed |= bad
        rows.append({"stage_context_concurrency": key, "baseline_tps": a, "current_tps": b,
                     "throughput_regression": regression, "p95_ttft_regression": ttft_regression,
                     "status": "FAIL" if bad else "PASS"})
    print(json.dumps({"status": "FAIL" if failed else "PASS", "cases": rows}, indent=2))
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    summary = sub.add_parser("summarize")
    summary.add_argument("directory", type=Path)
    summary.add_argument("--exit-code", type=int, default=0)
    comparison = sub.add_parser("compare")
    comparison.add_argument("baseline", type=Path)
    comparison.add_argument("current", type=Path)
    comparison.add_argument("--max-regression", type=float, default=.20)
    args = parser.parse_args()
    if args.mode == "summarize":
        return summarize(args.directory, args.exit_code)
    if not 0 < args.max_regression < 1:
        parser.error("--max-regression must be between 0 and 1")
    try:
        return compare(args.baseline, args.current, args.max_regression)
    except (ValueError, KeyError, OSError) as e:
        parser.error(str(e))


if __name__ == "__main__":
    raise SystemExit(main())
