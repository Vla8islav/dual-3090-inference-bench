#!/usr/bin/env python3
"""Portable CUDA and inference diagnostics. No dependencies until GPU stages."""
import argparse
import concurrent.futures
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import threading
import time
import urllib.request
import uuid

VERSION = "0.1.0"
GPU_FIELDS = "index,uuid,name,memory.total,memory.used,memory.free,temperature.gpu,utilization.gpu,power.draw,power.limit,pcie.link.gen.current,pcie.link.width.current"


def command(argv, timeout=15):
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return dict(returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    except (OSError, subprocess.TimeoutExpired) as e:
        return dict(returncode=-1, stdout="", stderr=str(e))


def gpu_snapshot():
    result = command(["nvidia-smi", "--query-gpu=" + GPU_FIELDS, "--format=csv,noheader,nounits"])
    if result["returncode"]:
        raise RuntimeError("nvidia-smi failed: " + result["stderr"])
    rows = []
    for values in csv.reader(result["stdout"].splitlines(), skipinitialspace=True):
        row = dict(zip(GPU_FIELDS.split(","), values))
        for key in ("memory.total", "memory.used", "memory.free", "temperature.gpu", "utilization.gpu"):
            row[key] = float(row[key])
        rows.append(row)
    if len(rows) != 2:
        raise RuntimeError(f"expected 2 GPUs, got {len(rows)}")
    return rows


def percentile(values, q):
    return sorted(values)[max(0, math.ceil(len(values) * q) - 1)] if values else None


def server_metrics(base):
    with http(base, "/metrics", timeout=5) as response:
        text = response.read().decode()
    wanted = ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc", "vllm:num_requests_running",
              "vllm:num_requests_waiting", "vllm:num_preemptions_total")
    metrics = {}
    for line in text.splitlines():
        if line.startswith(wanted):
            name, value = line.rsplit(None, 1)
            numeric = float(value)
            if math.isfinite(numeric):
                metrics[name] = numeric
    return metrics


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


class Run:
    def __init__(self, args):
        self.args = args
        self.directory = args.output
        self.directory.mkdir(parents=True, exist_ok=False)
        self.stop = threading.Event()
        self.monitor_done = threading.Event()
        self.monitor_errors = []
        self.samples = []
        self.peaks = {}
        self.data = {"schema": 1, "suite_version": VERSION, "status": "RUNNING",
                     "suite_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                     "started": dt.datetime.now(dt.timezone.utc).isoformat(), "hostname": platform.node(),
                     "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                     "checks": [], "warnings": [], "inventory": {}}
        self.thread = None
        self.save()

    def save(self):
        atomic_json(self.directory / "report.json", self.data)

    def check(self, name, fn):
        start = time.monotonic()
        try:
            self.guard()
            result = fn()
            entry = {"name": name, "status": "PASS", "result": result}
        except Exception as e:
            entry = {"name": name, "status": "FAIL", "error": str(e)}
        entry["seconds"] = time.monotonic() - start
        self.data["checks"].append(entry)
        self.save()
        print(f"{entry['status']}: {name}", flush=True)
        return entry

    def guard(self):
        if self.stop.is_set():
            raise RuntimeError("telemetry safety abort: " + "; ".join(self.monitor_errors))

    def monitor(self):
        with (self.directory / "telemetry.jsonl").open("w", buffering=1) as log:
            while not self.monitor_done.is_set():
                try:
                    rows = gpu_snapshot()
                    sample = {"time": time.time(), "gpus": rows}
                    for gpu in rows:
                        peaks = self.peaks.setdefault(gpu["uuid"], {"peak_used_MiB": 0, "peak_core_C": 0})
                        peaks["peak_used_MiB"] = max(peaks["peak_used_MiB"], gpu["memory.used"])
                        peaks["peak_core_C"] = max(peaks["peak_core_C"], gpu["temperature.gpu"])
                    if self.args.mode == "inference":
                        # Failure to collect metrics is diagnostic, not a hidden
                        # failure of otherwise successful inference requests.
                        try:
                            sample["server_metrics"] = server_metrics(self.args.api_base)
                        except Exception as e:
                            sample["server_metrics_error"] = str(e)
                    self.samples.append(sample)
                    # Bounded memory even for multi-hour runs; complete history is on disk.
                    if len(self.samples) > 3600:
                        self.samples.pop(0)
                    log.write(json.dumps(sample) + "\n")
                    if any(r["temperature.gpu"] >= self.args.max_temp for r in rows):
                        raise RuntimeError(f"GPU core temperature >= {self.args.max_temp} C")
                except Exception as e:
                    self.monitor_errors.append(str(e))
                    self.stop.set()
                    return
                self.monitor_done.wait(1)

    def start_monitor(self):
        self.thread = threading.Thread(target=self.monitor, daemon=True)
        self.thread.start()

    def finish(self):
        self.monitor_done.set()
        if self.thread:
            self.thread.join(timeout=25)
            if self.thread.is_alive():
                self.monitor_errors.append("telemetry thread did not exit")
        if self.monitor_errors:
            self.data["checks"].append({"name": "telemetry", "status": "FAIL", "errors": self.monitor_errors})
        if self.peaks:
            self.data["telemetry_summary"] = self.peaks
        self.data["status"] = "FAIL" if any(c["status"] == "FAIL" for c in self.data["checks"]) else "PASS_SELECTED_CHECKS"
        self.data["finished"] = dt.datetime.now(dt.timezone.utc).isoformat()
        self.save()
        lines = ["# Inference benchmark", "", f"**{self.data['status']}**", "",
                 "| Check | Result | Seconds |", "|---|---|---:|"]
        for c in self.data["checks"]:
            lines.append(f"| {c['name']} | {c['status']} | {c.get('seconds', 0):.2f} |")
        lines += ["", "See report.json for metrics and errors; telemetry.jsonl contains per-second samples.",
                  "Only selected stages are covered. Speed needs a matched baseline; this is not hardware certification."]
        (self.directory / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def inventory(run):
    for name, argv in {"cpu": ["lscpu"], "os": ["cat", "/etc/os-release"],
                       "host_memory": ["cat", "/proc/meminfo"], "gpu_details": ["nvidia-smi", "-q"],
                       "driver": ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                       "topology": ["nvidia-smi", "topo", "-m"],
                       "processes": ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv"]}.items():
        run.data["inventory"][name] = command(argv)
    rows = gpu_snapshot()
    run.data["inventory"]["gpus"] = rows
    failures = []
    cpu = run.data["inventory"]["cpu"]
    if cpu["returncode"] or run.args.expected_cpu.lower() not in cpu["stdout"].lower():
        failures.append(f"CPU identity differs from expected {run.args.expected_cpu}")
    if any("3090" not in g["name"] or g["memory.total"] < 24000 for g in rows):
        failures.append("expected two 24 GiB RTX 3090 cards")
    run.data["warnings"].append("Kernel Xid/modeset/OOM checks are collected by the host run.sh, not this unprivileged container.")
    if failures:
        raise RuntimeError("; ".join(failures))
    return rows


def load_torch(run):
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 2:
        raise RuntimeError("exactly two visible CUDA GPUs required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    run.data["inventory"]["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda,
        "devices": [str(torch.cuda.get_device_properties(i)) for i in range(2)]}
    return torch


def memory_test(torch, device, run):
    """Check a configurable fraction of TOTAL VRAM, not merely remaining VRAM."""
    with torch.cuda.device(device), torch.inference_mode():
        free, total = torch.cuda.mem_get_info(device)
        target = int(total * run.args.memory_fraction)
        reserve = 512 * 1024 ** 2
        if free < target + reserve:
            raise RuntimeError(f"GPU {device}: need {target + reserve} free bytes, have {free}; do not run alongside model servers")
        elements = 8 * 1024 * 1024  # 32 MiB chunks bound temporary verification memory.
        count = max(1, target // (elements * 4))
        chunks = [torch.empty(elements, device=device, dtype=torch.int32) for _ in range(count)]
        offsets = torch.arange(elements, device=device, dtype=torch.int32)
        patterns = [("zeros", 0), ("ones", -1), ("alternating", 0x55555555), ("inverse", -1431655766)]
        start = time.perf_counter()
        try:
            for pass_id in range(run.args.memory_passes):
                for name, value in patterns:
                    for chunk in chunks:
                        run.guard()
                        chunk.fill_(value)
                    torch.cuda.synchronize(device)
                    for i, chunk in enumerate(chunks):
                        run.guard()
                        if not torch.all(chunk == value).item():
                            raise RuntimeError(f"GPU {device}: memory mismatch {name} chunk {i} pass {pass_id}")
                # Address-dependent pattern detects more than constant/stuck bits.
                for i, chunk in enumerate(chunks):
                    run.guard()
                    chunk.copy_(offsets ^ (i + pass_id * 65537))
                torch.cuda.synchronize(device)
                for i, chunk in enumerate(chunks):
                    run.guard()
                    if not torch.equal(chunk, offsets ^ (i + pass_id * 65537)):
                        raise RuntimeError(f"GPU {device}: address-pattern mismatch chunk {i}")
            return {"device": device, "tested_bytes": count * elements * 4, "total_bytes": total,
                    "fraction_of_total": count * elements * 4 / total, "passes": run.args.memory_passes,
                    "patterns_per_pass": 5, "seconds": time.perf_counter() - start}
        finally:
            del chunks, offsets
            torch.cuda.empty_cache()


def speed_test(torch, device, dtype_name, run):
    with torch.cuda.device(device), torch.inference_mode():
        dtype = getattr(torch, dtype_name)
        a = torch.randn(256, 256, device=device, dtype=dtype) / 16
        b = torch.randn_like(a) / 16
        torch.testing.assert_close((a @ b).float(), a.float() @ b.float(),
                                   rtol=.03 if dtype_name == "bfloat16" else .01, atol=.01)
        n = run.args.matrix_size
        a = torch.randn(n, n, device=device, dtype=dtype) / 64
        b = torch.randn_like(a) / 64
        out = torch.empty_like(a)
        for _ in range(5):
            torch.mm(a, b, out=out)
        torch.cuda.synchronize(device)
        rates = []
        deadline = time.monotonic() + run.args.seconds
        while time.monotonic() < deadline:
            run.guard()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):
                torch.mm(a, b, out=out)
            end.record()
            end.synchronize()
            rates.append(20 * n ** 3 / (start.elapsed_time(end) / 1000) / 1e12)
            if not torch.isfinite(out).all().item():
                raise RuntimeError("non-finite matrix output")
        torch.testing.assert_close(out[:16, :16].float(), a[:16].float() @ b[:, :16].float(), rtol=.04, atol=.02)
        median = statistics.median(rates)
        if median < run.args.min_tflops:
            raise RuntimeError(f"{median:.2f} TFLOP/s below configured floor {run.args.min_tflops}")
        source = torch.randn(32 * 1024 * 1024, device=device, dtype=torch.float32)
        dest = torch.empty_like(source)
        dest.copy_(source)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):
            dest.copy_(source)
        end.record()
        end.synchronize()
        torch.testing.assert_close(dest, source, rtol=0, atol=0)
        return {"device": device, "dtype": dtype_name, "median_tflops": median,
                "copy_read_plus_write_GB_s": 40 * source.numel() * 4 / (start.elapsed_time(end) / 1000) / 1e9}


def transfers(torch, run):
    results = []
    for src, dst in [(0, 1), (1, 0)]:
        run.guard()
        a = torch.arange(4 * 1024 * 1024, device=src, dtype=torch.float32)
        b = torch.empty_like(a, device=dst)
        b.copy_(a)
        torch.cuda.synchronize(dst)
        start = time.perf_counter()
        for _ in range(20):
            b.copy_(a)
        torch.cuda.synchronize(src)
        torch.cuda.synchronize(dst)
        elapsed = time.perf_counter() - start
        torch.testing.assert_close(a.cpu(), b.cpu(), rtol=0, atol=0)
        results.append({"src": src, "dst": dst, "GB_s": 20 * a.numel() * 4 / elapsed / 1e9,
                        "peer_access": torch.cuda.can_device_access_peer(src, dst)})
    return results


def nccl_worker(rank, rendezvous, output, seconds):
    import torch
    import torch.distributed as dist
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=dt.timedelta(seconds=60))
    try:
        tensor = torch.empty(1024 * 1024, device=rank)
        start, iterations = time.perf_counter(), 0
        while time.perf_counter() - start < seconds:
            tensor.fill_(rank + 1)
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
            if not torch.all(tensor == 3).item():
                raise RuntimeError("incorrect NCCL all-reduce")
            iterations += 1
        atomic_json(Path(output) / f"nccl-rank-{rank}.json", {"rank": rank, "iterations": iterations,
                    "seconds": time.perf_counter() - start, "payload_bytes": tensor.numel() * 4})
    finally:
        dist.destroy_process_group()


def nccl_test(torch, run):
    rendezvous = "file://" + str((run.directory / "nccl-rendezvous").resolve())
    context = torch.multiprocessing.spawn(nccl_worker,
        args=(rendezvous, str(run.directory.resolve()), run.args.seconds), nprocs=2, join=False)
    deadline = time.monotonic() + run.args.seconds + 90
    try:
        while not context.join(timeout=1):
            run.guard()
            if time.monotonic() > deadline:
                raise RuntimeError("NCCL wall-clock timeout")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
    return [json.loads((run.directory / f"nccl-rank-{r}.json").read_text()) for r in range(2)]


def hardware(run):
    torch = load_torch(run)
    for device in range(2):
        run.check(f"memory_gpu{device}", lambda d=device: memory_test(torch, d, run))
    def parallel(fn):
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            return list(pool.map(fn, range(2)))
    run.check("memory_both", lambda: parallel(lambda d: memory_test(torch, d, run)))
    for dtype in ("float16", "bfloat16"):
        for device in range(2):
            run.check(f"speed_{dtype}_gpu{device}", lambda d=device, t=dtype: speed_test(torch, d, t, run))
        run.check(f"speed_{dtype}_both", lambda t=dtype: parallel(lambda d: speed_test(torch, d, t, run)))
    run.check("cross_gpu_copy", lambda: transfers(torch, run))
    run.check("nccl_all_reduce", lambda: nccl_test(torch, run))


def http(base, path, payload=None, timeout=180):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base.rstrip("/") + path, data=data, headers={"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)


def json_request(base, path, payload=None, timeout=180):
    with http(base, path, payload, timeout) as response:
        return json.load(response)


def validate_completion(result):
    if result.get("error") or not result.get("choices"):
        raise RuntimeError("API error or missing choices")
    text = result["choices"][0].get("text", "")
    usage = result.get("usage") or {}
    if not text.strip() or not isinstance(usage.get("completion_tokens"), int) or usage["completion_tokens"] < 1:
        raise RuntimeError("empty output or missing positive output token usage")
    return text, usage


def smoke(run):
    cases = [("What is 2 + 2? Reply with just the integer.", r"^\s*4\s*[.!]?\s*$"),
             ("What is the capital of France? Reply with just the city name.", r"^\s*Paris\s*[.!]?\s*$")]
    results = []
    for question, pattern in cases:
        result = json_request(run.args.api_base, "/v1/chat/completions", {
            "model": "benchmark", "messages": [{"role": "user", "content": question}],
            "temperature": 0, "max_tokens": 16}, run.args.timeout)
        if not result.get("choices"):
            raise RuntimeError("missing smoke-test response")
        text = result["choices"][0].get("message", {}).get("content", "")
        if not re.fullmatch(pattern, text, re.IGNORECASE):
            raise RuntimeError(f"smoke answer mismatch: {question!r} -> {text!r}")
        results.append({"question": question, "answer": text})
    return results


def stream_request(run, input_tokens, index):
    run.guard()
    # Integer token IDs give an EXACT input length. Use a random prefix to
    # prevent reuse if the operator enables prefix caching on the server.
    import random
    rng = random.Random(uuid.uuid4().int)
    prefix = [rng.randrange(1000, 20000) for _ in range(min(32, input_tokens))]
    prompt = prefix + [1000 + i % 1000 for i in range(input_tokens - len(prefix))]
    payload = {"model": "benchmark", "prompt": prompt, "max_tokens": run.args.output_tokens,
               "temperature": 0, "ignore_eos": True, "stream": True,
               "stream_options": {"include_usage": True}}
    start = time.perf_counter()
    first = None
    text_parts, usage, done = [], None, False
    # urllib's socket timeout is an inactivity timeout, not an overall deadline.
    # Enforce the overall deadline as chunks arrive, in addition to socket timeout.
    with http(run.args.api_base, "/v1/completions", payload, run.args.timeout) as response:
        for raw in response:
            now = time.perf_counter()
            if now - start > run.args.timeout:
                raise TimeoutError("request wall-clock deadline exceeded")
            run.guard()
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                done = True
                break
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            if event.get("usage"):
                usage = event["usage"]
            for choice in event.get("choices", []):
                text = choice.get("text", "")
                if text:
                    first = first if first is not None else now
                    text_parts.append(text)
    elapsed = time.perf_counter() - start
    if not done or first is None or not usage or usage.get("prompt_tokens") != input_tokens:
        raise RuntimeError("incomplete stream, empty output, or input token count mismatch")
    tokens = usage.get("completion_tokens")
    if tokens != run.args.output_tokens:
        raise RuntimeError(f"expected {run.args.output_tokens} output tokens, got {tokens}; context/capacity error")
    ttft = first - start
    return {"request": index, "input_tokens": input_tokens, "output_tokens": tokens,
            "seconds": elapsed, "ttft_seconds": ttft,
            "approx_time_per_output_token_seconds": (elapsed - ttft) / (tokens - 1) if tokens > 1 else None,
            "output_sha256": hashlib.sha256("".join(text_parts).encode()).hexdigest()}


def inference_case(run, context, concurrency):
    run.guard()
    before = gpu_snapshot()
    requests = max(run.args.requests, concurrency * 2)
    sample_start = time.time()
    started = time.perf_counter()
    results, errors = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(stream_request, run, context, i) for i in range(requests)]
        for future in concurrent.futures.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                errors.append(str(e))
    elapsed = time.perf_counter() - started
    after = gpu_snapshot()
    metrics = {"input_tokens": context, "concurrency": concurrency, "wall_seconds": elapsed,
               "requests": results, "errors": errors, "gpu_before": before, "gpu_after": after,
               "aggregate_output_tokens_s": sum(r["output_tokens"] for r in results) / elapsed,
               "ttft_p50_s": percentile([r["ttft_seconds"] for r in results], .5),
               "ttft_p95_s": percentile([r["ttft_seconds"] for r in results], .95),
               "latency_p95_s": percentile([r["seconds"] for r in results], .95)}
    samples = [s for s in list(run.samples) if s["time"] >= sample_start]
    metrics["observed_peak_used_MiB"] = {
        gpu["uuid"]: max([gpu["memory.used"], after[i]["memory.used"]] +
                          [s["gpus"][i]["memory.used"] for s in samples])
        for i, gpu in enumerate(before)}
    metrics["observed_peak_server_metrics"] = {
        key: max(s.get("server_metrics", {}).get(key, 0) for s in samples)
        for key in {key for s in samples for key in s.get("server_metrics", {})}}
    atomic_json(run.directory / f"case-{context}-{concurrency}-{uuid.uuid4().hex[:8]}.json", metrics)
    if errors:
        raise RuntimeError(f"{len(errors)}/{requests} requests failed: {errors[:3]}")
    if metrics["aggregate_output_tokens_s"] < run.args.min_output_tps:
        raise RuntimeError("output token rate below configured baseline floor")
    return metrics


def inference(run):
    models = json_request(run.args.api_base, "/v1/models", timeout=run.args.timeout)
    run.data["inventory"]["served_models"] = models
    if "benchmark" not in [m["id"] for m in models.get("data", [])]:
        raise RuntimeError("expected served model alias 'benchmark'")
    if run.check("model_smoke", lambda: smoke(run))["status"] == "FAIL":
        return
    # Exclude initial compilation/model warmup from measured cases.
    if run.check("model_warmup", lambda: stream_request(run, min(run.args.contexts), -1))["status"] == "FAIL":
        return
    run.data["inventory"]["loaded_model_gpu_memory"] = gpu_snapshot()
    deadline = time.monotonic() + run.args.soak_seconds
    iteration = 0
    while True:
        for context in run.args.contexts:
            for concurrency in run.args.concurrencies:
                result = run.check(f"inference_{iteration}_ctx{context}_c{concurrency}",
                                   lambda n=context, c=concurrency: inference_case(run, n, c))
                if result["status"] == "FAIL":
                    return  # Fail fast; don't drive an unstable service harder.
        iteration += 1
        if time.monotonic() >= deadline:
            break
    return {"sweep_iterations": iteration}


def positive_list(value):
    values = [int(v) for v in value.split(",")]
    if not values or min(values) <= 0:
        raise argparse.ArgumentTypeError("comma-separated positive integers required")
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["preflight", "hardware", "inference"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--expected-cpu", default="5700X")
    p.add_argument("--max-temp", type=float, default=85)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--memory-fraction", type=float, default=.25)
    p.add_argument("--memory-passes", type=int, default=1)
    p.add_argument("--matrix-size", type=int, default=4096)
    p.add_argument("--min-tflops", type=float, default=0)
    p.add_argument("--min-output-tps", type=float, default=0)
    p.add_argument("--api-base", default="http://server:8000")
    p.add_argument("--contexts", type=positive_list, default=[256, 2048, 4096])
    p.add_argument("--concurrencies", type=positive_list, default=[1, 2, 4])
    p.add_argument("--output-tokens", type=int, default=128)
    p.add_argument("--requests", type=int, default=4)
    p.add_argument("--timeout", type=float, default=180)
    p.add_argument("--soak-seconds", type=float, default=0)
    args = p.parse_args()
    if not 0 < args.memory_fraction <= .9:
        p.error("memory fraction must be > 0 and <= 0.9")
    if min(args.seconds, args.memory_passes, args.matrix_size, args.max_temp, args.requests, args.timeout, args.output_tokens) <= 0:
        p.error("durations, counts, sizes and max temperature must be positive")
    if min(args.soak_seconds, args.min_tflops, args.min_output_tps) < 0:
        p.error("soak duration and rate floors cannot be negative")
    run = Run(args)
    try:
        entry = run.check("preflight", lambda: inventory(run))
        if entry["status"] == "PASS":
            run.start_monitor()
            if args.mode == "hardware":
                run.check("hardware_suite", lambda: hardware(run))
            elif args.mode == "inference":
                run.check("inference_suite", lambda: inference(run))
    except BaseException as e:
        run.data["checks"].append({"name": "interrupted", "status": "FAIL", "error": repr(e)})
        raise
    finally:
        run.finish()
    print(f"{run.data['status']}: {run.directory}")
    return 1 if run.data["status"] == "FAIL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
