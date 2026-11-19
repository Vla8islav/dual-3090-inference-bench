# Dual RTX 3090 inference benchmark

A standalone Docker Compose test suite for a **generic Debian x86_64 machine
with a Ryzen 7 5700X and two 24 GiB RTX 3090s**. No lab-specific addresses,
credentials, Ansible or pre-existing inference service required.

It answers two separate questions:

1. **Memory/correctness:** can each GPU and both together reliably read/write a
   large VRAM allocation, compute, communicate, load a real model and serve the
   requested context/concurrency without corruption or OOM?
2. **Speed:** what are the compute, memory-copy, inter-GPU and end-to-end inference
   rates, and how do they compare across cards, TP=1/TP=2 and matched past runs?

## Prerequisites

- Debian 12/13, x86_64; Docker Engine and **Compose v2.24+**.
- NVIDIA Linux driver **570.26 or newer** for the pinned CUDA 12.8 image,
  and NVIDIA Container Toolkit configured for Docker.
- `python3`, `bash`, `flock` (util-linux); optional kernel-journal read access.
- Approximately **32 GiB system RAM minimum; 64 GiB recommended**. Keep swap
  pressure in mind. Free disk space: **50 GiB minimum, 80 GiB recommended** for
  the CUDA image, model cache and results. First run downloads about 15 GB of
  model weights plus a large container image; internet is needed initially.
- Exclusive access to the two selected GPUs for meaningful measurements.

Verify driver/container support before testing:

```bash
nvidia-smi
docker compose version
```

Install drivers/Docker/Toolkit using their official Debian instructions. This
repo deliberately does **not** install drivers, modify Docker, change clocks or
power caps, stop unrelated containers, or reboot the machine. Docker access
itself is effectively root-equivalent; use a trusted Docker installation.

## Quick start

```bash
git clone http://YOUR-GITLAB/master/dual-3090-inference-bench.git
cd dual-3090-inference-bench
cp .env.example .env
./run.sh preflight       # Container access + inventory; NOT qualification
./run.sh quick           # Hardware + real-model comparisons on GPU0/GPU1/TP=2
```

Default model: **Qwen/Qwen2.5-7B-Instruct**, immutable revision
`a09a35458c702b33eeacc393d103063234e8bc28`, FP16. It fits one 3090, enabling
apples-to-apples single-card and dual-card comparisons. The model is public
and requires no token. Images and dependencies come from the version-pinned
`vllm/vllm-openai:v0.10.2` image; runner and server use the same base. For
byte-identical image pinning, set `VLLM_IMAGE` to a verified `@sha256:` digest.
Resolved image IDs/digests and Compose configuration are saved in each run.

`quick` is a short workload, **not** a promise of a fast first download/startup.
Each server startup has a 30-minute readiness deadline. Compilation and model
loading are not counted as inference throughput. The three configurations run
**sequentially**, never alongside each other or the standalone VRAM tests.

## Modes

| Command | What runs |
|---|---|
| `./run.sh preflight` | Identity, GPU access, capacity, topology and process inventory |
| `./run.sh hardware` | 85% of total VRAM/card, 3 pattern passes, isolated/simultaneous memory checks; 30-second compute stages; copies and NCCL |
| `./run.sh inference` | Model smoke tests + context/concurrency sweeps on GPU0, GPU1 and TP=2 |
| `./run.sh quick` | 25% VRAM, 1 pass, 5-second compute stages, NCCL, then all inference configurations |
| `./run.sh extended` | Full hardware mode, then inference; TP=2 repeated for `SOAK_SECONDS` (default 3600) |
| `./run.sh soak` | Same as extended; set `SOAK_SECONDS=86400` in `.env` for a 24-hour inference soak |

`SOAK_SECONDS` is a minimum TP=2 sweep duration, not a hard whole-suite
deadline. It finishes the current sweep. API requests have bounded socket and
checked streaming deadlines; an already-blocked read can take an additional
socket-timeout interval to return. NCCL has a separate bounded process deadline.

The default CPU check is strict: a 5700G is **not** a 5700X. Edit `EXPECTED_CPU`
only when intentionally using another CPU. The 5700G's PCIe 3.0 link can make
dual-GPU performance materially different.

## What gets measured

### GPU memory and compute

- Allocate a configured fraction of **total** VRAM, rather than silently using
  a tiny fraction of whatever happens to remain free. Tests fail if there isn't
  enough free space, with 512 MiB reserved for temporary work.
- Write and verify zero, all-one, alternating-bit, inverse-bit and
  chunk/address-dependent patterns. Both isolated and simultaneous runs.
- FP16/BF16 GEMM with FP32-reference correctness checks and finite-output
  checks. Report median TFLOP/s; no CPU fallback is accepted.
- Device-local copy bandwidth (read + write traffic), bidirectional cross-GPU
  transfer throughput and equality verification. Transfer may be host-staged;
  peer-access support is recorded, not assumed.
- Two-rank NCCL all-reduce correctness and completed iterations/time. This is
  not `nccl-tests`' pure bus-bandwidth metric; copy throughput is reported
  separately. No NVLink bridge is assumed or required.

### Real inference and memory pressure

- Two simple deterministic semantic smoke tests using the model's chat template.
  These are liveness/obvious-corruption checks, **not** a general model-quality eval.
- Warmup excluded, then exact input lengths of **256, 2048, 4096 tokens** at
  concurrency **1, 2, 4**. Token-ID prompts avoid ambiguous character-based
  length estimates. Random leading tokens prevent repeated-prefix reuse.
- Fixed **128 output tokens** with EOS disabled for comparable throughput;
  server-reported input/output usage is checked. Performance prompts are
  synthetic, not meaningful questions; semantic smoke tests are separate.
- Streaming time to first **nonempty text chunk**, p50/p95 TTFT, p95 request
  latency, aggregate output tokens/sec, and approximate per-output-token time.
  A stream chunk can contain multiple tokens, so this is not a GPU-kernel
  decode microbenchmark. At least two requests per concurrency slot are sent;
  small samples are diagnostics, not statistically robust SLO estimates.
- Loaded-model GPU memory, per-case observed peak used VRAM, plus per-second
  vLLM KV-cache usage, running/waiting requests and preemption counters where
  exposed. vLLM preallocates VRAM/KV blocks: flat `nvidia-smi` memory during
  a context sweep does **not** mean zero KV-cache growth. Inspect cache metrics.
- OOM, failed/missing/truncated streams, usage mismatch and API errors fail;
  there are no automatic retries to conceal errors.

Default `MAX_MODEL_LEN=8192`, `MAX_NUM_SEQS=8`, GPU memory utilization 0.85.
Context sweep input + output must fit the configured model length. Edit
`CONTEXTS`, `CONCURRENCIES`, `OUTPUT_TOKENS` in `.env` to explore capacity.
Increase gradually; an intentionally impossible allocation is expected to fail.
This is a bounded supported-workload test, **not** automatic discovery of the
largest model or the exact OOM boundary.

For models that require combined VRAM, set **both** `MODEL` and its matching
immutable `MODEL_REVISION`. Quantized models must be supported by this vLLM
version (quantization is auto-detected from model metadata). The managed suite
always runs TP=1 comparisons too, so such models will intentionally fail at
the single-card stage. Use the manual TP=2-only workflow below for them.

## Results and pass/fail

Every run writes `results/<UTC timestamp>-<PID>/`:

- `SUMMARY.md`, `summary.json`, `STATUS`: overall selected-stage outcome.
- `<stage>/report.json`: detailed metrics, individual checks, arguments, GPU
  UUIDs, torch/CUDA versions, CPU/RAM/topology and co-tenant process inventory.
- `<stage>/telemetry.jsonl`: flushed per-second temperature, power, utilization,
  VRAM, negotiated PCIe generation/width and inference-server cache metrics.
- `case-*.json` under inference stages: each load case, including failed requests.
- Per-stage startup/server logs, resolved Compose YAML and image IDs/digests.
- `kernel.log`: **new** host Xid/modeset/fallen-off-bus/OOM messages, when journal
  access permits. Restricted/missing access is prominently warned, not claimed clean.

Exit 1 / `FAIL` means a selected check, server startup or host-error check
failed. `PASS_SELECTED_CHECKS` means the chosen workload completed without a
detected failure; examine warnings. An unfinished `RUNNING` file is **never a
pass**, including after power loss. A preflight-only pass proves no inference.

There is no made-up universal speed floor for a 3090. Power cap, cooling,
PCIe wiring, quantization, model/backend version and background work all matter.
By default rates are measurements. Set `MIN_TFLOPS` / `MIN_OUTPUT_TPS` for your
own minimums, or compare matched successful runs:

```bash
./run.sh compare results/BASELINE results/CURRENT --max-regression 0.20
```

This fails on a >20% throughput drop or p95-TTFT increase. It refuses unmatched
model/revision/configuration, image, suite source, driver, GPU identity/power caps or case sets.
Repeated soak sweeps are compared using medians per case. Re-run borderline
results; small request counts and uncontrolled co-tenants can add noise.

TP=2 is **not guaranteed faster** for a 7B model. Communication over PCIe can
make single-user latency worse while enabling larger models. Compare both
cards individually first, then multi-request throughput. This suite does not
benchmark two independent replicated model servers or CPU-only inference.

## Safety and operation

- Run during a maintenance window. **Do not** run VRAM tests alongside other
  GPU services. The suite neither kills nor pauses them for you.
- Defaults stop scheduling GPU work when observed **GPU core** temperature
  reaches 85 C; monitor/NVML failures also fail the run. Already-submitted
  kernels and blocking API reads cannot be instantly interrupted. The wrapper
  stops only its own model server during cleanup. Model startup is bounded by
  readiness polling but does not have a thermal-monitor container.
- 3090 GDDR6X **junction temperature** often isn't exposed by Linux tools;
  core temperature is not a substitute. No fan, clock or power modifications.
- No exhaustive VRAM certification, ECC proof, PSU/transient verification,
  CPU/RAM burn-in or automatic hardware diagnosis. GPU pattern tests cover the
  allocated region, not reserved memory. Use dedicated hardware diagnostics
  and physical inspection if failures occur.
- No privileged containers, host PID namespace, Docker socket mount or host
  API ports. The API stays on the private Compose network. The default model
  doesn't need `trust_remote_code`.
- Normal cleanup stops this Compose project's `server`, leaves model cache and
  results intact, and never runs `docker system prune` or deletes other services.
  Interrupted or powered-off runs retain their incomplete evidence.
- `.env` is sourced as shell configuration: only use a trusted local file.
  It is ignored by git. No credentials are bundled in this repository.
- Script creates output/cache directories as your UID and runs containers as
  that UID; direct Compose usage defaults to UID/GID 1000. Override
  `LOCAL_UID`/`LOCAL_GID` for other accounts. Use `bash ./run.sh` if executable
  file modes were lost copying the repo.

## Manual Compose / TP=2-only workflow

For custom workloads, with the selected GPUs otherwise idle:

```bash
mkdir -p results .cache/huggingface
export LOCAL_UID=$(id -u) LOCAL_GID=$(id -g)
docker compose --profile tests build bench
docker compose --profile tests run --rm bench preflight --output /results/manual-preflight

export TP_SIZE=2 VISIBLE_DEVICES=0,1
docker compose --profile inference up -d --wait --wait-timeout 1800 server
docker compose --profile tests run --rm bench inference --output /results/manual-tp2
docker compose --profile inference stop server
```

Use unique output directory names; existing result directories are never
overwritten. The manual workflow doesn't collect host kernel logs or an overall
run summary. For disconnected runs after downloading the model and building the
images, set `HF_HUB_OFFLINE=1` and `PULL_IMAGES=0` in `.env`.

## Development / validation

```bash
python3 -m unittest discover -s tests -v
bash -n run.sh
docker compose --profile tests --profile inference config --quiet
```

Tests use a real local mock streaming HTTP server and fake GPU snapshots; no
model or GPU is required. GitLab CI validates offline tests and Compose syntax.
Actual driver compatibility, CUDA kernels, NCCL and model execution must be
validated on the target GPU host; a CI pass is not a hardware benchmark result.
