#!/usr/bin/env bash
# Explicitly opt-in load tests. Only manages this project's benchmark server.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
MODE=${1:-quick}
if [[ "$MODE" == compare ]]; then
  shift
  exec python3 report.py compare "$@"
fi
case "$MODE" in preflight|hardware|inference|quick|extended|soak) ;; *)
  echo "Usage: ./run.sh {preflight|hardware|inference|quick|extended|soak}" >&2
  echo "       ./run.sh compare BASELINE_RUN NEW_RUN [--max-regression 0.20]" >&2
  exit 2 ;;
esac
if [[ $# -gt 1 ]]; then echo "Unexpected arguments" >&2; exit 2; fi
if [[ -f .env ]]; then set -a; source .env; set +a; fi
export LOCAL_UID=${LOCAL_UID:-$(id -u)} LOCAL_GID=${LOCAL_GID:-$(id -g)}
export COMPOSE_PROJECT_NAME=${COMPOSE_PROJECT_NAME:-dual3090bench-${LOCAL_UID}}
export MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
export MODEL_REVISION=${MODEL_REVISION:-a09a35458c702b33eeacc393d103063234e8bc28}
export VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:v0.10.2}
export EXPECTED_CPU=${EXPECTED_CPU:-5700X}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192} MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
export GPU0=${GPU0:-0} GPU1=${GPU1:-1}
export CONTEXTS=${CONTEXTS:-256,2048,4096} CONCURRENCIES=${CONCURRENCIES:-1,2,4}
export OUTPUT_TOKENS=${OUTPUT_TOKENS:-128}
export MAX_TEMP=${MAX_TEMP:-85} MIN_TFLOPS=${MIN_TFLOPS:-0} MIN_OUTPUT_TPS=${MIN_OUTPUT_TPS:-0}
export TP_SIZE=2 VISIBLE_DEVICES=0,1
for tool in docker python3 flock; do command -v "$tool" >/dev/null || { echo "Missing prerequisite: $tool" >&2; exit 2; }; done
docker compose version >/dev/null
mkdir -p .cache/huggingface results
exec 9>.cache/suite.lock
flock -n 9 || { echo "Another suite run holds .cache/suite.lock" >&2; exit 2; }
DC=(docker compose --project-name "$COMPOSE_PROJECT_NAME")
if [[ -n "$("${DC[@]}" --profile inference ps --status running -q server)" ]]; then
  echo "This project's benchmark server is already running; stop it explicitly before starting a managed run." >&2
  exit 2
fi
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$$
export RUN_ID
OUT="results/$RUN_ID"
mkdir "$OUT"
printf 'RUNNING\n' > "$OUT/STATUS"
date -u --iso-8601=seconds > "$OUT/started.txt"
cat /proc/sys/kernel/random/boot_id > "$OUT/boot-id.txt"
python3 - "$OUT" "$MODE" <<'PY'
import json, os, pathlib, sys
keys = ['MODEL', 'MODEL_REVISION', 'VLLM_IMAGE', 'EXPECTED_CPU', 'GPU0', 'GPU1',
        'GPU_MEMORY_UTILIZATION', 'MAX_MODEL_LEN', 'MAX_NUM_SEQS', 'CONTEXTS',
        'CONCURRENCIES', 'OUTPUT_TOKENS', 'MAX_TEMP', 'MIN_TFLOPS', 'MIN_OUTPUT_TPS']
data = {k: os.environ[k] for k in keys}
data['mode'] = sys.argv[2]
pathlib.Path(sys.argv[1], 'configuration.json').write_text(json.dumps(data, indent=2))
PY
CURSOR=""
if command -v journalctl >/dev/null; then
  journalctl -k -n 0 --show-cursor --no-pager > "$OUT/journal-start.txt" 2> "$OUT/journal-start.err" || true
  CURSOR=$(sed -n 's/^-- cursor: //p' "$OUT/journal-start.txt")
fi
OWN_SERVER=0
cleanup() {
  code=$?
  trap - EXIT INT TERM
  set +e
  if [[ "$OWN_SERVER" == 1 ]]; then
    "${DC[@]}" --profile inference logs --no-color server > "$OUT/server-final.log" 2>&1
    "${DC[@]}" --profile inference stop -t 30 server >> "$OUT/lifecycle.log" 2>&1
    if [[ $? != 0 ]]; then code=1; fi
  fi
  if [[ -n "$CURSOR" ]]; then
    journalctl -k --after-cursor="$CURSOR" --no-pager > "$OUT/kernel.log" 2> "$OUT/kernel.err"
    if [[ $? != 0 ]]; then printf 'Journal read failed\n' >> "$OUT/kernel.err"; fi
  fi
  if [[ "$(cat /proc/sys/kernel/random/boot_id)" != "$(cat "$OUT/boot-id.txt")" ]]; then
    echo 'Host boot ID changed during run' > "$OUT/host-error.txt"
    code=1
  fi
  python3 report.py summarize "$OUT" --exit-code "$code"
  summary_code=$?
  echo "Results: $PWD/$OUT"
  exit "$summary_code"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
"${DC[@]}" --profile tests --profile inference config > "$OUT/compose-resolved.yaml"
if [[ "${PULL_IMAGES:-1}" == 1 ]]; then
  docker pull "$VLLM_IMAGE" 2>&1 | tee "$OUT/image-pull.log"
  "${DC[@]}" --profile tests build --pull bench 2>&1 | tee "$OUT/build.log"
else
  "${DC[@]}" --profile tests build --pull=false bench 2>&1 | tee "$OUT/build.log"
fi
docker image inspect "$VLLM_IMAGE" > "$OUT/server-image.json"
docker image inspect dual3090bench-runner:0.1.0 > "$OUT/runner-image.json"
run_bench() {
  local stage=$1
  shift
  "${DC[@]}" --profile tests run --rm --no-deps bench "$@" \
    --expected-cpu "$EXPECTED_CPU" --max-temp "$MAX_TEMP" --output "/results/$RUN_ID/$stage" \
    2>&1 | tee "$OUT/$stage.log"
}
run_bench preflight preflight
if [[ "$MODE" == preflight ]]; then exit 0; fi
if [[ "$MODE" == hardware || "$MODE" == quick || "$MODE" == extended || "$MODE" == soak ]]; then
  fraction=.25; seconds=5; passes=1
  if [[ "$MODE" == hardware || "$MODE" == extended || "$MODE" == soak ]]; then
    fraction=.85; seconds=30; passes=3
  fi
  run_bench hardware hardware --memory-fraction "$fraction" --memory-passes "$passes" \
    --seconds "$seconds" --min-tflops "$MIN_TFLOPS"
fi
if [[ "$MODE" == hardware ]]; then exit 0; fi
start_and_test() {
  local stage=$1
  export TP_SIZE=$2 VISIBLE_DEVICES=$3
  local soak=$4
  OWN_SERVER=1
  "${DC[@]}" --profile inference config > "$OUT/$stage-compose.yaml"
  "${DC[@]}" --profile inference up -d --no-deps --force-recreate server >> "$OUT/lifecycle.log" 2>&1
  # Startup can include a multi-GB model download. Health is polled with a
  # finite 30-minute bound; don't declare readiness from a running container.
  local ready=0
  for ((attempt=0; attempt<900; attempt++)); do
    local container status
    container=$("${DC[@]}" --profile inference ps -a -q server)
    status=$(docker inspect --format '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{end}}' "$container")
    case "$status" in
      'running healthy') ready=1; break ;;
      exited*|dead*|*unhealthy*) break ;;
    esac
    sleep 2
  done
  "${DC[@]}" --profile inference logs --no-color server > "$OUT/$stage-startup.log" 2>&1
  if [[ "$ready" != 1 ]]; then echo "Model server failed startup: $OUT/$stage-startup.log" >&2; return 1; fi
  run_bench "$stage" inference --contexts "$CONTEXTS" --concurrencies "$CONCURRENCIES" \
    --output-tokens "$OUTPUT_TOKENS" --min-output-tps "$MIN_OUTPUT_TPS" --soak-seconds "$soak"
  "${DC[@]}" --profile inference logs --no-color server > "$OUT/$stage-server.log" 2>&1
  "${DC[@]}" --profile inference stop -t 30 server >> "$OUT/lifecycle.log" 2>&1
  OWN_SERVER=0
}
start_and_test tp1-gpu0 1 0 0
start_and_test tp1-gpu1 1 1 0
SOAK=0
if [[ "$MODE" == extended || "$MODE" == soak ]]; then SOAK=${SOAK_SECONDS:-3600}; fi
start_and_test tp2 2 0,1 "$SOAK"
