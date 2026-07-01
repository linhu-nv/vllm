#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# AGGREGATED harness, pure shell (vllm + dynamo.select_service only).
#   - 2 aggregated vLLM replicas, tensor-parallel=2 each (4 GPUs)
#   - dynamo.select_service runs the KV router automatically (no etcd/NATS)
#   - workers registered with `curl POST /workers` (no Python glue)
#   - redirection per request = curl: tokenize -> /select -> forward to worker
#
# select_service is ADVISORY: it returns which worker, it does NOT proxy the
# request. The forwarding below is the "service redirection", done client-side.
#
# Deps: vllm, curl, jq.   Usage: ./agg.sh [--model NAME] [--block-size N]
set -euo pipefail

MODEL="Qwen/Qwen3-0.6B-FP8"
BLOCK_SIZE=64
SELECT_PORT=8092
CT='content-type: application/json'

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --block-size) BLOCK_SIZE="$2"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^#//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

# Block-hash determinism so the router's hashes match vLLM's KV-event hashes.
export PYTHONHASHSEED=0
trap 'echo "[cleanup] stopping..."; kill 0 2>/dev/null || true' EXIT

# --- worker layout: id -> "gpus http_port zmq_port" -------------------------
launch_worker() {  # $1=gpus $2=http $3=zmq
    echo "[launch] worker gpus=$1 http=$2 zmq=$3 tp=2"
    CUDA_VISIBLE_DEVICES="$1" PYTHONHASHSEED=0 \
    vllm serve "$MODEL" \
        --port "$2" \
        --block-size "$BLOCK_SIZE" \
        --tensor-parallel-size 2 \
        --enforce-eager --disable-log-requests \
        --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:$3\",\"enable_kv_cache_events\":true}" &
}

wait_http() {  # $1=url  $2=label
    echo -n "[wait ] $2 "
    until curl -sf "$1" >/dev/null 2>&1; do echo -n "."; sleep 2; done
    echo " ready"
}

register() {  # $1=worker_id $2=http $3=zmq
    curl -sf -X POST "http://127.0.0.1:$SELECT_PORT/workers" -H "$CT" -d "{
        \"worker_id\": $1,
        \"model_name\": \"$MODEL\",
        \"endpoint\": \"http://127.0.0.1:$2\",
        \"kv_events_endpoint\": \"tcp://127.0.0.1:$3\",
        \"block_size\": $BLOCK_SIZE,
        \"data_parallel_size\": 1
    }" >/dev/null
    echo "[upsert] worker $1 -> pool '$MODEL'"
}

route_one() {  # $1=prompt
    local prompt="$1" tokens sel ep wid
    # 1) tokenize on any worker (same model) to get token_ids for the router
    tokens=$(curl -sf -X POST "http://127.0.0.1:8001/tokenize" -H "$CT" \
        -d "{\"model\":\"$MODEL\",\"prompt\":$(jq -Rn --arg p "$prompt" '$p')}" | jq -c '.tokens')
    # 2) ask the router for the best worker by KV overlap + load
    sel=$(curl -sf -X POST "http://127.0.0.1:$SELECT_PORT/select" -H "$CT" \
        -d "{\"model_name\":\"$MODEL\",\"token_ids\":$tokens}")
    ep=$(echo "$sel" | jq -r '.endpoint')
    wid=$(echo "$sel" | jq -r '.worker_id')
    echo "  -> worker $wid  overlap=$(echo "$sel" | jq -c '.overlap')  $ep"
    # 3) forward the actual request to the chosen worker
    curl -sf -X POST "$ep/v1/completions" -H "$CT" \
        -d "{\"model\":\"$MODEL\",\"prompt\":$(jq -Rn --arg p "$prompt" '$p'),\"max_tokens\":64,\"temperature\":0}" \
        | jq -r '.choices[0].text' | head -c 100
    echo
}

# --- bring everything up ----------------------------------------------------
launch_worker "0,1" 8001 20081
launch_worker "2,3" 8002 20082
python -m dynamo.select_service --port "$SELECT_PORT" &

wait_http "http://127.0.0.1:8001/health" ":8001"
wait_http "http://127.0.0.1:8002/health" ":8002"
wait_http "http://127.0.0.1:$SELECT_PORT/health" "select_service:$SELECT_PORT"

register 1 8001 20081
register 2 8002 20082

# --- demo: two rounds; round 2 reuses prefixes cached in round 1 ------------
PROMPTS=(
    "Explain KV cache reuse in one sentence."
    "Explain KV cache reuse in one sentence, then give an example."
    "What is tensor parallelism?"
)
for round in 1 2; do
    echo "===== round $round (agg) ====="
    for p in "${PROMPTS[@]}"; do echo "[req] $p"; route_one "$p"; done
    sleep 1   # let KV events propagate into the router index
done

echo "[done] services still running; Ctrl-C to tear down. Manual route example:"
echo "  curl -s :$SELECT_PORT/workers | jq"
wait