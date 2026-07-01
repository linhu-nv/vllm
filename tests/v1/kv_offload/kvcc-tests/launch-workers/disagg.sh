#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# DISAGGREGATED harness, pure shell (vllm + dynamo.select_service only).
#   - 2 prefill + 2 decode vLLM workers, TP=1 each (2P2D, 4 GPUs)
#   - dynamo.select_service runs the KV router automatically (no etcd/NATS)
#   - prefill and decode are separate pools (distinct model_name)
#   - KV moves prefill->decode over vLLM's NIXL (dynamo not in that path)
#   - redirection per request = curl: tokenize -> /select prefill ->
#     prefill request -> /select_and_reserve decode -> decode request -> free
#
# select_service is ADVISORY (returns which worker, does not proxy). The
# prefill->decode handshake (kv_transfer_params) is vLLM's NIXL protocol and is
# VERSION-DEPENDENT -- adjust field names to your vLLM build.
#
# Deps: vllm, curl, jq.   Usage: ./disagg.sh [--model NAME] [--block-size N]
set -euo pipefail

MODEL="Qwen/Qwen3-0.6B-FP8"
BLOCK_SIZE=64
SELECT_PORT=8092
CT='content-type: application/json'
KVT='{"kv_connector":"NixlConnector","kv_role":"kv_both"}'

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2 ;;
        --block-size) BLOCK_SIZE="$2"; shift 2 ;;
        -h|--help) grep '^#' "$0" | sed 's/^#//'; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 1 ;;
    esac
done

export PYTHONHASHSEED=0
trap 'echo "[cleanup] stopping..."; kill 0 2>/dev/null || true' EXIT

PREFILL_POOL="${MODEL}::prefill"
DECODE_POOL="${MODEL}::decode"

launch_prefill() {  # $1=gpu $2=http $3=zmq $4=nixl
    echo "[launch] prefill gpu=$1 http=$2 zmq=$3 nixl=$4"
    CUDA_VISIBLE_DEVICES="$1" PYTHONHASHSEED=0 VLLM_NIXL_SIDE_CHANNEL_PORT="$4" \
    vllm serve "$MODEL" \
        --port "$2" --block-size "$BLOCK_SIZE" --tensor-parallel-size 1 \
        --enforce-eager --disable-log-requests \
        --kv-transfer-config "$KVT" \
        --kv-events-config "{\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:$3\",\"enable_kv_cache_events\":true}" &
}

launch_decode() {  # $1=gpu $2=http $3=nixl
    echo "[launch] decode  gpu=$1 http=$2 nixl=$3"
    CUDA_VISIBLE_DEVICES="$1" PYTHONHASHSEED=0 VLLM_NIXL_SIDE_CHANNEL_PORT="$3" \
    vllm serve "$MODEL" \
        --port "$2" --block-size "$BLOCK_SIZE" --tensor-parallel-size 1 \
        --enforce-eager --disable-log-requests \
        --kv-transfer-config "$KVT" &
}

wait_http() {  # $1=url $2=label
    echo -n "[wait ] $2 "
    until curl -sf "$1" >/dev/null 2>&1; do echo -n "."; sleep 2; done
    echo " ready"
}

register() {  # $1=id $2=http $3=pool $4=zmq(optional)
    local body="{\"worker_id\":$1,\"model_name\":\"$3\",\"endpoint\":\"http://127.0.0.1:$2\",\"block_size\":$BLOCK_SIZE,\"data_parallel_size\":1"
    [[ -n "${4:-}" ]] && body="$body,\"kv_events_endpoint\":\"tcp://127.0.0.1:$4\""
    body="$body}"
    curl -sf -X POST "http://127.0.0.1:$SELECT_PORT/workers" -H "$CT" -d "$body" >/dev/null
    echo "[upsert] worker $1 -> pool '$3'"
}

route_one() {  # $1=prompt
    local prompt="$1" pjson tokens psel pep dsel dep rid pr kvp dreq
    pjson=$(jq -Rn --arg p "$prompt" '$p')
    # 1) tokenize (any worker serves the same model)
    tokens=$(curl -sf -X POST "http://127.0.0.1:8001/tokenize" -H "$CT" \
        -d "{\"model\":\"$MODEL\",\"prompt\":$pjson}" | jq -c '.tokens')
    # 2) KV-overlap-aware PREFILL worker
    psel=$(curl -sf -X POST "http://127.0.0.1:$SELECT_PORT/select" -H "$CT" \
        -d "{\"model_name\":\"$PREFILL_POOL\",\"token_ids\":$tokens}")
    pep=$(echo "$psel" | jq -r '.endpoint')
    # 3) load-based DECODE worker (reserve its load)
    dsel=$(curl -sf -X POST "http://127.0.0.1:$SELECT_PORT/select_and_reserve" -H "$CT" \
        -d "{\"model_name\":\"$DECODE_POOL\",\"token_ids\":$tokens,\"expected_output_tokens\":64}")
    dep=$(echo "$dsel" | jq -r '.endpoint')
    rid=$(echo "$dsel" | jq -r '.reservation_id')
    echo "  -> prefill $(echo "$psel" | jq -r '.worker_id') (overlap=$(echo "$psel" | jq -c '.overlap')) | decode $(echo "$dsel" | jq -r '.worker_id')"

    # 4) prefill request: stage KV for remote pull (NIXL). VERSION-DEPENDENT.
    pr=$(curl -sf -X POST "$pep/v1/completions" -H "$CT" \
        -d "{\"model\":\"$MODEL\",\"prompt\":$pjson,\"max_tokens\":1,\"temperature\":0,\"kv_transfer_params\":{\"do_remote_decode\":true}}")
    kvp=$(echo "$pr" | jq -c '.kv_transfer_params // {}')
    # 5) decode request: pull KV and generate
    dreq=$(jq -nc --argjson kv "$kvp" --argjson tok 64 --arg m "$MODEL" --argjson pr "$pjson" \
        '{model:$m, prompt:$pr, max_tokens:$tok, temperature:0, kv_transfer_params:({do_remote_prefill:true} + $kv)}')
    curl -sf -X POST "$dep/v1/completions" -H "$CT" -d "$dreq" | jq -r '.choices[0].text' | head -c 100
    echo
    # 6) release the reservation's tracked load
    curl -sf -X DELETE "http://127.0.0.1:$SELECT_PORT/reservations/$rid" >/dev/null || true
}

# --- bring everything up ----------------------------------------------------
launch_prefill 0 8001 20081 20091
launch_prefill 1 8002 20082 20092
launch_decode  2 8003       20093
launch_decode  3 8004       20094
python -m dynamo.select_service --port "$SELECT_PORT" &

for port in 8001 8002 8003 8004; do wait_http "http://127.0.0.1:$port/health" ":$port"; done
wait_http "http://127.0.0.1:$SELECT_PORT/health" "select_service:$SELECT_PORT"

register 1 8001 "$PREFILL_POOL" 20081
register 2 8002 "$PREFILL_POOL" 20082
register 3 8003 "$DECODE_POOL"
register 4 8004 "$DECODE_POOL"

PROMPTS=(
    "Explain KV cache reuse in one sentence."
    "Explain KV cache reuse in one sentence, then give an example."
    "What is tensor parallelism?"
)
for round in 1 2; do
    echo "===== round $round (disagg 2P2D) ====="
    for p in "${PROMPTS[@]}"; do echo "[req] $p"; route_one "$p"; done
    sleep 1
done

echo "[done] services still running; Ctrl-C to tear down."
wait
