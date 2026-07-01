# Tests

## manual run

**run vllm directly**

```shell
# install uv if it is not installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# create venv
uv venv venv --python 3.12
source venv/bin/activate
uv pip install pip
uv pip install pandas # needed for multiturn test

# clone & install vllm
git clone https://github.com/vllm-project/vllm.git
cd vllm
VLLM_USE_PRECOMPILED=1 uv pip install --editable . --torch-backend=auto
```

**run vllm container**
```shell
docker run --gpus '"device=0,1"' \
   -v /images/models:/images/models \
   -p 8000:8000 \
   --ipc=host \
   vllm/vllm-openai:latest \
   	--model /images/models/Qwen/Qwen3-0.6B-FP8 \
	--host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 2 \
    --block-size 64 \
    --max-model-len 32000 \
    --trust-remote-code \
    --enable-prefix-caching \
    --disable-custom-all-reduce \
    --disable-hybrid-kv-cache-manager \
    --kv-offloading-size 100.0 \
    --kv-offloading-backend native
```

**Quick test**
```shell
curl localhost:8000/v1/chat/completions \
-H 'Content-Type: application/json' \
-d '{
  "model": "/images/models/Qwen/Qwen3-0.6B-FP8",
  "messages": [{"role":"user","content":"Hello"}]
}'
```

**multi-turn test**
```shell
# inside vLLM repo
cd benchmarks/multi_turn

wget https://www.gutenberg.org/ebooks/1184.txt.utf-8
mv 1184.txt.utf-8 pg1184.txt

export MODEL_PATH=/images/models/Qwen/Qwen3-0.6B-FP8

python benchmark_serving_multi_turn.py --model $MODEL_PATH \
--input-file generate_multi_turn.json --num-clients 2 \
--max-active-conversations 6
```

## agg vs disagg

`launch-workers` has scripts to launcher vLLM workers in agg and disagg modes.

**Agg mode: two workers + router**

```shell
./agg.sh --model /images/models/Qwen/Qwen3-0.6B-FP8 --block-size 64
```

**DisAgg mode: 2P2D + router**

```shell
./disagg.sh --model /images/models/Qwen/Qwen3-0.6B-FP8 --block-size 64
```

## Correctness Test

see `golden-validator` and `promptfoo` in `correctness/` folder.