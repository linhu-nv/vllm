# LLM Golden Validator

Generate and validate golden-output tests for an OpenAI-compatible LLM
server (e.g. vLLM). Useful for catching output regressions when changing
server config, code, or hardware while keeping the model and prompts fixed.

## How it works

1. **`generate`** — sends a fixed suite of prompts to a baseline server and
   records each prompt, its output, and the exact sampling parameters used
   (`temperature`, `top_p`, `seed`, `max_tokens`, `endpoint`, `model`) into a
   golden JSONL file.
2. **`validate`** — reads that same golden file, replays each prompt against
   a (possibly different/modified) server using the *same* stored sampling
   parameters, and compares the new output against the recorded golden
   output.

Because `temperature`, `top_p`, `seed`, and `max_tokens` are pinned and
reused verbatim between `generate` and `validate`, any diff reported by
`validate` reflects a real behavioral change in the server/model, not
sampling noise.

The prompt suite exercises a mix of short exact-match tasks, JSON
formatting, long-context lookups, and shared-prefix prompts — useful for
probing prefill/decode correctness, KV cache handling, and batching effects.

Qwen3-style thinking traces are disabled on every request via
`chat_template_kwargs: {"enable_thinking": false}`, so goldens capture the
final answer rather than a truncated `<think>...</think>` fragment.

## Requirements

- Python 3.9+
- An OpenAI-compatible server (e.g. `vllm serve ...`) reachable at
  `--base-url`. No third-party packages are required — the script only uses
  the standard library.

## Usage

### 1. Generate goldens against a known-good server

```bash
python llm_golden_validator.py generate \
  --base-url http://127.0.0.1:8000/v1 \
  --model /images/models/Qwen/Qwen3-0.6B-FP8 \
  --golden golden-test-prompts.jsonl
```

### 2. Validate a new/modified server against those goldens

```bash
python llm_golden_validator.py validate \
  --base-url http://127.0.0.1:8000/v1 \
  --golden golden-test-prompts.jsonl
```

`validate` does not require `--model`; it falls back to the `model`
recorded in each golden row (and likewise for `endpoint`, `temperature`,
`top_p`, and `seed`) unless you explicitly override them on the CLI.

Exit code is `0` if all cases pass, `1` if any case fails.

## CLI reference

### Common flags (both subcommands)

| Flag | Default | Description |
| --- | --- | --- |
| `--base-url` | `http://127.0.0.1:8000/v1` | OpenAI-compatible API base URL |
| `--api-key` | `EMPTY` | Bearer token sent as `Authorization` header |
| `--model` | *(required for `generate`)* | Model name/path passed to the server |
| `--golden` | *(required)* | Path to the golden JSONL file |
| `--endpoint` | `chat` for `generate`, golden's value for `validate` | `chat` or `completions` |
| `--timeout` | `120.0` | Per-request timeout in seconds |
| `--limit` | *(all)* | Only run the first N test cases |

### `generate`-only flags

| Flag | Default | Description |
| --- | --- | --- |
| `--temperature` | `0.0` | Sampling temperature (0.0 = greedy/deterministic) |
| `--top-p` | `1.0` | Nucleus sampling top-p |
| `--seed` | `1234` | Sampling seed |

### `validate`-only flags

| Flag | Default | Description |
| --- | --- | --- |
| `--temperature` | golden's stored value | Override the temperature used for replay |
| `--top-p` | golden's stored value | Override the top-p used for replay |
| `--seed` | golden's stored value | Override the seed used for replay |
| `--normalize` | `exact` | `exact`, `strip`, or `space` — how to normalize text before comparing |
| `--fail-fast` | off | Stop at the first failing case |
| `--default-max-tokens` | `64` | Used only if a golden row has no `max_tokens` |

## Golden file format

Each line of the golden file is a JSON object:

```json
{
  "case_id": "exact_echo_short",
  "prompt": "Return exactly this string and nothing else: KV_CACHE_OK_17",
  "expected": "KV_CACHE_OK_17",
  "max_tokens": 16,
  "stop": null,
  "endpoint": "chat",
  "temperature": 0.0,
  "top_p": 1.0,
  "seed": 1234,
  "model": "/images/models/Qwen/Qwen3-0.6B-FP8",
  "elapsed_ms": 64.63,
  "usage": {"completion_tokens": 7, "prompt_tokens": 26, "total_tokens": 33}
}
```

The file is both the artifact `generate` produces and the input `validate`
consumes — no separate prompt list is needed. It's plain JSONL, so it's easy
to diff, edit, or check into version control.

## Example output

```
[01/12] PASS exact_echo_short
[02/12] PASS arithmetic_short
...
[12/12] PASS negative_control_nonempty

Passed 12/12 cases
```

On failure, `validate` prints a unified diff between the expected and actual
output for each failing case, then lists all failed case IDs at the end.
