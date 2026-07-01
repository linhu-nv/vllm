# Promptfoo Golden Tests

This README shows how to use Promptfoo to run simple golden-output checks against a vLLM OpenAI-compatible server.

## Install

Promptfoo requires Node.js. The easiest way to run it without a global install is:

```bash
# install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.5/install.sh | bash

# install node
nvm install node

# install promptfoo
npx promptfoo@latest --version
```

Or (less preferred) install it globally:

```bash
npm install -g promptfoo
promptfoo --version
```

## Initialize

Create a Promptfoo config in your test directory:

```bash
npx promptfoo@latest init
```

This creates a `promptfooconfig.yaml`. Edit it to point to your vLLM server.

Example provider for local vLLM:

```yaml
providers:
  - id: openai:chat:local-vllm
    config:
      apiBaseUrl: http://127.0.0.1:8000/v1
      apiKey: EMPTY
      temperature: 0
      max_tokens: 64
```

## Example Test Config

```yaml
prompts:
  - "{{prompt}}"

providers:
  - id: openai:chat:local-vllm
    config:
      apiBaseUrl: http://127.0.0.1:8000/v1
      apiKey: EMPTY
      temperature: 0
      max_tokens: 64

tests:
  - description: exact echo
    vars:
      prompt: "Return exactly this string and nothing else: KV_CACHE_OK_17"
    assert:
      - type: equals
        value: "KV_CACHE_OK_17"

  - description: arithmetic
    vars:
      prompt: "Answer with only the final integer. What is 137 + 268 - 45?"
    assert:
      - type: equals
        value: "360"
```

## Run Tests

Start vLLM first:

```bash
vllm serve /path/to/model --host 0.0.0.0 --port 8000
```

Then run Promptfoo:

```bash
npx promptfoo@latest eval -c promptfooconfig.yaml --no-cache
```

Open the local report UI:

```bash
npx promptfoo@latest view
```

## Result Explanation

Promptfoo prints a table with each test case and assertion result.

- `PASS`: the model output matched the assertion.
- `FAIL`: the output did not match; inspect the actual output and diff.
- `ERROR`: the request failed, timed out, or the server returned an error.

For KV-cache correctness checks, exact-match failures are useful signals. If a known-good baseline passes but the modified vLLM build fails on the same deterministic prompts, inspect the KV-cache change first.

Use deterministic decoding for golden tests:

```yaml
temperature: 0
max_tokens: 64
```

If exact string comparison is too strict for a model, use softer assertions:

```yaml
assert:
  - type: contains
    value: "KV_CACHE_OK"
```

or regex:

```yaml
assert:
  - type: regex
    value: "^KV_CACHE_OK_17\\s*$"
```

