#!/usr/bin/env python3
"""
Generate and validate golden-output tests for an OpenAI-compatible LLM server.

Typical use:

  # 1. Start a known-good vLLM server, then generate goldens.
  python llm_golden_validator.py generate \
    --base-url http://127.0.0.1:8000/v1 \
    --model /images/models/Qwen/Qwen3-0.6B-FP8 \
    --golden qwen3_0_6b_goldens.jsonl

  # 2. Start the modified vLLM server, then validate against the goldens.
  python llm_golden_validator.py validate \
    --base-url http://127.0.0.1:8000/v1 \
    --model /images/models/Qwen/Qwen3-0.6B-FP8 \
    --golden qwen3_0_6b_goldens.jsonl
"""

from __future__ import annotations

import argparse
import difflib
import json
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_API_KEY = "EMPTY"


@dataclass(frozen=True)
class TestCase:
    case_id: str
    prompt: str
    max_tokens: int
    stop: list[str] | None = None


def build_prompts() -> list[TestCase]:
    """Prompts chosen to exercise prefill, decode, long-context KV, and batching.

    Each prompt is plain, human-readable text (no synthetic token filler),
    so a person reading the golden file can tell at a glance what's being
    asked and why the expected answer looks the way it does.
    """
    employee_lockers = "\n".join(
        f"Employee number {i} was assigned locker number {(i * 17) % 997}."
        for i in range(180)
    )
    neighborhood_owners = "\n".join(
        f"House number {i} on Maple Street is owned by a person named Owner{i % 13}."
        for i in range(240)
    )
    filler_story = (
        "On a quiet Tuesday afternoon, the town's bakery smelled like fresh bread "
        "and cinnamon, and the regulars lingered over their coffee a little longer "
        "than usual. "
    ) * 60

    return [
        TestCase(
            "exact_echo_short",
            "Please repeat this exact phrase back to me and nothing else: KV_CACHE_OK_17",
            16,
        ),
        TestCase(
            "arithmetic_short",
            "What is 137 plus 268 minus 45? Reply with only the final number.",
            16,
        ),
        TestCase(
            "json_shape",
            (
                "Give me a compact JSON object with no markdown formatting. It should "
                "have three keys: status set to 'ok', count set to 7, and tag set to 'kv'."
            ),
            64,
        ),
        TestCase(
            "copy_sequence",
            (
                "Please repeat this list of items back to me exactly, with nothing added:\n"
                "red-01, blue-02, green-03, amber-04, violet-05"
            ),
            64,
        ),
        TestCase(
            "multi_step_instruction",
            (
                "Take the word 'cache', make it uppercase, and add '-PASS' to the end. "
                "Reply with only the final result, no explanation."
            ),
            32,
        ),
        TestCase(
            "long_context_lookup_a",
            (
                "Here is a list of employees and the locker numbers they were assigned.\n\n"
                f"{employee_lockers}\n\n"
                "Question: What locker number was employee number 137 assigned? "
                "Answer with only the number."
            ),
            32,
        ),
        TestCase(
            "long_context_lookup_b",
            (
                "Here is a list of houses on Maple Street and who owns each one.\n\n"
                f"{neighborhood_owners}\n\n"
                "Question: Who owns house number 211? Answer with only the owner's name."
            ),
            32,
        ),
        TestCase(
            "long_repetition_tail",
            (
                "Here is a short story. Please ignore all of it and only follow the "
                "instruction at the very end.\n\n"
                f"{filler_story}\n\n"
                "Instruction: reply with exactly TAIL_OK and nothing else."
            ),
            32,
        ),
        TestCase(
            "prefix_cache_candidate_1",
            (
                "Here is some shared context: Alice, Bob, and Carol are meeting for "
                "coffee tomorrow morning to plan their weekend trip. "
                "Based on that context, reply with only the word: FIRST"
            ),
            16,
        ),
        TestCase(
            "prefix_cache_candidate_2",
            (
                "Here is some shared context: Alice, Bob, and Carol are meeting for "
                "coffee tomorrow morning to plan their weekend trip. "
                "Based on that context, reply with only the word: SECOND"
            ),
            16,
        ),
        TestCase(
            "code_like",
            (
                "Write a single Python list comprehension, with no explanation, that "
                "builds the list [1, 4, 9] by squaring each number from 1 to 3."
            ),
            64,
        ),
        TestCase(
            "negative_control_nonempty",
            "In one short sentence, tell me that you understood my question.",
            48,
        ),
    ]


def normalize_text(text: str, mode: str) -> str:
    if mode == "exact":
        return text
    if mode == "strip":
        return text.strip()
    if mode == "space":
        return " ".join(text.split())
    raise ValueError(f"Unknown normalization mode: {mode}")


def post_json(url: str, api_key: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc


def generate_one(
    *,
    base_url: str,
    api_key: str,
    model: str,
    endpoint: str,
    test: TestCase,
    temperature: float,
    top_p: float,
    seed: int | None,
    timeout: float,
) -> tuple[str, dict[str, Any]]:
    base_url = base_url.rstrip("/")
    if endpoint == "chat":
        url = f"{base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": test.prompt}],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": test.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if test.stop:
            payload["stop"] = test.stop
        if seed is not None:
            payload["seed"] = seed
        response = post_json(url, api_key, payload, timeout)
        output = response["choices"][0]["message"].get("content") or ""
        return output, response

    if endpoint == "completions":
        url = f"{base_url}/completions"
        payload = {
            "model": model,
            "prompt": test.prompt,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": test.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if test.stop:
            payload["stop"] = test.stop
        if seed is not None:
            payload["seed"] = seed
        response = post_json(url, api_key, payload, timeout)
        output = response["choices"][0].get("text") or ""
        return output, response

    raise ValueError(f"Unknown endpoint: {endpoint}")


def load_goldens(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_goldens(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def command_generate(args: argparse.Namespace) -> int:
    tests = build_prompts()
    if args.limit is not None:
        tests = tests[: args.limit]

    rows: list[dict[str, Any]] = []
    for index, test in enumerate(tests, 1):
        started = time.time()
        output, response = generate_one(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            endpoint=args.endpoint,
            test=test,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.seed,
            timeout=args.timeout,
        )
        elapsed_ms = round((time.time() - started) * 1000, 2)
        usage = response.get("usage", {})
        rows.append(
            {
                "case_id": test.case_id,
                "prompt": test.prompt,
                "expected": output,
                "max_tokens": test.max_tokens,
                "stop": test.stop,
                "endpoint": args.endpoint,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "seed": args.seed,
                "model": args.model,
                "elapsed_ms": elapsed_ms,
                "usage": usage,
            }
        )
        preview = output.replace("\n", "\\n")[:100]
        print(f"[{index:02d}/{len(tests):02d}] wrote {test.case_id}: {preview!r}")

    write_goldens(Path(args.golden), rows)
    print(f"\nWrote {len(rows)} golden cases to {args.golden}")
    return 0


def diff_text(expected: str, actual: str) -> str:
    diff = difflib.unified_diff(
        expected.splitlines(),
        actual.splitlines(),
        fromfile="expected",
        tofile="actual",
        lineterm="",
    )
    return "\n".join(diff)


def command_validate(args: argparse.Namespace) -> int:
    rows = load_goldens(Path(args.golden))
    if args.limit is not None:
        rows = rows[: args.limit]

    failures: list[str] = []
    for index, row in enumerate(rows, 1):
        test = TestCase(
            case_id=row["case_id"],
            prompt=row["prompt"],
            max_tokens=int(row.get("max_tokens", args.default_max_tokens)),
            stop=row.get("stop"),
        )
        output, _ = generate_one(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model or row.get("model", ""),
            endpoint=args.endpoint or row.get("endpoint", "chat"),
            test=test,
            temperature=args.temperature
            if args.temperature is not None
            else float(row.get("temperature", 0.0)),
            top_p=args.top_p if args.top_p is not None else float(row.get("top_p", 1.0)),
            seed=args.seed if args.seed is not None else row.get("seed"),
            timeout=args.timeout,
        )
        expected = row["expected"]
        expected_cmp = normalize_text(expected, args.normalize)
        output_cmp = normalize_text(output, args.normalize)
        if expected_cmp == output_cmp:
            print(f"[{index:02d}/{len(rows):02d}] PASS {test.case_id}")
            continue

        print(f"[{index:02d}/{len(rows):02d}] FAIL {test.case_id}")
        print(diff_text(expected_cmp, output_cmp) or "(single-line output differs)")
        failures.append(test.case_id)
        if args.fail_fast:
            break

    print(f"\nPassed {len(rows) - len(failures)}/{len(rows)} cases")
    if failures:
        print("Failed cases: " + ", ".join(failures))
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate and validate golden LLM outputs through a vLLM/OpenAI-compatible API."
    )
    subparsers = parser.add_subparsers(required=True)

    def add_common(subparser: argparse.ArgumentParser, model_required: bool = False, endpoint_default=None) -> None:
        subparser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
        subparser.add_argument("--api-key", default=DEFAULT_API_KEY)
        subparser.add_argument("--model", required=model_required)
        subparser.add_argument("--golden", required=True)
        subparser.add_argument("--endpoint", choices=["chat", "completions"], default=endpoint_default)
        subparser.add_argument("--timeout", type=float, default=120.0)
        subparser.add_argument("--limit", type=int)

    generate = subparsers.add_parser("generate", help="Query a baseline server and write goldens.")
    add_common(generate, model_required=True, endpoint_default="chat")
    generate.add_argument("--temperature", type=float, default=0.0)
    generate.add_argument("--top-p", type=float, default=1.0)
    generate.add_argument("--seed", type=int, default=1234)
    generate.set_defaults(func=command_generate)

    validate = subparsers.add_parser("validate", help="Query a test server and compare with goldens.")
    add_common(validate)
    validate.add_argument("--temperature", type=float)
    validate.add_argument("--top-p", type=float)
    validate.add_argument("--seed", type=int)
    validate.add_argument("--normalize", choices=["exact", "strip", "space"], default="exact")
    validate.add_argument("--fail-fast", action="store_true")
    validate.add_argument("--default-max-tokens", type=int, default=64)
    validate.set_defaults(func=command_validate)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

