#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Real-engine validation of the primary-tier pin/unpin API.

Drives a real vLLM engine (in-process InprocClient) through three roles:

1. Cache-fill client: plain llm.generate() calls that push KV blocks into
   the CPU/DRAM primary offload tier.
2. Key-discovery client (BlockKeyListener): an independent ZMQ subscriber
   that reconstructs OffloadKeys for blocks resident in the CPU tier,
   purely from KV events -- it never touches the manager object.
3. NIXL tester client (NixlPinTester): owns a second real NIXL agent and
   DRAM buffer; uses the PrimaryPinningAPI surface and does real UCX
   transfers.

See docs/superpowers/specs/2026-07-02-primary-tier-pinning-nixl-e2e-design.md
for the full design and
tests/v1/kv_offload/kvcc-tests/pinning/README.md for a detailed walkthrough.

Requires a real GPU and a real NIXL/UCX installation. Not pytest-collected.

Usage:
    /images/nixl/venv-vllm-cu/bin/python validate_pinning.py \\
        --model Qwen/Qwen3-0.6B-FP8 --cpu-offload-gb 0.1 --num-pins 5
"""

from __future__ import annotations

import argparse
import hashlib
import os
import random
import sys
import threading
import time
import uuid

# Must run before `import vllm`: controls whether KV events carry the raw
# block-hash bytes (needed for an exact OffloadKey reconstruction) or a
# lossy 64-bit int digest. See maybe_convert_block_hash in
# vllm/v1/core/kv_cache_utils.py; default is "1" (lossy) per vllm/envs.py.
os.environ["VLLM_KV_EVENTS_USE_INT_BLOCK_HASHES"] = "0"

# Must also run before `import vllm`. get_primary_tier() below requires the
# in-process InprocClient, but VLLM_ENABLE_V1_MULTIPROCESSING defaults to
# True (vllm/envs.py) -- vllm.LLM does NOT default to in-process EngineCore.
# Force it off so the scheduler/manager stay reachable in this process.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import numpy as np
import zmq
from msgspec.msgpack import Decoder, Encoder

from vllm import LLM, SamplingParams
from vllm.config.kv_events import KVEventsConfig
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed import nixl_utils
from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVEventBatch
from vllm.v1.kv_offload.base import (
    LookupResult,
    OffloadKey,
    ReqContext,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.pinning import PrimaryPinningAPI

DEFAULT_MODEL = "Qwen/Qwen3-0.6B-FP8"
_CTX = ReqContext(req_id="nixl-pin-e2e")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    # Deliberately small: GPU->CPU offload is write-through (every prefill
    # block offloads on first computation, see run()'s docstring note), so
    # this isn't about forcing GPU eviction -- it's sized so the "pressure"
    # batch in run() actually exceeds CPU-tier capacity and forces a real
    # CPU-tier eviction, exercising eviction protection under real pressure.
    parser.add_argument("--cpu-offload-gb", type=float, default=0.1)
    # Not load-bearing for getting blocks into the CPU tier (see run()) --
    # only here to keep the test's GPU footprint small and startup fast.
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.3)
    # Must be >= max_model_len / block_size (16 tokens/block for this
    # model): vLLM's KV-cache-memory check sizes required cache off the
    # model's max_model_len, not off actual prompt length, so too few
    # blocks here fails engine startup outright regardless of
    # gpu_memory_utilization. See --max-model-len below.
    parser.add_argument("--num-gpu-blocks", type=int, default=256)
    # The model's own default (40960 for Qwen3-0.6B-FP8) needs ~4.4 GiB of
    # KV cache just to pass vLLM's startup memory check -- far more than
    # this harness's intentionally tiny --num-gpu-blocks provides. Cap it
    # to a value that both passes that check and comfortably fits the
    # ~800-token fill/pressure prompts.
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--zmq-port", type=int, default=5557)
    parser.add_argument("--num-fill-prompts", type=int, default=12)
    parser.add_argument("--num-pressure-prompts", type=int, default=12)
    parser.add_argument("--num-pins", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="Run the BlockKeyListener self-test (no GPU/NIXL) and exit",
    )
    return parser.parse_args(argv)


def build_llm(args: argparse.Namespace) -> LLM:
    kv_transfer_config = KVTransferConfig(
        kv_connector="OffloadingConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "spec_name": "TieringOffloadingSpec",
            "cpu_bytes_to_use": args.cpu_offload_gb * (1 << 30),
            "enable_external_pinning": True,
        },
    )
    kv_events_config = KVEventsConfig(
        enable_kv_cache_events=True,
        publisher="zmq",
        endpoint=f"tcp://*:{args.zmq_port}",
        topic="pin-test",
    )
    return LLM(
        model=args.model,
        kv_transfer_config=kv_transfer_config,
        kv_events_config=kv_events_config,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_gpu_blocks_override=args.num_gpu_blocks,
        max_model_len=args.max_model_len,
        enable_prefix_caching=True,
        enforce_eager=True,
    )


def get_primary_tier(llm: LLM):
    """Reach the scheduler's CPUPrimaryTierOffloadingManager in-process.

    Only works with the InprocClient (no multiprocessing).
    VLLM_ENABLE_V1_MULTIPROCESSING defaults to true, so this module sets
    it to "0" before `import vllm` (top of this file) to force InprocClient.
    """
    client = llm.llm_engine.engine_core
    engine_core = getattr(client, "engine_core", None)
    if engine_core is None:
        raise RuntimeError(
            "Expected an in-process InprocClient (no engine_core.engine_core "
            "attribute found); set VLLM_ENABLE_V1_MULTIPROCESSING=0 or unset it."
        )
    connector = engine_core.scheduler.get_kv_connector()
    if connector is None:
        raise RuntimeError("No KV connector configured on the scheduler")
    return connector.connector_scheduler.manager.primary_tier


class BlockKeyListener:
    """Independent ZMQ subscriber that reconstructs OffloadKeys for blocks
    resident in the primary (CPU) offload tier, observed purely over the
    wire. Never touches the manager object -- this is the key-discovery
    client, kept separate from the cache-fill and NIXL-tester roles."""

    def __init__(self, endpoint: str, topic: str = "pin-test"):
        self._decoder = Decoder(type=KVEventBatch)
        self._ctx = zmq.Context()
        self._sub = self._ctx.socket(zmq.SUB)
        self._sub.connect(endpoint)
        self._sub.setsockopt_string(zmq.SUBSCRIBE, topic)
        self._lock = threading.Lock()
        self._keys: set[OffloadKey] = set()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=5)
        self._sub.close()
        self._ctx.term()

    def snapshot(self) -> set[OffloadKey]:
        with self._lock:
            return set(self._keys)

    def _run(self) -> None:
        while not self._stop.is_set():
            if not self._sub.poll(100):
                continue
            _, _, payload = self._sub.recv_multipart()
            batch = self._decoder.decode(payload)
            for event in batch.events:
                self._handle_event(event)

    def _handle_event(self, event) -> None:
        if isinstance(event, BlockStored):
            self._on_stored(event)
        elif isinstance(event, BlockRemoved):
            self._on_removed(event)

    def _on_stored(self, event: BlockStored) -> None:
        if event.medium != "CPU" or event.group_idx is None:
            return
        # Placeholder BlockStored events (self-describing events disabled,
        # the default) are emitted one-per-key, so block_hashes is always a
        # singleton here -- but iterate all of them defensively.
        with self._lock:
            for block_hash in event.block_hashes:
                self._keys.add(make_offload_key(bytes(block_hash), event.group_idx))

    def _on_removed(self, event: BlockRemoved) -> None:
        if event.medium != "CPU" or event.group_idx is None:
            return
        # Unlike BlockStored, placeholder BlockRemoved events batch the
        # hashes of every key evicted in the same manager call that shares
        # this group_idx (see OffloadingEventsTracker._take_removed_event's
        # `by_group` aggregation) -- block_hashes may have more than one
        # entry, each an independent evicted key. Discarding only
        # block_hashes[0] would silently ignore the rest, leaking stale
        # "resident" keys that were actually evicted.
        with self._lock:
            for block_hash in event.block_hashes:
                self._keys.discard(make_offload_key(bytes(block_hash), event.group_idx))


def _selftest_block_key_listener() -> None:
    port = 15570 + (os.getpid() % 100)
    endpoint = f"tcp://127.0.0.1:{port}"
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{port}")
    time.sleep(0.2)  # let the SUB socket join before publishing

    listener = BlockKeyListener(endpoint, topic="selftest")
    listener.start()
    time.sleep(0.2)

    encoder = Encoder()
    key_bytes = b"deadbeefdeadbeef"
    expected = make_offload_key(key_bytes, 0)

    stored = BlockStored(
        block_hashes=[key_bytes],
        parent_block_hash=None,
        token_ids=[],
        block_size=0,
        lora_id=None,
        medium="CPU",
        lora_name=None,
        group_idx=0,
    )
    batch = KVEventBatch(ts=0.0, events=[stored])
    pub.send_multipart((b"selftest", (0).to_bytes(8, "big"),
                        encoder.encode(batch)))

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and expected not in listener.snapshot():
        time.sleep(0.05)
    assert expected in listener.snapshot(), \
        "listener did not observe BlockStored"

    removed = BlockRemoved(block_hashes=[key_bytes], medium="CPU",
                           group_idx=0)
    batch2 = KVEventBatch(ts=0.0, events=[removed])
    pub.send_multipart((b"selftest", (1).to_bytes(8, "big"),
                        encoder.encode(batch2)))

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and expected in listener.snapshot():
        time.sleep(0.05)
    assert expected not in listener.snapshot(), \
        "listener did not observe BlockRemoved"

    listener.stop()
    pub.close()
    ctx.term()
    print("BlockKeyListener self-test passed")


def make_distinct_prompts(n: int, seed: int) -> list[str]:
    """Build n long, content-distinct prompts so their KV blocks hash
    differently and don't dedupe against each other or earlier calls."""
    rng = random.Random(seed)
    filler = (
        "The history of distributed systems is full of surprising lessons "
        "about consistency, latency, and failure recovery. "
    )
    prompts = []
    for i in range(n):
        tag = "-".join(str(rng.randint(0, 999_999)) for _ in range(8))
        prompts.append(f"[prompt-{i}-{tag}] " + filler * 40)
    return prompts


def fill_cache(llm: LLM, prompts: list[str]) -> None:
    sampling_params = SamplingParams(max_tokens=8, temperature=0.0)
    llm.generate(prompts, sampling_params)


def wait_for_listener_to_settle(
    listener: "BlockKeyListener", timeout: float = 10.0, quiet_period: float = 1.0
) -> set[OffloadKey]:
    """Wait until the listener's key set stops changing.

    generate() can emit a large burst of BlockStored/BlockRemoved events
    (e.g. one fill batch can churn through many more blocks than the tiny
    CPU tier can hold). The ZMQ SUB thread drains that backlog
    asynchronously, so snapshotting as soon as the set is merely non-empty
    races the eviction events still in flight and yields a stale,
    inflated set of keys that were already evicted for real. Wait for the
    set size to stop changing for `quiet_period` seconds instead.
    """
    deadline = time.monotonic() + timeout
    last_size = -1
    stable_since = time.monotonic()
    while time.monotonic() < deadline:
        size = len(listener.snapshot())
        if size != last_size:
            last_size = size
            stable_since = time.monotonic()
        elif time.monotonic() - stable_since >= quiet_period:
            break
        time.sleep(0.1)
    return listener.snapshot()


class NixlPinTester:
    """Owns a separate NIXL agent and separate DRAM; exercises
    get_transport_endpoint / search_and_pin / unpin against a live
    primary tier -- the NIXL-tester client role."""

    def __init__(self, primary_tier):
        self.primary_tier = primary_tier
        self.primary_pinning: PrimaryPinningAPI = primary_tier
        config_factory = nixl_utils.nixl_agent_config
        config = (
            config_factory(backends=["UCX"]) if config_factory is not None else None
        )
        agent_name = f"pin-tester-{uuid.uuid4().hex[:8]}"
        self.agent = nixl_utils.NixlWrapper(agent_name, config)
        self._buffers: dict[OffloadKey, np.ndarray] = {}
        self._registrations: list[object] = []

    def close(self) -> None:
        for reg in self._registrations:
            self.agent.deregister_memory(reg)
        self._registrations.clear()

    def pin_and_transfer(
        self, keys: list[OffloadKey]
    ) -> tuple[str, dict[OffloadKey, bytes]]:
        pre_evictable = self.primary_tier._num_evictable_cache_blocks
        blocks_before = {k: self.primary_tier._policy.get(k) for k in keys}
        for k in keys:
            block = blocks_before[k]
            assert block is not None, f"key {k!r} not found in primary tier"
            assert block.ref_cnt == 0, f"key {k!r} expected ref_cnt 0 before pin"

        pin_result = self.primary_pinning.search_and_pin(keys)
        assert pin_result is not None, "search_and_pin unexpectedly failed"
        pin_handle, descriptors = pin_result
        assert len(descriptors) == len(keys)

        for k in keys:
            block = self.primary_tier._policy.get(k)
            assert block.ref_cnt == 1, f"key {k!r} ref_cnt should be 1 after pin"
            assert k not in self.primary_tier._policy.evictable_blocks, (
                f"key {k!r} should not be evictable while pinned"
            )
        assert (
            self.primary_tier._num_evictable_cache_blocks
            == pre_evictable - len(keys)
        ), "evictable-block counter did not drop by exactly len(keys)"

        # get_kv_memoryview() returns a raw multi-dimensional memoryview;
        # CPython's memoryview.__getitem__ can't produce a row sub-view for
        # ndim > 1 buffers ("multi-dimensional sub-views are not
        # implemented"). Wrap it in a numpy array (zero-copy) first, which
        # has no such restriction.
        kv_view = np.asarray(self.primary_tier.get_kv_memoryview())
        reference_bytes = {
            k: bytes(kv_view[blocks_before[k].block_id]) for k in keys
        }

        primary_agent = self.primary_pinning.get_transport_endpoint().end_point
        self.agent.add_remote_agent(primary_agent.get_agent_metadata())

        for key, descriptor in descriptors.items():
            self._transfer_one(key, descriptor, primary_agent)

        return pin_handle, reference_bytes

    def _transfer_one(self, key: OffloadKey, descriptor, primary_agent) -> None:
        buf = np.zeros(descriptor.size, dtype=np.uint8)
        reg = self.agent.register_memory(
            [(buf.ctypes.data, buf.nbytes, 0, "")], mem_type="DRAM"
        )
        self._registrations.append(reg)
        self._buffers[key] = buf

        local_descs = self.agent.get_xfer_descs(
            [(buf.ctypes.data, descriptor.size, 0)], mem_type="DRAM"
        )
        remote_descs = self.agent.get_xfer_descs(
            [(descriptor.addr, descriptor.size, descriptor.device_Id)],
            mem_type=descriptor.mem_type,
        )
        handle = self.agent.initialize_xfer(
            "READ", local_descs, remote_descs, primary_agent.name
        )
        try:
            state = self.agent.transfer(handle)
            while state == "PROC":
                state = self.agent.check_xfer_state(handle)
            assert state == "DONE", f"NIXL transfer for key {key!r} incomplete"
        finally:
            self.agent.release_xfer_handle(handle)

    def assert_transfer_correct(
        self, reference_bytes: dict[OffloadKey, bytes]
    ) -> None:
        for key, reference in reference_bytes.items():
            pulled = bytes(self._buffers[key])
            ref_digest = hashlib.sha256(reference).hexdigest()
            pulled_digest = hashlib.sha256(pulled).hexdigest()
            assert ref_digest == pulled_digest, (
                f"checksum mismatch for key {key!r}: "
                f"{ref_digest} != {pulled_digest}"
            )
            assert pulled == reference, f"byte mismatch for key {key!r}"

    def assert_still_pinned(self, keys: list[OffloadKey]) -> None:
        for k in keys:
            assert self.primary_tier.lookup(k, _CTX) is LookupResult.HIT, (
                f"pinned key {k!r} did not survive competing eviction pressure"
            )

    def unpin_and_assert(self, pin_handle: str, keys: list[OffloadKey]) -> None:
        pre_evictable = self.primary_tier._num_evictable_cache_blocks
        assert self.primary_pinning.unpin(pin_handle) is True

        for k in keys:
            block = self.primary_tier._policy.get(k)
            assert block.ref_cnt == 0, f"key {k!r} ref_cnt should be 0 after unpin"
            assert k in self.primary_tier._policy.evictable_blocks, (
                f"key {k!r} should be evictable again after unpin"
            )
        assert (
            self.primary_tier._num_evictable_cache_blocks
            == pre_evictable + len(keys)
        ), "evictable-block counter did not rise by exactly len(keys)"

        assert self.primary_pinning.unpin(pin_handle) is False, (
            "second unpin of the same handle should return False"
        )


def run(args: argparse.Namespace) -> None:
    """
    GPU->CPU offload is write-through, not GPU-pressure-driven:
    
    1. OffloadingConnectorScheduler.build_connector_meta() calls
    _build_store_jobs() every scheduler step, which stores every
    newly-computed prefill block to the CPU tier immediately
    (vllm/distributed/kv_transfer/kv_connector/v1/offloading/scheduler.py:844).
    
    2. TieringOffloadingSpec also forces store_threshold=1 (spec.py:178), so
    there's no repeated-access gate either. 
    
    So getting blocks into the CPU tier only needs cpu_bytes_to_use > 0 and 
    prompts long enough to span at least one full block -- GPU cache size 
    plays no role in that. 
    
    The later "pressure" batch is a different mechanism: it's sized (via
    --cpu-offload-gb) to fill the CPU tier itself, forcing a real
    CPU-tier eviction that the pinned blocks must survive.
    """
    if not nixl_utils.is_nixl_available():
        raise RuntimeError("NIXL is not installed; this script requires real NIXL")

    listener = BlockKeyListener(f"tcp://127.0.0.1:{args.zmq_port}", topic="pin-test")
    listener.start()
    llm: LLM | None = None
    tester: NixlPinTester | None = None
    try:
        llm = build_llm(args)
        primary_tier = get_primary_tier(llm)

        print(
            "Note: --cpu-offload-gb is deliberately small, so you may see "
            "vLLM scheduler warnings like 'Request ...: cannot store "
            "blocks' below. That means the CPU tier is genuinely full and "
            "couldn't evict enough room for a block in that scheduling "
            "step -- expected here, not a failure. It gets more likely "
            "once blocks are pinned (pinned blocks are never evictable) "
            "and during the pressure batch further below."
        )
        fill_prompts = make_distinct_prompts(args.num_fill_prompts, seed=args.seed)
        fill_cache(llm, fill_prompts)

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and not listener.snapshot():
            time.sleep(0.2)
        resident_keys = list(wait_for_listener_to_settle(listener))
        if not resident_keys:
            raise RuntimeError(
                "No blocks observed in the CPU tier. GPU->CPU offload is "
                "write-through (not GPU-pressure-driven), so check: "
                "--cpu-offload-gb is > 0, --num-fill-prompts is at least 1, "
                "and each prompt is long enough to span a full KV block "
                "(the default make_distinct_prompts() filler already is)."
            )
        print(f"Discovered {len(resident_keys)} resident CPU-tier block keys")

        num_pins = min(args.num_pins, len(resident_keys))
        rng = random.Random(args.seed)
        chosen_keys = rng.sample(resident_keys, num_pins)
        print(f"Pinning {num_pins} keys: {[k.hex() for k in chosen_keys]}")

        tester = NixlPinTester(primary_tier)
        pin_handle, reference_bytes = tester.pin_and_transfer(chosen_keys)
        print("search_and_pin + get_transport_endpoint + real NIXL transfer OK")

        print(
            f"Applying pressure: {num_pins} blocks are now pinned and "
            "non-evictable, so 'cannot store blocks' warnings below are "
            "expected while this batch forces the CPU tier to evict "
            "everything else it can."
        )
        pressure_prompts = make_distinct_prompts(
            args.num_pressure_prompts, seed=args.seed + 1
        )
        fill_cache(llm, pressure_prompts)
        tester.assert_still_pinned(chosen_keys)
        print("Pinned blocks survived real competing eviction pressure")

        tester.assert_transfer_correct(reference_bytes)
        print("Checksum + byte-equality verification passed")

        tester.unpin_and_assert(pin_handle, chosen_keys)
        print("unpin correctly restored evictability and rejected double-unpin")

        print("ALL CHECKS PASSED")
    finally:
        if tester is not None:
            tester.close()
        listener.stop()
        if llm is not None:
            engine_core = getattr(llm.llm_engine.engine_core, "engine_core", None)
            if engine_core is not None:
                try:
                    engine_core.shutdown()
                except Exception as exc:  # best-effort cleanup
                    print(f"warning: engine shutdown failed: {exc}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.selftest:
        _selftest_block_key_listener()
        return 0
    try:
        run(args)
    except AssertionError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
