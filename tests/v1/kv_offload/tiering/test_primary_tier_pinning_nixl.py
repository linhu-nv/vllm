# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Real-NIXL validation of the primary-tier pin/export API.

This exercises the "First Validation" scenario from the design note
vllm_primary_tier_pinning_api.md (kept outside this git repo, alongside the
checkout it applies to): a tester that owns separate DRAM and a separate
NIXL agent pins a ready primary-tier block via search_and_pin(),
reads the bytes described by the returned MemDescriptor into tester-owned
memory over a real NIXL transfer, verifies them, and unpins. It also checks
eviction protection: a pinned block must not be evictable, and must become
evictable again after unpin.

Requires a real nixl/rixl installation (skipped otherwise) since the point
is to prove actual DRAM<->DRAM data movement, not mocked calls.
"""

import numpy as np
import pytest

from vllm.distributed import nixl_utils
from vllm.v1.kv_offload.tiering.manager import CPUPrimaryTierOffloadingManager

from .test_tiering_offloading import _mock_mmap_region, store_ready_blocks, to_keys

pytestmark = pytest.mark.skipif(
    not nixl_utils.is_nixl_available(), reason="NIXL is not installed"
)


def _make_tester_agent(name: str):
    config_factory = nixl_utils.nixl_agent_config
    config = config_factory(backends=["UCX"]) if config_factory is not None else None
    return nixl_utils.NixlWrapper(name, config)


class TestPrimaryTierPinningRealNixl:
    def test_search_and_pin_real_dram_to_dram_transfer(self):
        row_bytes = 4096
        mock_region = _mock_mmap_region(2, row_bytes=row_bytes)
        primary_tier = CPUPrimaryTierOffloadingManager(
            num_blocks=2,
            mmap_region=mock_region,
            enable_external_pinning=True,
        )
        tester_agent = None
        tester_reg = None
        try:
            key = to_keys([0])[0]
            store_ready_blocks(primary_tier, [key])
            block = primary_tier._policy.get(key)
            assert block is not None

            # Simulate GPU->CPU offload having already written data into the
            # primary-tier block, via the primary tier's own memoryview.
            pattern = bytes((i % 256 for i in range(row_bytes)))
            view = primary_tier.get_kv_memoryview()
            view.obj[block.block_id, :] = np.frombuffer(pattern, dtype=np.int8)

            pin_result = primary_tier.search_and_pin([key])
            assert pin_result is not None
            pin_handle, descriptors = pin_result
            descriptor = descriptors[key]

            assert key not in primary_tier._policy.evictable_blocks
            assert primary_tier._num_evictable_cache_blocks == 0

            primary_agent = primary_tier.get_transport_endpoint().end_point

            # Tester owns separate DRAM and a separate NIXL agent.
            tester_buf = np.zeros(descriptor.size, dtype=np.uint8)
            tester_agent = _make_tester_agent("pin-tester")
            tester_reg = tester_agent.register_memory(
                [(tester_buf.ctypes.data, tester_buf.nbytes, 0, "")],
                mem_type="DRAM",
            )
            tester_agent.add_remote_agent(primary_agent.get_agent_metadata())

            local_descs = tester_agent.get_xfer_descs(
                [(tester_buf.ctypes.data, descriptor.size, 0)], mem_type="DRAM"
            )
            remote_descs = tester_agent.get_xfer_descs(
                [(descriptor.addr, descriptor.size, descriptor.device_Id)],
                mem_type=descriptor.mem_type,
            )
            handle = tester_agent.initialize_xfer(
                "READ", local_descs, remote_descs, primary_agent.name
            )
            try:
                state = tester_agent.transfer(handle)
                while state == "PROC":
                    state = tester_agent.check_xfer_state(handle)
                assert state == "DONE"
            finally:
                tester_agent.release_xfer_handle(handle)

            assert bytes(tester_buf) == pattern

            assert primary_tier.unpin(pin_handle) is True
            assert key in primary_tier._policy.evictable_blocks
            assert primary_tier._num_evictable_cache_blocks == 1
            assert primary_tier.unpin(pin_handle) is False
        finally:
            if tester_agent is not None and tester_reg is not None:
                tester_agent.deregister_memory(tester_reg)
            primary_tier.shutdown()
