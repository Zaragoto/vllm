"""Tests for SWA post-processing utilities.

These tests verify that the vectorized implementation of
``post_process_vectorized`` produces **identical** results to the
loop-based reference ``post_process_loop`` across a range of scenarios.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pytest

from vllm.v1.worker.swa_utils import SWAPostProcessor

BLOCK_SIZE = SWAPostProcessor.BLOCK_SIZE  # 128


# ---------------------------------------------------------------------------
# Lightweight stubs to replace real torch / NPU objects
# ---------------------------------------------------------------------------

class FakeTensor:
    """Minimal ndarray wrapper that supports .clone() and item assignment."""

    def __init__(self, data: np.ndarray):
        self._data = data.copy()

    def clone(self) -> "FakeTensor":
        return FakeTensor(self._data)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, FakeTensor):
            return np.array_equal(self._data, other._data)
        return NotImplemented

    def __repr__(self) -> str:
        return f"FakeTensor({self._data})"


class FakePool(dict):
    """dict subclass that stores FakeTensor values."""
    pass


@dataclass
class FakeKVCacheGroup:
    layer_names: List[str] = field(default_factory=lambda: ["layer0"])


@dataclass
class FakeKVCacheConfig:
    kv_cache_groups: Dict[int, FakeKVCacheGroup] = field(default_factory=dict)


@dataclass
class FakeCommonAttnMetadata:
    query_start_loc_cpu: np.ndarray = field(default_factory=lambda: np.array([0]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_hbm_pool(grp_id: int, num_slots: int = 20) -> Dict[int, Dict[str, FakePool]]:
    """Create a fake HBM buffer pool with *num_slots* FakeTensors per layer."""
    pool: Dict[str, FakeTensor] = {}
    for i in range(num_slots):
        pool[i] = FakeTensor(np.array([i * 10 + grp_id]))
    return {grp_id: {"layer0": pool}}


def _make_block_table_pool(grp_id: int, req_offset: int = 4) -> dict:
    return {grp_id: {"req_offset": req_offset}}


def _deep_copy_state(proc: SWAPostProcessor):
    """Snapshot mutable dictionaries so we can reset between runs."""
    return {
        "last_num_non_zeros": copy.deepcopy(proc.last_num_non_zeros),
        "record_req_sched_times": copy.deepcopy(proc.record_req_sched_times),
        "hbm_buffer_pool": copy.deepcopy(proc.hbm_buffer_pool),
    }


def _restore_state(proc: SWAPostProcessor, snap: dict):
    proc.last_num_non_zeros = copy.deepcopy(snap["last_num_non_zeros"])
    proc.record_req_sched_times = copy.deepcopy(snap["record_req_sched_times"])
    proc.hbm_buffer_pool = copy.deepcopy(snap["hbm_buffer_pool"])


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestSWAPostProcessor:
    """Ensure vectorized == loop for various request layouts."""

    @staticmethod
    def _build_processor(
        num_reqs: int,
        block_table: np.ndarray,
        slot_mapping: np.ndarray,
        query_start_loc: np.ndarray,
        swa_blocks_per_req: Dict[str, Dict[int, List[int]]],
        req_id_to_idx: Dict[str, int] | None = None,
        grp_id: int = 0,
        req_offset: int = 4,
    ) -> SWAPostProcessor:
        req_ids = [f"req_{i}" for i in range(num_reqs)]
        if req_id_to_idx is None:
            req_id_to_idx = {rid: i for i, rid in enumerate(req_ids)}

        hbm_pool = _make_hbm_pool(grp_id, num_slots=40)
        bt_pool = _make_block_table_pool(grp_id, req_offset=req_offset)
        kv_cfg = FakeKVCacheConfig(
            kv_cache_groups={grp_id: FakeKVCacheGroup(layer_names=["layer0"])}
        )

        return SWAPostProcessor(
            req_id_to_idx=req_id_to_idx,
            req_ids_update_buffer=req_ids,
            record_req_ids_swa_block=swa_blocks_per_req,
            hbm_buffer_block_table_pool=bt_pool,
            hbm_buffer_pool=hbm_pool,
            kv_cache_config=kv_cfg,
            metadata_grp_id=grp_id,
        )

    def _run_both_and_compare(
        self,
        proc: SWAPostProcessor,
        num_reqs: int,
        slot_mapping: np.ndarray,
        block_table: np.ndarray,
        query_start_loc: np.ndarray,
        draft_index: int = -1,
    ):
        """Run loop and vectorized, assert identical outputs."""
        meta = FakeCommonAttnMetadata(query_start_loc_cpu=query_start_loc)

        # Snapshot inputs
        bt_loop = block_table.copy()
        sm_loop = slot_mapping.copy()
        bt_vec = block_table.copy()
        sm_vec = slot_mapping.copy()

        # Snapshot processor state
        snap = _deep_copy_state(proc)

        # Run loop
        bt_loop_out, sm_loop_out = proc.post_process_loop(
            num_reqs, sm_loop, bt_loop, meta, draft_index
        )

        # Restore state for vectorized run
        _restore_state(proc, snap)

        # Run vectorized
        bt_vec_out, sm_vec_out = proc.post_process_vectorized(
            num_reqs, sm_vec, bt_vec, meta, draft_index
        )

        np.testing.assert_array_equal(
            bt_loop_out,
            bt_vec_out,
            err_msg="block_table mismatch",
        )
        np.testing.assert_array_equal(
            sm_loop_out,
            sm_vec_out,
            err_msg="slot_mapping mismatch",
        )

    # ---- Scenario 1: single request, count == 1 ----
    def test_single_req_count1(self):
        """One request with exactly 1 non-zero block."""
        num_reqs = 1
        # block_table: 1 row, multiple columns
        # Place one non-zero block at column 2 with value 10
        block_table = np.zeros((4, 16), dtype=np.int64)
        block_table[0, 2] = 10  # block id 10

        # slot_mapping: first token maps to block 10 (slot = 10*128 + 5 = 1285)
        # last token also maps to the same block (count will be 1)
        slot_mapping = np.array([1285, 1290], dtype=np.int64)
        query_start_loc = np.array([0, 2], dtype=np.int64)

        swa_blocks = {
            "req_0": {0: [10]},  # swa_blocks[0] matches block_table_old value
        }

        proc = self._build_processor(
            num_reqs, block_table, slot_mapping, query_start_loc,
            swa_blocks, grp_id=0, req_offset=4,
        )
        self._run_both_and_compare(
            proc, num_reqs, slot_mapping, block_table, query_start_loc
        )

    # ---- Scenario 2: single request, count == 2 ----
    def test_single_req_count2(self):
        """One request with 2 non-zero blocks."""
        num_reqs = 1
        block_table = np.zeros((4, 16), dtype=np.int64)
        block_table[0, 3] = 10
        block_table[0, 4] = 11

        # first token → block 10: slot = 10*128 + 0 = 1280
        # last  token → block 11: slot = 11*128 + 5 = 1413
        slot_mapping = np.array([1280, 1350, 1413], dtype=np.int64)
        query_start_loc = np.array([0, 3], dtype=np.int64)

        swa_blocks = {"req_0": {0: [10, 11]}}

        proc = self._build_processor(
            num_reqs, block_table, slot_mapping, query_start_loc,
            swa_blocks, grp_id=0, req_offset=4,
        )
        self._run_both_and_compare(
            proc, num_reqs, slot_mapping, block_table, query_start_loc
        )

    # ---- Scenario 3: single request, count >= 3 ----
    def test_single_req_count3(self):
        """One request with 3 non-zero blocks."""
        num_reqs = 1
        block_table = np.zeros((4, 16), dtype=np.int64)
        block_table[0, 2] = 10
        block_table[0, 3] = 11
        block_table[0, 4] = 12

        # first token → block 10: slot = 1280
        # last  token → block 12: slot = 12*128 + 10 = 1546
        slot_mapping = np.array([1280, 1400, 1500, 1546], dtype=np.int64)
        query_start_loc = np.array([0, 4], dtype=np.int64)

        swa_blocks = {"req_0": {0: [10, 11, 12]}}

        proc = self._build_processor(
            num_reqs, block_table, slot_mapping, query_start_loc,
            swa_blocks, grp_id=0, req_offset=4,
        )
        self._run_both_and_compare(
            proc, num_reqs, slot_mapping, block_table, query_start_loc
        )

    # ---- Scenario 4: multiple requests, mixed counts ----
    def test_multi_req_mixed_counts(self):
        """Two requests: one with count==1, another with count==2."""
        num_reqs = 2
        block_table = np.zeros((4, 16), dtype=np.int64)
        # req 0: 1 block
        block_table[0, 1] = 5
        # req 1: 2 blocks
        block_table[1, 3] = 20
        block_table[1, 4] = 21

        # req 0: first=last → block 5, slot = 5*128+10 = 650
        # req 1: first → block 20 (slot=2560), last → block 21 (slot=2688+5=2693)
        slot_mapping = np.array([650, 655, 2560, 2600, 2693], dtype=np.int64)
        query_start_loc = np.array([0, 2, 5], dtype=np.int64)

        swa_blocks = {
            "req_0": {0: [5]},
            "req_1": {0: [20, 21]},
        }

        proc = self._build_processor(
            num_reqs, block_table, slot_mapping, query_start_loc,
            swa_blocks, grp_id=0, req_offset=4,
        )
        self._run_both_and_compare(
            proc, num_reqs, slot_mapping, block_table, query_start_loc
        )

    # ---- Scenario 5: request with no match (should be skipped) ----
    def test_no_match_skipped(self):
        """Request whose target block id is not in block_table → skip."""
        num_reqs = 1
        block_table = np.zeros((4, 16), dtype=np.int64)
        block_table[0, 0] = 99  # block 99

        # first token maps to block 50 (not 99), so has_match = False
        slot_mapping = np.array([50 * 128, 50 * 128 + 5], dtype=np.int64)
        query_start_loc = np.array([0, 2], dtype=np.int64)

        swa_blocks = {"req_0": {0: [99]}}

        proc = self._build_processor(
            num_reqs, block_table, slot_mapping, query_start_loc,
            swa_blocks, grp_id=0, req_offset=4,
        )
        self._run_both_and_compare(
            proc, num_reqs, slot_mapping, block_table, query_start_loc
        )

    # ---- Scenario 6: if_new_block_lookahead triggers ----
    def test_lookahead_trigger(self):
        """last_token_slot+1 is exactly divisible by 128 → lookahead path."""
        num_reqs = 1
        block_table = np.zeros((4, 16), dtype=np.int64)
        block_table[0, 2] = 10

        # last token slot: we need (slot+1) % 128 == 0, so slot = 128*k - 1
        # e.g. slot = 10*128 - 1 = 1279 → block 9 (1279//128=9)
        # But we need block 10 to be in the table, so first_token → block 10
        # Actually let's set it up so both first and last point to block 10
        # last_slot = 10*128 + 127 = 1407, (1407+1)%128 = 0 ✓
        slot_mapping = np.array([1280, 1350, 1407], dtype=np.int64)
        query_start_loc = np.array([0, 3], dtype=np.int64)

        swa_blocks = {"req_0": {0: [10]}}

        proc = self._build_processor(
            num_reqs, block_table, slot_mapping, query_start_loc,
            swa_blocks, grp_id=0, req_offset=4,
        )
        self._run_both_and_compare(
            proc, num_reqs, slot_mapping, block_table, query_start_loc
        )

    # ---- Scenario 7: count >= 3 with lookahead ----
    def test_count3_lookahead(self):
        """count >= 3 combined with if_new_block_lookahead == 0."""
        num_reqs = 1
        block_table = np.zeros((4, 16), dtype=np.int64)
        block_table[0, 2] = 10
        block_table[0, 3] = 11
        block_table[0, 4] = 12

        # last token → block 12, slot = 12*128 + 127 = 1663, (1663+1)%128=0
        slot_mapping = np.array([1280, 1400, 1500, 1663], dtype=np.int64)
        query_start_loc = np.array([0, 4], dtype=np.int64)

        swa_blocks = {"req_0": {0: [10, 11, 12]}}

        proc = self._build_processor(
            num_reqs, block_table, slot_mapping, query_start_loc,
            swa_blocks, grp_id=0, req_offset=4,
        )
        self._run_both_and_compare(
            proc, num_reqs, slot_mapping, block_table, query_start_loc
        )

    # ---- Scenario 8: end != end_mtp ----
    def test_end_not_equal_end_mtp(self):
        """first token and last token in different blocks (end != end_mtp)."""
        num_reqs = 1
        block_table = np.zeros((4, 16), dtype=np.int64)
        block_table[0, 1] = 5
        block_table[0, 2] = 6
        block_table[0, 3] = 7

        # first token → block 5 (slot=640), last → block 7 (slot=896+10=906)
        # end_indices for block 5 → col 1
        # end_indices_mtp for block 7 → col 3
        slot_mapping = np.array([640, 770, 850, 906], dtype=np.int64)
        query_start_loc = np.array([0, 4], dtype=np.int64)

        swa_blocks = {"req_0": {0: [5, 6, 7]}}

        proc = self._build_processor(
            num_reqs, block_table, slot_mapping, query_start_loc,
            swa_blocks, grp_id=0, req_offset=4,
        )
        self._run_both_and_compare(
            proc, num_reqs, slot_mapping, block_table, query_start_loc
        )

    # ---- Scenario 9: draft_index >= 0 ----
    def test_draft_index_positive(self):
        """With draft_index >= 0, cur_metadata_grp_id should be 1."""
        num_reqs = 1
        block_table = np.zeros((4, 16), dtype=np.int64)
        block_table[0, 2] = 10

        slot_mapping = np.array([1280, 1300], dtype=np.int64)
        query_start_loc = np.array([0, 2], dtype=np.int64)

        swa_blocks = {"req_0": {1: [10]}}  # grp_id = 1

        grp_id = 0
        req_ids = ["req_0"]
        req_id_to_idx = {"req_0": 0}

        hbm_pool_0 = _make_hbm_pool(0, num_slots=40)
        hbm_pool_1 = _make_hbm_pool(1, num_slots=40)
        hbm_pool = {0: hbm_pool_0[0], 1: hbm_pool_1[1]}

        bt_pool = {0: {"req_offset": 4}, 1: {"req_offset": 4}}
        kv_cfg = FakeKVCacheConfig(
            kv_cache_groups={
                0: FakeKVCacheGroup(layer_names=["layer0"]),
                1: FakeKVCacheGroup(layer_names=["layer0"]),
            }
        )

        proc = SWAPostProcessor(
            req_id_to_idx=req_id_to_idx,
            req_ids_update_buffer=req_ids,
            record_req_ids_swa_block=swa_blocks,
            hbm_buffer_block_table_pool=bt_pool,
            hbm_buffer_pool=hbm_pool,
            kv_cache_config=kv_cfg,
            metadata_grp_id=grp_id,
        )
        self._run_both_and_compare(
            proc, num_reqs, slot_mapping, block_table, query_start_loc,
            draft_index=1,
        )

    # ---- Scenario 10: req_id not in req_id_to_idx ----
    def test_req_id_missing_from_mapping(self):
        """Request id not in mapping → should be skipped gracefully."""
        num_reqs = 1
        block_table = np.zeros((4, 16), dtype=np.int64)
        block_table[0, 2] = 10

        slot_mapping = np.array([1280, 1300], dtype=np.int64)
        query_start_loc = np.array([0, 2], dtype=np.int64)

        swa_blocks = {"req_0": {0: [10]}}

        # req_id_to_idx does NOT contain "req_0"
        proc = self._build_processor(
            num_reqs, block_table, slot_mapping, query_start_loc,
            swa_blocks, req_id_to_idx={}, grp_id=0, req_offset=4,
        )
        self._run_both_and_compare(
            proc, num_reqs, slot_mapping, block_table, query_start_loc
        )

    # ---- Scenario 11: count==2, end_mtp == end ----
    def test_count2_end_match(self):
        """count==2 and end_mtp equals end → different offset path."""
        num_reqs = 1
        block_table = np.zeros((4, 16), dtype=np.int64)
        # 2 blocks: both first and last token in same final block
        block_table[0, 3] = 10
        block_table[0, 4] = 11

        # first token → block 11 (slot=11*128=1408)
        # last token → block 11 (slot=11*128+5=1413)
        # So target_block_id = 11, mtp_block_id = 11
        # end_indices = col 4, end_indices_mtp = col 4 → end == end_mtp
        # But matched_idx for first nonzero in [:end+1] = col 3
        # count = end_mtp - matched_idx + 1 = 4 - 3 + 1 = 2
        slot_mapping = np.array([1408, 1410, 1413], dtype=np.int64)
        query_start_loc = np.array([0, 3], dtype=np.int64)

        swa_blocks = {"req_0": {0: [10, 11]}}

        proc = self._build_processor(
            num_reqs, block_table, slot_mapping, query_start_loc,
            swa_blocks, grp_id=0, req_offset=4,
        )
        self._run_both_and_compare(
            proc, num_reqs, slot_mapping, block_table, query_start_loc
        )

    # ---- Scenario 12: 3 requests with different counts ----
    def test_three_reqs_various(self):
        """Three requests: count 1, 2, 3."""
        num_reqs = 3
        block_table = np.zeros((4, 16), dtype=np.int64)
        # req 0: count 1 block at col 5 = block 30
        block_table[0, 5] = 30
        # req 1: count 2 blocks at col 2,3 = blocks 40,41
        block_table[1, 2] = 40
        block_table[1, 3] = 41
        # req 2: count 3 blocks at col 1,2,3 = blocks 50,51,52
        block_table[2, 1] = 50
        block_table[2, 2] = 51
        block_table[2, 3] = 52

        # req 0: first=last → block 30, slot = 30*128+10=3850
        # req 1: first → block 40 (5120), last → block 41 (5248+5=5253)
        # req 2: first → block 50 (6400), last → block 52 (6656+20=6676)
        slot_mapping = np.array([
            3850, 3860,                    # req 0: 2 tokens
            5120, 5200, 5253,              # req 1: 3 tokens
            6400, 6530, 6600, 6676,        # req 2: 4 tokens
        ], dtype=np.int64)
        query_start_loc = np.array([0, 2, 5, 9], dtype=np.int64)

        swa_blocks = {
            "req_0": {0: [30]},
            "req_1": {0: [40, 41]},
            "req_2": {0: [50, 51, 52]},
        }

        proc = self._build_processor(
            num_reqs, block_table, slot_mapping, query_start_loc,
            swa_blocks, grp_id=0, req_offset=4,
        )
        self._run_both_and_compare(
            proc, num_reqs, slot_mapping, block_table, query_start_loc
        )

    # ---- Scenario 13: first-time scheduling with mismatched swa block ----
    def test_first_time_swa_mismatch(self):
        """First schedule where block_table_old[matched_idx] != swa_blocks[0]."""
        num_reqs = 1
        block_table = np.zeros((4, 16), dtype=np.int64)
        block_table[0, 2] = 10

        slot_mapping = np.array([1280, 1300], dtype=np.int64)
        query_start_loc = np.array([0, 2], dtype=np.int64)

        # swa_blocks[0] = 99, but block_table_old[0, matched_idx] = 10 → mismatch
        swa_blocks = {"req_0": {0: [99]}}

        proc = self._build_processor(
            num_reqs, block_table, slot_mapping, query_start_loc,
            swa_blocks, grp_id=0, req_offset=4,
        )
        self._run_both_and_compare(
            proc, num_reqs, slot_mapping, block_table, query_start_loc
        )

    # ---- Scenario 15: count==2 with lookahead ----
    def test_count2_lookahead(self):
        """count==2 with if_new_block_lookahead == 0."""
        num_reqs = 1
        block_table = np.zeros((4, 16), dtype=np.int64)
        block_table[0, 3] = 10
        block_table[0, 4] = 11

        # last token → block 11, slot=11*128+127=1535, (1535+1)%128=0
        slot_mapping = np.array([1280, 1400, 1535], dtype=np.int64)
        query_start_loc = np.array([0, 3], dtype=np.int64)
        swa_blocks = {"req_0": {0: [10, 11]}}

        proc = self._build_processor(
            num_reqs, block_table, slot_mapping, query_start_loc,
            swa_blocks, grp_id=0, req_offset=4,
        )
        self._run_both_and_compare(
            proc, num_reqs, slot_mapping, block_table, query_start_loc
        )

    # ---- Scenario 16: count>=3, end == end_mtp ----
    def test_count3_end_eq_mtp(self):
        """count>=3 where end == end_mtp."""
        num_reqs = 1
        block_table = np.zeros((4, 16), dtype=np.int64)
        block_table[0, 1] = 5
        block_table[0, 2] = 6
        block_table[0, 3] = 7

        slot_mapping = np.array([896, 900, 905, 906], dtype=np.int64)
        query_start_loc = np.array([0, 4], dtype=np.int64)
        swa_blocks = {"req_0": {0: [5, 6, 7]}}

        proc = self._build_processor(
            num_reqs, block_table, slot_mapping, query_start_loc,
            swa_blocks, grp_id=0, req_offset=4,
        )
        self._run_both_and_compare(
            proc, num_reqs, slot_mapping, block_table, query_start_loc
        )

    # ---- Scenario 17: all-zero block table ----
    def test_all_zero_block_table(self):
        """Block table is all zeros."""
        num_reqs = 1
        block_table = np.zeros((4, 16), dtype=np.int64)

        slot_mapping = np.array([0, 5], dtype=np.int64)
        query_start_loc = np.array([0, 2], dtype=np.int64)
        swa_blocks = {"req_0": {0: [0]}}

        proc = self._build_processor(
            num_reqs, block_table, slot_mapping, query_start_loc,
            swa_blocks, grp_id=0, req_offset=4,
        )
        self._run_both_and_compare(
            proc, num_reqs, slot_mapping, block_table, query_start_loc
        )
