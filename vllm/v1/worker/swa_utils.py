"""Utilities for Sliding Window Attention (SWA) block table post-processing.

This module provides both a loop-based (reference) and a vectorized
implementation of ``_post_process_fake_metadata``, which remaps physical
block-table entries into an HBM ping-pong buffer layout and adjusts the
corresponding ``slot_mapping`` array.

The vectorized path is functionally equivalent to the loop path.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class SWAPostProcessor:
    """Encapsulates SWA post-processing state and logic."""

    # The block size used for slot → block-id mapping.
    BLOCK_SIZE: int = 128

    def __init__(
        self,
        req_id_to_idx: Dict[str, int],
        req_ids_update_buffer: List[str],
        record_req_ids_swa_block: Dict[str, Dict[int, List[int]]],
        hbm_buffer_block_table_pool: Dict[int, Dict[str, Any]],
        hbm_buffer_pool: Dict[int, Dict[str, Any]],
        kv_cache_config: Any,
        metadata_grp_id: int,
        last_num_non_zeros: Optional[Dict[Tuple, int]] = None,
        record_req_sched_times: Optional[Dict[Tuple, int]] = None,
        device: Any = None,
        swa_copy_stream: Any = None,
    ):
        self.req_id_to_idx = req_id_to_idx
        self.req_ids_update_buffer = req_ids_update_buffer
        self.record_req_ids_swa_block = record_req_ids_swa_block
        self.hbm_buffer_block_table_pool = hbm_buffer_block_table_pool
        self.hbm_buffer_pool = hbm_buffer_pool
        self.kv_cache_config = kv_cache_config
        self.metadata_grp_id = metadata_grp_id
        self.last_num_non_zeros: Dict[Tuple, int] = (
            last_num_non_zeros if last_num_non_zeros is not None else {}
        )
        self.record_req_sched_times: Dict[Tuple, int] = (
            record_req_sched_times if record_req_sched_times is not None else {}
        )
        self.device = device
        self.swa_copy_stream = swa_copy_stream

    # ------------------------------------------------------------------
    # Helpers shared by both implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _find_first_nonzero_index(req_block_row: np.ndarray) -> Optional[int]:
        non_zero_mask = req_block_row != 0
        if not non_zero_mask.any():
            return None
        return int(non_zero_mask.argmax())

    def _update_req_block_table(
        self,
        table_row: np.ndarray,
        start: int,
        end: int,
        offset_blc: int,
        swa_len: int,
        count: int,
        end_mtp: int,
        if_new_block_lookahead: int,
    ) -> Tuple[int, int]:
        """3-block ping-pong update (reference implementation)."""
        if count == 1:
            table_row[end] = offset_blc + 1
            if if_new_block_lookahead == 0:
                table_row[end + 1] = offset_blc + 2
            return offset_blc + 1, offset_blc + 1
        elif count == 2:
            table_row[end_mtp - 1: end_mtp + 1] = [
                offset_blc + 1,
                offset_blc + 2,
            ]
            if if_new_block_lookahead == 0:
                table_row[end_mtp + 1] = offset_blc + 3
                table_row[end_mtp] = offset_blc + 2
                table_row[end_mtp - 1] = offset_blc + 1
            if end_mtp == end:
                return offset_blc + 2, offset_blc + 2
            else:
                return offset_blc + 1, offset_blc + 2
        else:
            table_row[end_mtp] = offset_blc + 3
            table_row[end_mtp - 1] = offset_blc + 2
            table_row[end_mtp - 2] = offset_blc + 1
            if if_new_block_lookahead == 0:
                table_row[end_mtp + 1] = offset_blc + 3
                table_row[end_mtp] = offset_blc + 2
                table_row[end_mtp - 1] = offset_blc + 1
                table_row[end_mtp - 2] = 0
                return offset_blc + 2, offset_blc + 2
            if end_mtp == end:
                return offset_blc + 3, offset_blc + 3
            else:
                return offset_blc + 2, offset_blc + 3

    def _move_hbm_buffer_slots(
        self, offset_blc: int, cur_metadata_grp_id: int, draft_index: int
    ) -> None:
        if draft_index >= 0:
            return
        layers = self.kv_cache_config.kv_cache_groups[
            cur_metadata_grp_id
        ].layer_names
        for ln in layers:
            pool = self.hbm_buffer_pool[cur_metadata_grp_id][ln]
            pool[offset_blc + 1] = pool[offset_blc + 2].clone()
            pool[offset_blc + 2] = pool[offset_blc + 3].clone()
            pool[offset_blc + 3] = 0

    # ------------------------------------------------------------------
    # Loop-based reference implementation (known-correct)
    # ------------------------------------------------------------------

    def post_process_loop(
        self,
        num_reqs: int,
        slot_mapping: np.ndarray,
        block_table: np.ndarray,
        common_attn_metadata: Any,
        draft_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Loop-based reference implementation."""
        if draft_index < 0:
            cur_metadata_grp_id = self.metadata_grp_id
        else:
            cur_metadata_grp_id = 1

        query_start_loc = common_attn_metadata.query_start_loc_cpu
        query_lens = query_start_loc[1:] - query_start_loc[:-1]

        block_table_old = block_table.copy()
        block_table.fill(0)

        block_offsets_batch = np.zeros((num_reqs, 2), dtype=np.int64)

        first_token_slots = slot_mapping[query_start_loc[:num_reqs]]
        first_token_slots = np.atleast_1d(first_token_slots)
        target_block_ids = first_token_slots // self.BLOCK_SIZE

        mask = block_table_old[:num_reqs] == target_block_ids[:, np.newaxis]
        has_match = mask.any(axis=1)
        end_indices = mask.argmax(axis=1)

        last_token_slots = slot_mapping[query_start_loc[1: num_reqs + 1] - 1]
        last_token_slots = np.atleast_1d(last_token_slots)
        mtp_block_ids = last_token_slots // self.BLOCK_SIZE
        mask_mtp = block_table_old[:num_reqs] == mtp_block_ids[:, np.newaxis]
        has_match_mtp = mask_mtp.any(axis=1)  # noqa: F841
        end_indices_mtp = mask_mtp.argmax(axis=1)
        if_new_block_lookahead = (last_token_slots + 1) % self.BLOCK_SIZE

        actual_num_reqs = min(num_reqs, len(self.req_ids_update_buffer))

        for idx_req in range(actual_num_reqs):
            if not has_match[idx_req]:
                continue

            req_id = self.req_ids_update_buffer[idx_req]
            idx_req_real = self.req_id_to_idx.get(req_id)
            if idx_req_real is None:
                continue

            end_idx = end_indices[idx_req]
            swa_blocks = self.record_req_ids_swa_block[req_id][
                cur_metadata_grp_id
            ]

            matched_idx = self._find_first_nonzero_index(
                block_table_old[idx_req, : end_idx + 1]
            )

            if matched_idx is not None:
                num_non_zeros = end_indices_mtp[idx_req] - matched_idx + 1
                offset_blc = (
                    self.hbm_buffer_block_table_pool[cur_metadata_grp_id][
                        "req_offset"
                    ]
                    * idx_req_real
                )

                offset, offset_mtp = self._update_req_block_table(
                    block_table[idx_req],
                    matched_idx,
                    end_idx,
                    offset_blc,
                    len(swa_blocks),
                    num_non_zeros,
                    end_indices_mtp[idx_req],
                    if_new_block_lookahead[idx_req],
                )
                block_offsets_batch[idx_req, 0] = (
                    block_table_old[idx_req, end_idx] - offset
                )
                block_offsets_batch[idx_req, 1] = (
                    block_table_old[idx_req, end_indices_mtp[idx_req]]
                    - offset_mtp
                )

                state_key = (req_id, cur_metadata_grp_id)
                last_count = self.last_num_non_zeros.get(state_key, 0)

                if state_key not in self.record_req_sched_times:
                    if block_table_old[idx_req, matched_idx] != swa_blocks[0]:
                        layers = self.kv_cache_config.kv_cache_groups[
                            cur_metadata_grp_id
                        ].layer_names
                        for ln in layers:
                            pool = self.hbm_buffer_pool[cur_metadata_grp_id][
                                ln
                            ]
                            pool[offset_blc + 1] = pool[
                                offset_blc + 2
                            ].clone()
                    self.record_req_sched_times[state_key] = 1

                if num_non_zeros > 3 and num_non_zeros > last_count:
                    self._move_hbm_buffer_slots(
                        offset_blc, cur_metadata_grp_id, draft_index
                    )
                self.last_num_non_zeros[state_key] = num_non_zeros

        # Slot-mapping adjustment
        base_offs = block_offsets_batch[:actual_num_reqs, 0]
        mtp_offs = block_offsets_batch[:actual_num_reqs, 1]

        repeat_counts = query_lens[:actual_num_reqs]
        token_offsets = np.repeat(mtp_offs, repeat_counts)

        first_token_indices = (
            query_start_loc[:actual_num_reqs] - query_start_loc[0]
        )
        token_offsets[first_token_indices] = base_offs

        limit = token_offsets.shape[0]
        slot_mapping[:limit] -= self.BLOCK_SIZE * token_offsets

        return block_table, slot_mapping

    # ------------------------------------------------------------------
    # Vectorized implementation (fixed to match loop logic)
    # ------------------------------------------------------------------

    def post_process_vectorized(
        self,
        num_reqs: int,
        slot_mapping: np.ndarray,
        block_table: np.ndarray,
        common_attn_metadata: Any,
        draft_index: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized implementation, strictly aligned with loop logic."""
        if draft_index < 0:
            cur_metadata_grp_id = self.metadata_grp_id
        else:
            cur_metadata_grp_id = 1

        query_start_loc = common_attn_metadata.query_start_loc_cpu
        query_lens = query_start_loc[1:] - query_start_loc[:-1]

        # ---- FIX 1: Copy then zero, matching loop order ----
        block_table_old = block_table.copy()
        block_table.fill(0)

        # 1. Basic index preparation (computed from block_table_old)
        first_token_slots = np.atleast_1d(
            slot_mapping[query_start_loc[:num_reqs]]
        )
        target_block_ids = first_token_slots // self.BLOCK_SIZE

        last_token_slots = slot_mapping[
            query_start_loc[1: num_reqs + 1] - 1
        ]
        mtp_token_slots = np.atleast_1d(last_token_slots)
        mtp_block_ids = mtp_token_slots // self.BLOCK_SIZE

        # ---- FIX 2: Compute masks from block_table_old ----
        bt_old_view = block_table_old[:num_reqs]

        mask_mtp = bt_old_view == mtp_block_ids[:, np.newaxis]
        end_indices_mtp = mask_mtp.argmax(axis=1)

        if_new_block_lookahead = (mtp_token_slots + 1) % self.BLOCK_SIZE
        is_look = np.atleast_1d(if_new_block_lookahead == 0)

        mask_base = bt_old_view == target_block_ids[:, np.newaxis]
        has_match = mask_base.any(axis=1)
        end_indices = mask_base.argmax(axis=1)

        actual_num_reqs = min(num_reqs, len(self.req_ids_update_buffer))
        req_ids = self.req_ids_update_buffer[:actual_num_reqs]
        idx_req_real_all = np.array(
            [self.req_id_to_idx.get(rid, -1) for rid in req_ids]
        )

        valid_mask = has_match[:actual_num_reqs] & (idx_req_real_all != -1)
        if not valid_mask.any():
            return block_table, slot_mapping

        valid_indices = np.where(valid_mask)[0]
        valid_ends_mtp = end_indices_mtp[valid_indices]
        valid_ends = end_indices[valid_indices]
        valid_real_idxs = idx_req_real_all[valid_indices]

        # ---- FIX 3: Use block_table_old for subset_table ----
        subset_table = bt_old_view[valid_indices]

        # ---- FIX 4: Limit nonzero search to columns 0..end_idx ----
        # Build a column mask so we only search within [:end_idx+1] per row
        ncols = subset_table.shape[1]
        col_range = np.arange(ncols)
        col_mask = col_range[np.newaxis, :] <= valid_ends[:, np.newaxis]
        masked_nonzero = (subset_table != 0) & col_mask
        has_nonzero = masked_nonzero.any(axis=1)
        # For rows with no non-zero, argmax returns 0 but we must skip them
        matched_indices = masked_nonzero.argmax(axis=1)

        # Filter to only rows that actually have a nonzero entry
        keep = has_nonzero
        if not keep.all():
            # Further restrict valid set to those with a matched nonzero
            valid_indices = valid_indices[keep]
            valid_ends_mtp = valid_ends_mtp[keep]
            valid_ends = valid_ends[keep]
            valid_real_idxs = valid_real_idxs[keep]
            matched_indices = matched_indices[keep]
            subset_table = subset_table[keep]

        if valid_indices.size == 0:
            return block_table, slot_mapping

        num_non_zeros = valid_ends_mtp - matched_indices + 1

        pool_offsets = self.hbm_buffer_block_table_pool[
            cur_metadata_grp_id
        ]["req_offset"]
        offset_blc_all = pool_offsets * valid_real_idxs

        block_offsets_batch = np.zeros((num_reqs, 2), dtype=np.int64)

        is_end_match = valid_ends == valid_ends_mtp
        p1, p2, p3 = 1, 2, 3

        # --- mask1: count == 1 ---
        mask1 = num_non_zeros == 1
        if mask1.any():
            sub_idx = np.where(mask1)[0]
            idx = valid_indices[sub_idx]
            e = valid_ends[mask1]
            e_m = valid_ends_mtp[mask1]  # FIX 5: need e_m for offsets col 1
            base = offset_blc_all[mask1]

            block_table[idx, e] = base + p1

            l_mask = is_look[idx]
            if l_mask.any():
                block_table[idx[l_mask], e[l_mask] + 1] = (
                    base[l_mask] + p2
                )

            block_offsets_batch[idx, 0] = (
                block_table_old[idx, e] - (base + p1)
            )
            # ---- FIX 5: Use e_m (end_indices_mtp) for column 1 ----
            block_offsets_batch[idx, 1] = (
                block_table_old[idx, e_m] - (base + p1)
            )

        # --- mask2: count == 2 ---
        mask2 = num_non_zeros == 2
        if mask2.any():
            sub_idx = np.where(mask2)[0]
            idx = valid_indices[sub_idx]
            e = valid_ends[mask2]
            e_m = valid_ends_mtp[mask2]
            base = offset_blc_all[mask2]

            block_table[idx, e_m] = base + p2
            block_table[idx, e_m - 1] = base + p1

            l_mask = is_look[idx]
            if l_mask.any():
                block_table[idx[l_mask], e_m[l_mask] + 1] = (
                    base[l_mask] + p3
                )
                # When is_look, the loop version also re-assigns these:
                block_table[idx[l_mask], e_m[l_mask]] = (
                    base[l_mask] + p2
                )
                block_table[idx[l_mask], e_m[l_mask] - 1] = (
                    base[l_mask] + p1
                )

            m_mask = is_end_match[mask1.sum():mask1.sum() + mask2.sum()]
            # ---- FIX: Recompute m_mask correctly from sub_idx ----
            m_mask = is_end_match[sub_idx]
            block_offsets_batch[idx, 0] = np.where(
                m_mask,
                block_table_old[idx, e] - (base + p2),
                block_table_old[idx, e] - (base + p1),
            )
            block_offsets_batch[idx, 1] = (
                block_table_old[idx, e_m] - (base + p2)
            )

        # --- mask3: count >= 3 ---
        mask3 = num_non_zeros >= 3
        if mask3.any():
            sub_idx = np.where(mask3)[0]
            idx = valid_indices[sub_idx]
            e = valid_ends[mask3]
            e_m = valid_ends_mtp[mask3]
            base = offset_blc_all[mask3]

            block_table[idx, e_m] = base + p3
            block_table[idx, e_m - 1] = base + p2
            block_table[idx, e_m - 2] = base + p1

            l_mask = is_look[idx]
            if l_mask.any():
                l_idx = idx[l_mask]
                l_em = e_m[l_mask]
                l_base = base[l_mask]
                block_table[l_idx, l_em + 1] = l_base + p3
                block_table[l_idx, l_em] = l_base + p2
                block_table[l_idx, l_em - 1] = l_base + p1
                block_table[l_idx, l_em - 2] = 0

            # Compute return values matching _update_req_block_table logic
            m_mask = is_end_match[sub_idx]

            # Default to 0, then fill per branch
            res_base = np.zeros(sub_idx.size, dtype=block_table.dtype)
            res_mtp = np.zeros(sub_idx.size, dtype=block_table.dtype)

            # Branch: if_new_block_lookahead == 0 → return (base+p2, base+p2)
            res_base = np.where(l_mask, base + p2, res_base)
            res_mtp = np.where(l_mask, base + p2, res_mtp)

            # Branch: NOT lookahead AND end_mtp == end → (base+p3, base+p3)
            other = ~l_mask
            res_base = np.where(other & m_mask, base + p3, res_base)
            res_mtp = np.where(other & m_mask, base + p3, res_mtp)

            # Branch: NOT lookahead AND end_mtp != end → (base+p2, base+p3)
            final_else = other & ~m_mask
            res_base = np.where(final_else, base + p2, res_base)
            res_mtp = np.where(final_else, base + p3, res_mtp)

            block_offsets_batch[idx, 0] = block_table_old[idx, e] - res_base
            block_offsets_batch[idx, 1] = (
                block_table_old[idx, e_m] - res_mtp
            )

        # --- HBM state maintenance ---
        state_keys = [
            (rid, cur_metadata_grp_id) for rid in req_ids[:actual_num_reqs]
        ]

        # ---- FIX 6: First-time check aligned with loop ----
        for vi_pos, batch_idx in enumerate(valid_indices):
            req_id = req_ids[batch_idx]
            state_key = (req_id, cur_metadata_grp_id)

            if state_key not in self.record_req_sched_times:
                swa_blocks = self.record_req_ids_swa_block[req_id][
                    cur_metadata_grp_id
                ]
                if (
                    block_table_old[batch_idx, matched_indices[vi_pos]]
                    != swa_blocks[0]
                ):
                    layers = self.kv_cache_config.kv_cache_groups[
                        cur_metadata_grp_id
                    ].layer_names
                    for ln in layers:
                        pool = self.hbm_buffer_pool[cur_metadata_grp_id][ln]
                        pool[offset_blc_all[vi_pos] + 1] = pool[
                            offset_blc_all[vi_pos] + 2
                        ].clone()
                self.record_req_sched_times[state_key] = 1

        # Move buffer slots when count grows beyond 3
        for vi_pos, batch_idx in enumerate(valid_indices):
            req_id = req_ids[batch_idx]
            state_key = (req_id, cur_metadata_grp_id)
            last_count = self.last_num_non_zeros.get(state_key, 0)
            cnt = num_non_zeros[vi_pos]
            if cnt > 3 and cnt > last_count:
                self._move_hbm_buffer_slots(
                    int(offset_blc_all[vi_pos]),
                    cur_metadata_grp_id,
                    draft_index,
                )
            self.last_num_non_zeros[state_key] = int(cnt)

        # Slot-mapping adjustment
        base_offs = block_offsets_batch[:actual_num_reqs, 0]
        mtp_offs = block_offsets_batch[:actual_num_reqs, 1]

        repeat_counts = query_lens[:actual_num_reqs]
        token_offsets = np.repeat(mtp_offs, repeat_counts)

        first_token_indices = (
            query_start_loc[:actual_num_reqs] - query_start_loc[0]
        )
        token_offsets[first_token_indices] = base_offs

        limit = token_offsets.shape[0]
        slot_mapping[:limit] -= self.BLOCK_SIZE * token_offsets

        return block_table, slot_mapping
