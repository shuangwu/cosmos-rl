# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Correctness tests for globally aligned P2R NCCL grouping."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cosmos_rl.policy.config import Config
from cosmos_rl.utils import constant
from cosmos_rl.utils.parallelism_map import (
    WeightSyncInstructionsGroup,
    build_p2r_sync_group_index,
    iter_p2r_sync_rounds,
)


def _instruction(group_index: int | None) -> WeightSyncInstructionsGroup:
    return WeightSyncInstructionsGroup([], sync_group_index=group_index)


def _round_indices(instructions, groups_per_round: int) -> list[list[int]]:
    return [
        [instruction.sync_group_index for instruction in sync_round]
        for sync_round in iter_p2r_sync_rounds(instructions, groups_per_round)
    ]


def test_global_group_index_keeps_joined_parameters_together() -> None:
    group_index = build_p2r_sync_group_index(
        [["a", "k_proj", "q_proj", "z"], ["a", "z"]],
        [["a", "k_proj", "q_proj", "z"]],
        [("k_proj", "q_proj")],
    )

    assert group_index == {"a": 0, "z": 1, "k_proj": 2, "q_proj": 2}


def test_rounds_use_global_indices_instead_of_local_counts() -> None:
    # This is the shape of the #307 failure: a receiver sees an intervening
    # group that a particular policy rank does not. Local batches of two put
    # group 2 in different NCCL calls; global rounds keep it in round 1.
    policy_rank = [_instruction(index) for index in (0, 2, 3)]
    rollout_rank = [_instruction(index) for index in (0, 1, 2, 3)]

    assert _round_indices(policy_rank, 2) == [[0], [2, 3]]
    assert _round_indices(rollout_rank, 2) == [[0, 1], [2, 3]]


def test_disabled_grouping_preserves_legacy_instructions() -> None:
    instructions = [_instruction(None), _instruction(None)]

    assert list(iter_p2r_sync_rounds(instructions, 0)) == [
        [instructions[0]],
        [instructions[1]],
    ]


def test_enabled_grouping_rejects_missing_or_reordered_metadata() -> None:
    with pytest.raises(RuntimeError, match="requires controller-assigned"):
        list(iter_p2r_sync_rounds([_instruction(None)], 2))

    with pytest.raises(RuntimeError, match="monotonically"):
        list(iter_p2r_sync_rounds([_instruction(2), _instruction(1)], 2))


def test_p2r_group_size_uses_config_with_legacy_env_override(monkeypatch) -> None:
    config = SimpleNamespace(train=SimpleNamespace(p2r_sync_groups_per_round=4))
    monkeypatch.delenv("COSMOS_P2R_NCCL_GROUP_SIZE", raising=False)
    assert constant.get_p2r_nccl_group_size(config) == 4

    monkeypatch.setenv("COSMOS_P2R_NCCL_GROUP_SIZE", "8")
    assert constant.get_p2r_nccl_group_size(config) == 8


def test_batching_knobs_are_config_file_fields() -> None:
    defaults = Config()
    assert defaults.train.p2r_sync_groups_per_round == 0
    assert defaults.rollout.r2r_sync_pack_tensors is False
    assert defaults.rollout.r2r_sync_bucket_size_bytes == 512 * 1024 * 1024

    configured = Config(
        train={"p2r_sync_groups_per_round": 4},
        rollout={
            "r2r_sync_pack_tensors": True,
            "r2r_sync_bucket_size_bytes": 256 * 1024 * 1024,
        },
    )
    assert configured.train.p2r_sync_groups_per_round == 4
    assert configured.rollout.r2r_sync_pack_tensors is True
    assert configured.rollout.r2r_sync_bucket_size_bytes == 256 * 1024 * 1024
