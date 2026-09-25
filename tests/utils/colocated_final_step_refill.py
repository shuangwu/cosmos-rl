# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Test-only producer quality mask; never replace training or collectives.

Used by test_colocated.py with two one-GPU policy/embedded-rollout replicas.
Reject the second group of the last update on exactly one replica, require a
replacement group, and verify all three real optimizer updates on both replicas.
"""

import os
from collections import Counter

from cosmos_rl.policy.trainer import GRPOTrainer
from cosmos_rl.rollout.worker.colocated.rollout_control import (
    ColocatedRolloutControlWorker,
)


def install():
    case = os.environ["COLOCATED_REFILL_CASE"]
    assert case in {"healthy", "reject-final-group"}
    target = os.environ["COLOCATED_REFILL_TARGET"] == "1"
    generated = Counter()
    updates = []
    rejected = []
    original_filter = (
        ColocatedRolloutControlWorker._filter_valid_rollout_results_and_report
    )
    original_train = GRPOTrainer.step_training

    def quality_mask(self, results, payloads):
        step = self.current_weight_version + 1
        assert len(results) == len(payloads) == 1
        generated[step] += 1
        drop = (
            case == "reject-final-group"
            and target
            and step == 3
            and generated[step] == 2
        )
        result = results[0]
        assert len(result.completions) == 8
        result.completion_trainable = [not drop] * 8
        if drop:
            rejected.append(step)
            print("COLOCATED_REFILL_REJECT step=3 group=2 count=8", flush=True)
        return original_filter(self, results, payloads)

    def train(self, rollouts, current_step, total_steps, *args, **kwargs):
        assert len(rollouts) == 16 and total_steps == 3
        assert current_step == len(updates) + 1
        result = original_train(
            self, rollouts, current_step, total_steps, *args, **kwargs
        )
        updates.append(current_step)
        print(f"COLOCATED_REFILL_UPDATE step={current_step} count=16", flush=True)
        return result

    ColocatedRolloutControlWorker._filter_valid_rollout_results_and_report = (
        quality_mask
    )
    GRPOTrainer.step_training = train

    def verify():
        drop = case == "reject-final-group" and target
        assert updates == [1, 2, 3], updates
        assert rejected == ([3] if drop else []), rejected
        assert generated == {1: 2, 2: 2, 3: 3 if drop else 2}, generated
        print(
            f"COLOCATED_REFILL_PASS case={case} target={target} updates=3 final_groups={generated[3]}",
            flush=True,
        )

    return verify
