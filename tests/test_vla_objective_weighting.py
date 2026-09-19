# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real VLA trainer loops with CPU model/clock substitutes."""

from contextlib import nullcontext
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from cosmos_rl.policy.trainer.vla_trainer.pi_grpo_trainer import PI05GRPOTrainer
from cosmos_rl.policy.trainer.vla_trainer.vla_trainer import OpenVLAGRPOTrainer


class Model(torch.nn.Module):
    action_chunk = 1
    action_env_dim = 1

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

    def _set_fsdp_reshard_after_forward(self, _):
        pass

    def get_log_prob_value(self, *, state, **kwargs):
        values = (self.weight * state).reshape(-1, 1, 1, 1)
        return values, torch.zeros_like(values)

    def forward_with_trajectory_structure(self, input_ids, *args, **kwargs):
        return SimpleNamespace(logprobs=self.weight * input_ids)


class Packer:
    def get_policy_input(self, rollout, device):
        return rollout

    def policy_collate_fn(self, policy_input, max_chunks):
        values = torch.tensor(
            policy_input.values + [0.0] * (max_chunks - len(policy_input.values)),
            dtype=torch.float64,
        ).reshape(-1, 1)
        mask = torch.arange(max_chunks).reshape(-1, 1) < len(policy_input.values)
        return dict(
            input_ids=values,
            states=values,
            logprob_masks=mask,
            images=values.reshape(-1, 1, 1),
            image_masks=mask,
            pixel_values=values,
            attention_mask=mask,
            responses=values,
            chains=values,
            denoise_inds=values,
            old_log_probs=torch.zeros_like(values),
        )


@pytest.mark.parametrize("cls", [PI05GRPOTrainer, OpenVLAGRPOTrainer])
@pytest.mark.parametrize(
    "weighting,expected", [("sample", 0.3), ("episode", 0.4), (None, 0.4)]
)
@pytest.mark.parametrize("chunk_size", [1, 2, 3])
@pytest.mark.parametrize("empty", [False, True])
def test_production_objective_is_chunk_invariant(
    monkeypatch, cls, weighting, expected, chunk_size, empty, partial=False
):
    if weighting is None and empty:
        pytest.skip(
            "Legacy empty-episode behavior is not changed by this opt-in feature"
        )
    if weighting is None and cls is PI05GRPOTrainer:
        expected = {1: 0.6, 2: 0.525, 3: 0.4}[chunk_size]
    monkeypatch.setattr(
        torch.cuda,
        "Event",
        lambda **kwargs: SimpleNamespace(
            record=lambda: None, elapsed_time=lambda other: 0.0
        ),
    )
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch, "autocast", lambda **kwargs: nullcontext())
    # Simulator/model dependencies are not involved in this loss-loop test.
    libero = ModuleType("cosmos_rl.simulators.libero.utils")
    libero.LIBERO_MAX_STEPS_MAP = {"test": 3}
    monkeypatch.setitem(sys.modules, "cosmos_rl.simulators.libero.utils", libero)
    from cosmos_rl.dispatcher.data.packer import vla_data_packer

    monkeypatch.setattr(vla_data_packer, "_get_vla_constants", lambda: (1, 1, 1))
    trainer = object.__new__(cls)
    trainer.device = torch.device("cpu")
    trainer.model = Model()
    trainer.data_packer = Packer()
    if partial:
        original = trainer.data_packer.policy_collate_fn

        def partially_valid(policy_input, max_chunks):
            data = original(policy_input, max_chunks)
            for key in ("input_ids", "logprob_masks", "old_log_probs"):
                data[key] = data[key].repeat(1, 2)
            if len(policy_input.values) == 3:
                data["logprob_masks"][2, 1] = False
            return data

        trainer.data_packer.policy_collate_fn = partially_valid
    trainer.data_packer.policy_collate_fn = Mock(
        wraps=trainer.data_packer.policy_collate_fn
    )
    trainer.parallel_dims = SimpleNamespace(dp_enabled=False)
    trainer.global_rank = 0
    trainer.config = SimpleNamespace(
        vla=SimpleNamespace(
            objective_weighting=weighting, training_chunk_size=chunk_size
        ),
        logging=SimpleNamespace(logger=[]),
        train=SimpleNamespace(
            fsdp_reshard_after_forward="default",
            train_policy=SimpleNamespace(
                dataset=SimpleNamespace(subset="test"),
                temperature=1.0,
                epsilon_low=0.2,
                epsilon_high=0.2,
            ),
        ),
    )
    trainer.optimizers = torch.optim.SGD(trainer.model.parameters(), lr=0.1)
    trainer.lr_schedulers = torch.optim.lr_scheduler.StepLR(
        trainer.optimizers, 1, gamma=1.0
    )

    def step(_):
        trainer.optimizers.step()
        return 0.0

    trainer.all_reduce_states = Mock(side_effect=step)
    comm = SimpleNamespace(
        wait_comm_ready=Mock(),
        world_size=lambda: 1,
        allreduce=Mock(),
    )
    episodes = [
        SimpleNamespace(
            values=values,
            chains=torch.zeros(3, 1),
            task_id=0,
            trial_id=0,
            weight_version=0,
            advantage=1.0,
            finish_step=len(values),
        )
        for values in ([[], []] if empty else [[1.0, 2.0, 3.0], [6.0]])
    ]
    trainer.step_training(episodes, 1, 2, 0, comm, False)
    assert trainer.model.weight.item() == pytest.approx(0.0 if empty else expected)
    assert trainer.all_reduce_states.call_count == (0 if empty else 1)
    assert trainer.lr_schedulers.last_epoch == (0 if empty else 1)
    if weighting is None:
        # Default execution must not pay for the opt-in count/collation pass.
        assert trainer.data_packer.policy_collate_fn.call_count == len(episodes)
        comm.wait_comm_ready.assert_not_called()
        comm.allreduce.assert_not_called()


@pytest.mark.parametrize("chunk_size", [1, 2, 3])
@pytest.mark.parametrize(
    "weighting,expected", [(None, 0.39), ("episode", 0.4), ("sample", 0.3)]
)
def test_partial_final_chunk_keeps_legacy_default(
    monkeypatch, chunk_size, weighting, expected
):
    # First episode: component mean (1+1+2+2+3)/5 = 1.8, not chunk mean 2.
    # Second episode: mean 6. Legacy gradient = (1.8+6)/2 = 3.9.
    test_production_objective_is_chunk_invariant(
        monkeypatch,
        OpenVLAGRPOTrainer,
        weighting,
        expected,
        chunk_size,
        False,
        partial=True,
    )


def test_new_objective_requires_explicit_configuration():
    from cosmos_rl.policy.config import VLAConfig

    assert VLAConfig().objective_weighting is None
    assert VLAConfig(objective_weighting="sample").objective_weighting == "sample"
    assert VLAConfig(objective_weighting="episode").objective_weighting == "episode"
