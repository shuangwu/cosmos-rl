# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU unit tests for the multi-rank end-of-data shutdown protocol.

Covers the corner cases enumerated in ``rollout_multirank_shutdown.md``:

  * Corner 1 (P2R-gates-the-stop) and corners 2/3 (stop never emitted /
    R2R ``in_mesh`` gate) -- neutralised on the controller side by
    excluding ``status.ended`` rollouts from ``trigger_weight_sync`` so
    no P2R recv / R2R broadcast is ever issued to a worker that is
    leaving ``main_loop``.
  * Corner 4 (lockstep invariant) and corner E (``dp>1`` uneven tail) --
    handled on the rollout side by the Option-C drain vote, whose pure
    decision (``multirank_synchronous_should_self_terminate``) only fires
    when *every* rank reports drained, so all ranks exit on the same
    iteration.

The true cross-rank lockstep itself is exercised by the GPU integration
test ``test_process_flow.py`` (bounded so a regression fails fast); these
tests pin the decision logic that test depends on, on CPU.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from cosmos_rl.rollout.worker.rollout_control import (
    DisaggregatedRolloutControlWorker,
    multirank_synchronous_should_self_terminate,
)
from cosmos_rl.dispatcher.status import PolicyStatusManager


# ---------------------------------------------------------------------------
# Option C -- pure self-terminate decision
# ---------------------------------------------------------------------------
class TestMultirankSelfTerminateDecision(unittest.TestCase):
    """``multirank_synchronous_should_self_terminate`` decision matrix."""

    def _decide(self, **overrides):
        kwargs = dict(
            world_size=2,
            is_async_rollout=False,
            validation_enabled=False,
            prompt_fetch_end=True,
            drain_vote_sum=2,
        )
        kwargs.update(overrides)
        return multirank_synchronous_should_self_terminate(**kwargs)

    def test_unanimous_vote_terminates(self):
        """All ranks drained (sum == world_size) -> self-terminate."""
        self.assertTrue(self._decide(world_size=2, drain_vote_sum=2))
        self.assertTrue(self._decide(world_size=4, drain_vote_sum=4))

    def test_partial_vote_does_not_terminate(self):
        """Corner E: dp>1 uneven tail -- only some ranks drained yet.

        The vote sum is below ``world_size`` while a peer is still
        generating its (longer) prompt-share tail; terminating now would
        strand that peer in the next collective.
        """
        self.assertFalse(self._decide(world_size=2, drain_vote_sum=1))
        self.assertFalse(self._decide(world_size=4, drain_vote_sum=3))
        self.assertFalse(self._decide(world_size=4, drain_vote_sum=0))

    def test_single_process_never_uses_vote(self):
        """world_size==1 keeps its direct shutdown_signal.set() fast path."""
        self.assertFalse(self._decide(world_size=1, drain_vote_sum=1))

    def test_async_rollout_excluded(self):
        """async-rollout path: prompt_fetch_end is not lockstep, so no vote."""
        self.assertFalse(self._decide(is_async_rollout=True, drain_vote_sum=2))

    def test_validation_enabled_excluded(self):
        """Validation on -> stay on the controller R2R shutdown (which runs
        the final validation); never self-terminate here."""
        self.assertFalse(self._decide(validation_enabled=True, drain_vote_sum=2))

    def test_before_fetch_end_no_vote(self):
        """Before the lockstep prompt_fetch_end signal there is no vote."""
        self.assertFalse(self._decide(prompt_fetch_end=False, drain_vote_sum=None))

    def test_no_vote_this_iteration(self):
        """drain_vote_sum is None when the vote did not run this iteration."""
        self.assertFalse(self._decide(drain_vote_sum=None))

    def test_overcount_is_not_unanimous(self):
        """Defensive: a vote sum above world_size is not treated as
        unanimous (would indicate an accounting bug, never a stop)."""
        self.assertFalse(self._decide(world_size=2, drain_vote_sum=3))


# ---------------------------------------------------------------------------
# Option C -- gate + collective helpers
# ---------------------------------------------------------------------------
class TestMultirankDrainVoteGate(unittest.TestCase):
    """``_multirank_drain_vote_enabled``: every clause must be identical
    across ranks (constants / config / the lockstep prompt_fetch_end), so
    that all ranks agree on whether to enter the collective vote."""

    @staticmethod
    def _worker(world_size, is_async, validation, fetch_end):
        return SimpleNamespace(
            parallel_dims=SimpleNamespace(world_size=world_size),
            _is_async_rollout=is_async,
            config=SimpleNamespace(validation=SimpleNamespace(enable=validation)),
            state=SimpleNamespace(prompt_fetch_end=lambda: fetch_end),
        )

    def _enabled(self, **kw):
        worker = self._worker(**kw)
        return DisaggregatedRolloutControlWorker._multirank_drain_vote_enabled(worker)

    def test_enabled_for_multirank_sync_no_validation_after_fetch_end(self):
        self.assertTrue(
            self._enabled(
                world_size=2, is_async=False, validation=False, fetch_end=True
            )
        )

    def test_disabled_single_process(self):
        self.assertFalse(
            self._enabled(
                world_size=1, is_async=False, validation=False, fetch_end=True
            )
        )

    def test_disabled_async_rollout(self):
        self.assertFalse(
            self._enabled(
                world_size=2, is_async=True, validation=False, fetch_end=True
            )
        )

    def test_disabled_when_validation_enabled(self):
        self.assertFalse(
            self._enabled(
                world_size=2, is_async=False, validation=True, fetch_end=True
            )
        )

    def test_disabled_before_fetch_end(self):
        self.assertFalse(
            self._enabled(
                world_size=2, is_async=False, validation=False, fetch_end=False
            )
        )


class TestMultirankDrainVoteSum(unittest.TestCase):
    """``_multirank_drain_vote_sum`` reduces the local drained flag."""

    @staticmethod
    def _worker(consume_end):
        return SimpleNamespace(
            state=SimpleNamespace(prompt_consume_end=lambda: consume_end),
        )

    def test_votes_local_drained_flag_and_returns_reduced_sum(self):
        # Simulate a 2-rank worker where the peer is also drained: the
        # all-reduce returns SUM == 2.  Assert the local contribution and
        # the returned int.
        import torch

        captured = {}

        def fake_all_reduce(tensor, op):
            captured["local"] = int(tensor.item())
            return torch.tensor([2], dtype=torch.int64)

        worker = self._worker(consume_end=True)
        with patch(
            "cosmos_rl.rollout.worker.rollout_control.dist_utils."
            "all_reduce_tensor_object_cpu",
            side_effect=fake_all_reduce,
        ):
            total = DisaggregatedRolloutControlWorker._multirank_drain_vote_sum(worker)
        self.assertEqual(captured["local"], 1, "drained rank votes 1")
        self.assertEqual(total, 2)

    def test_not_drained_votes_zero(self):
        import torch

        captured = {}

        def fake_all_reduce(tensor, op):
            captured["local"] = int(tensor.item())
            return torch.tensor([1], dtype=torch.int64)

        worker = self._worker(consume_end=False)
        with patch(
            "cosmos_rl.rollout.worker.rollout_control.dist_utils."
            "all_reduce_tensor_object_cpu",
            side_effect=fake_all_reduce,
        ):
            total = DisaggregatedRolloutControlWorker._multirank_drain_vote_sum(worker)
        self.assertEqual(captured["local"], 0, "non-drained rank votes 0")
        self.assertEqual(total, 1)


# ---------------------------------------------------------------------------
# Controller -- trigger_weight_sync excludes ended rollouts
# ---------------------------------------------------------------------------
def _replica(name, ended, start_time):
    return SimpleNamespace(
        name=name,
        start_time=start_time,
        status=SimpleNamespace(ended=ended),
    )


class _RolloutMgrStub:
    def __init__(self, replicas):
        self._replicas = replicas
        self.rollout_atoms_in_replica = 1

    def get_all_atoms_arrived_replicas(self):
        return list(self._replicas)


class TestTriggerWeightSyncExcludesEnded(unittest.TestCase):
    """Corners 1-3: the controller must not issue P2R/R2R to a rollout
    that has already POSTed ``is_end`` (and is self-terminating), unless
    validation is enabled (then the rollout stays for the final
    validation and must keep receiving the sync)."""

    def _run(self, replicas, validation_enabled):
        mgr = PolicyStatusManager()
        mgr.config = SimpleNamespace(
            validation=SimpleNamespace(enable=validation_enabled)
        )
        mgr.policy_atoms_in_replica = 1
        mgr.redis_handler = object()
        policy_replica = _replica("policy-0", ended=False, start_time=0)
        rollout_mgr = _RolloutMgrStub(replicas)

        with patch(
            "cosmos_rl.dispatcher.command.PolicyToRolloutUnicastCommand.trigger"
        ) as p2r, patch(
            "cosmos_rl.dispatcher.command.RolloutToRolloutBroadcastCommand.trigger"
        ) as r2r:
            mgr.trigger_weight_sync(
                policy_replica, rollout_mgr, current_step=10, total_steps=10
            )
        return p2r, r2r

    def test_single_ended_replica_no_sync(self):
        """1 replica, ended, validation off -> early return, no P2R/R2R.

        This is the failing-test topology (single rollout replica, intra-
        replica TP): once it ends, the controller stops syncing entirely
        and the rollout self-terminates via Option C.
        """
        p2r, r2r = self._run(
            [_replica("r0", ended=True, start_time=1)], validation_enabled=False
        )
        p2r.assert_not_called()
        r2r.assert_not_called()

    def test_mixed_excludes_only_ended(self):
        """Live + ended replicas, validation off -> sync targets only the
        live replica (ended dropped from P2R target and R2R recipients)."""
        live = _replica("r-live", ended=False, start_time=1)
        ended = _replica("r-ended", ended=True, start_time=0)
        p2r, r2r = self._run([live, ended], validation_enabled=False)
        p2r.assert_called_once()
        r2r.assert_called_once()
        # P2R unicast target is the (only) live replica.
        self.assertIs(p2r.call_args.kwargs["dst_replica"], live)
        # R2R recipient list excludes the ended replica.
        dst_replicas = r2r.call_args.kwargs["dst_replicas"]
        self.assertEqual(dst_replicas, [live])

    def test_ended_included_when_validation_enabled(self):
        """Validation on -> exclusion disabled; ended replica still synced
        (it stays alive to serve the final validation)."""
        ended = _replica("r-ended", ended=True, start_time=0)
        p2r, r2r = self._run([ended], validation_enabled=True)
        p2r.assert_called_once()
        r2r.assert_called_once()
        self.assertIs(p2r.call_args.kwargs["dst_replica"], ended)
        self.assertEqual(r2r.call_args.kwargs["dst_replicas"], [ended])

    def test_no_ended_unchanged_behavior(self):
        """No ended replicas, validation off -> all replicas synced
        (behaviour identical to before the exclusion)."""
        a = _replica("r-a", ended=False, start_time=0)
        b = _replica("r-b", ended=False, start_time=1)
        p2r, r2r = self._run([a, b], validation_enabled=False)
        p2r.assert_called_once()
        r2r.assert_called_once()
        self.assertIs(p2r.call_args.kwargs["dst_replica"], a)
        self.assertEqual(r2r.call_args.kwargs["dst_replicas"], [a, b])


if __name__ == "__main__":
    unittest.main()
