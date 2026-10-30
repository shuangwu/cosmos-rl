# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Application readers and diagnostics never own native admission accounting."""

from queue import Queue
from types import SimpleNamespace
from unittest.mock import Mock
import weakref

import pytest
import requests
import torch

from cosmos_rl.dispatcher.algo.grpo import GRPO
from cosmos_rl.dispatcher.data.admission import CompletionFailure, CompletionIdentity
from cosmos_rl.dispatcher.data.admission_state import CompletionAdmissionState
from cosmos_rl.dispatcher.data.diagnostics import (
    CompletionDiagnostics,
    detached_metadata,
)
from cosmos_rl.dispatcher.data.packer import BaseDataPacker
from cosmos_rl.dispatcher.data.schema import RLPayload
from cosmos_rl.dispatcher.protocol import RolloutRequest
from cosmos_rl.dispatcher.status import PolicyStatusManager
from cosmos_rl.reward.callbacks import reward_read_scope
from cosmos_rl.reward.dispatcher import RewardDispatcher
from cosmos_rl.reward.identity import CompletionReporter
from cosmos_rl.reward.local_calculator import LocalRewardCalculator
from cosmos_rl.reward.remote_calculator import RemoteRewardCalculator


class Algo(GRPO):
    def __init__(self):
        super().__init__(reward_fn=None)
        self.compute_reward = Mock(
            return_value=([1.0, 100.0, 3.0], [1.0, 100.0, 3.0], [{}, {}, {}])
        )
        self.compute_advantage = Mock(wraps=super().compute_advantage)


class View:
    pass


class Packer(BaseDataPacker):
    """Reward views and policy views intentionally have different owners."""

    def __init__(self):
        super().__init__()
        self.reward_views = {}
        self.notifications = []
        self.results = []

    def get_rollout_input(self, item):
        return item

    def get_policy_input(self, sample, rollout_output, n_ignore_prefix_tokens=0):
        return rollout_output

    def policy_compute_max_len(self, processed_samples):
        return len(processed_samples)

    def policy_collate_fn(self, processed_samples, computed_max_len):
        return {"samples": processed_samples}

    def annotate_reward_result(self, payload, result):
        self.results.append(result)
        return [
            {"slot": index, "details": ["diagnostic"]}
            for index in range(len(result.rewards))
        ]

    def on_reward_read_complete(self, payload, *, is_validation, error):
        self.reward_views.pop(payload.prompt_idx, None)
        self.notifications.append((payload.prompt_idx, is_validation, error))


def payload(mask=None):
    return RLPayload(
        prompt="p",
        prompt_idx=5,
        reference_answer="a",
        completions=[View(), View(), View()],
        completion_trainable=mask,
        completion_drop_reasons=[None, "quality", None],
    )


def local(packer):
    calculator = LocalRewardCalculator()
    calculator.data_packer = calculator.val_data_packer = packer
    calculator.rl_algo = calculator.val_rl_algo = Algo()
    calculator.config = SimpleNamespace(
        train=SimpleNamespace(
            non_text=True, train_policy=SimpleNamespace(min_filter_prefix_tokens=0)
        )
    )
    return calculator


@pytest.mark.parametrize("mode", ["colocated", "disaggregated"])
@pytest.mark.parametrize(
    "mask", [[True, False, True], [False] * 3, [True, False, False]]
)
def test_custom_packer_releases_only_reward_views_and_retains_original_slots(
    mode, mask
):
    packer = Packer()
    calculator = local(packer)
    calculator.config.mode = mode
    source = payload(mask)
    packer.reward_views[source.prompt_idx] = View()
    reward_view = weakref.ref(packer.reward_views[source.prompt_idx])
    policy_views = [weakref.ref(value) for value in source.completions]
    reporter = CompletionReporter("source", 0)
    reporter.reserve([source], 3, 7)
    output, _, _ = calculator.compute_rewards([source], False, 7)
    selected = [0, 2] if mask == [True, False, True] else []
    assert reward_view() is None
    assert all(view() is not None for view in policy_views)
    assert calculator.rl_algo.compute_reward.call_count == 1
    assert calculator.rl_algo.compute_advantage.call_count == int(bool(selected))
    assert packer.results[0].selected_indices == tuple(selected)
    assert packer.results[0].rewards == (1.0, 100.0, 3.0)
    assert [
        item.original_index for item in output[0].completion_diagnostics
    ] == selected
    assert [
        item.original_index for item in output[0].rejected_completion_diagnostics
    ] == [index for index in range(3) if index not in selected]
    report = reporter.report(output, packer)
    report = RolloutRequest.model_validate(report.model_dump())
    assert [
        item.original_index for item in report.completion_diagnostics or []
    ] == selected
    assert {item.diagnostics.original_index for item in report.completion_failures} == (
        set(range(3)) - set(selected)
    )
    # An application can release the reward boundary repeatedly while a later
    # colocated policy reader still uses the same generated object.
    packer.on_reward_read_complete(source, is_validation=False, error=None)
    if selected:
        view = packer.get_policy_input(None, output[0].completions[0])
        assert view is policy_views[0]()
        # Drop producer/report aliases. The policy reader alone may retain the
        # completion after reward cleanup, then release it at its own boundary.
        del report
        source.completions = None
        calculator.rl_algo.compute_reward.reset_mock()
        output[0].completions.clear()
        assert policy_views[0]() is view
        del view
        assert policy_views[0]() is None
    assert len(packer.notifications) == 2


@pytest.mark.parametrize("failure", ["reward", "advantage", "annotation", "cleanup"])
def test_reward_annotation_and_cleanup_failures_preserve_original_exception(failure):
    packer = Packer()
    calculator = local(packer)
    original = RuntimeError(f"{failure} failed")
    cleanup = packer.on_reward_read_complete
    if failure == "reward":
        calculator.rl_algo.compute_reward.side_effect = original
    elif failure == "advantage":
        calculator.rl_algo.compute_advantage.side_effect = original
    elif failure == "annotation":
        packer.annotate_reward_result = Mock(side_effect=original)

    def finish(*args, **kwargs):
        cleanup(*args, **kwargs)
        if failure != "cleanup":
            raise ValueError("secondary cleanup failure")
        raise original

    packer.on_reward_read_complete = finish
    first, second = payload(), payload()
    second.prompt_idx = 6
    with pytest.raises(RuntimeError) as error:
        calculator.compute_rewards([first, second], False, 7)
    assert error.value is original
    assert [item[0] for item in packer.notifications] == [5, 6]
    assert all(
        item[2] is (None if failure == "cleanup" else original)
        for item in packer.notifications
    )


@pytest.mark.parametrize("validation", [False, True])
def test_bypass_reward_runs_annotations_and_final_reader_cleanup(validation):
    packer = Packer()
    dispatcher = RewardDispatcher()
    dispatcher.data_packer = dispatcher.val_data_packer = packer
    dispatcher.is_remote = False
    dispatcher.minimum_trainable_completions = 2
    source = payload([False] * 3)
    dispatcher.enqueue_rewards_cal([source], validation, 7, bypass_reward=True)
    output, stage, _, _ = dispatcher.dequeue_rewards_cal()
    assert stage is validation
    assert packer.results[0].selected_indices == ((0, 1, 2) if validation else ())
    assert packer.notifications == [(5, validation, None)]
    assert len(output[0].completions) == (3 if validation else 0)


def remote(packer, source):
    calculator = RemoteRewardCalculator()
    calculator.data_packer = calculator.val_data_packer = packer
    calculator.minimum_trainable_completions = 2
    calculator.rl_algo = Algo()
    calculator.uuid2payload = {"task": [source]}
    calculator.uuid2replica = {"task": None}
    calculator.uuid2stage = {"task": "training"}
    calculator.uuid2step = {"task": 7}
    calculator.uuid2completions_per_payload = {"task": [3]}
    calculator.fetch_reward = Mock(return_value=torch.tensor([1.0, 100.0, 3.0]))
    pending = Queue()
    pending.put("task")
    return calculator, pending


@pytest.mark.parametrize("failure", [None, "annotation", "advantage", "enqueue"])
def test_remote_reward_callbacks_and_terminal_failure_cleanup(failure):
    packer = Packer()
    source = payload([True, False, True])
    calculator, pending = remote(packer, source)
    original = RuntimeError("remote callback fault")
    if failure == "annotation":
        packer.annotate_reward_result = Mock(side_effect=original)
    elif failure == "advantage":
        calculator.rl_algo.compute_advantage.side_effect = original
    elif failure == "enqueue":
        calculator._enqueue_rewards = Mock(side_effect=original)
    if failure:
        with pytest.raises(RuntimeError) as error:
            (
                calculator.compute_rewards([source], False, 7)
                if failure == "enqueue"
                else calculator.get_results(pending)
            )
        assert error.value is original
        assert packer.notifications == [(5, False, original)]
    else:
        output, _, _ = calculator.get_results(pending)
        assert [item.original_index for item in output[0].completion_diagnostics] == [
            0,
            2,
        ]
        assert packer.notifications == [(5, False, None)]


def test_retryable_remote_fetch_keeps_readers_until_success():
    packer = Packer()
    calculator, pending = remote(packer, payload())
    calculator.fetch_reward.side_effect = requests.ConnectionError(
        "transient server fault"
    )
    assert calculator.get_results(pending) == (None, False, -1)
    assert not packer.notifications and not pending.empty()
    calculator.fetch_reward.side_effect = None
    calculator.get_results(pending)
    assert packer.notifications == [(5, False, None)]


def test_validation_keeps_every_original_slot_and_runs_its_own_packer():
    training, validation = Packer(), Packer()
    calculator = local(training)
    calculator.val_data_packer = validation
    source = payload([False])  # Stale training admission metadata is ignored.
    result, stage, _ = calculator.compute_validation_rewards([source], 7)
    assert stage and not training.notifications
    assert validation.notifications == [(5, True, None)]
    assert validation.results[0].selected_indices == (0, 1, 2)
    assert len(result[0].completion_diagnostics) == 3


@pytest.mark.parametrize(
    "bad", [torch.zeros(1), float("nan"), {"nested": View()}, [1], {"huge": "x" * 4097}]
)
def test_diagnostics_reject_owned_or_oversized_objects(bad):
    with pytest.raises(ValueError):
        detached_metadata(bad)


def test_diagnostics_are_detached_before_application_mutates_its_reports():
    original = {"details": ["safe"]}
    diagnostic = CompletionDiagnostics(original_index=2, metadata=original)
    original["details"].append("later")
    assert diagnostic.metadata == {"details": ["safe"]}


def test_multibyte_group_diagnostic_limit_counts_encoded_bytes():
    from cosmos_rl.reward.admission import resolve_completion_admission
    from cosmos_rl.reward.callbacks import annotate_reward_result

    source = RLPayload(completions=["completion"] * 100)
    # Each metadata entry is under its 4096-byte limit even when ASCII escaped.
    # Its group envelope is below 65536 characters but above 65536 UTF-8 bytes.
    packer = SimpleNamespace(
        annotate_reward_result=lambda *_: [{"details": "é" * 400} for _ in range(100)]
    )
    with pytest.raises(ValueError, match="65536 bytes"):
        annotate_reward_result(
            packer,
            source,
            resolve_completion_admission(source, 2),
            [0.0] * 100,
            [0.0] * 100,
            [0.0] * 100,
            [{}] * 100,
            is_validation=False,
        )


def test_native_settlement_observer_is_replay_suppressed_after_accounting():
    status = PolicyStatusManager()
    status.current_step = 7
    status.samples_on_the_fly = 3
    status.rollout_admission_closed = lambda: False
    status._publish_payload_transport_cleanup = Mock()
    state = CompletionAdmissionState()
    controller = SimpleNamespace(
        policy_status_manager=status,
        rollout_status_manager={
            "source": SimpleNamespace(
                n_atoms_per_replica=lambda: 1, status=SimpleNamespace(ended=False)
            )
        },
    )
    observed = []

    def observe(events):
        assert status.samples_on_the_fly == 0
        assert len(state.sources[("source", 0)][1].versions) == 3
        observed.extend(events)

    status.hook_fns = {"completion_settled": observe}
    request = RolloutRequest(
        src_replica_name="source",
        src_global_rank=0,
        payloads=[],
        completion_identities=[],
        completion_failures=[
            CompletionFailure(
                identity=CompletionIdentity(sequence=index, weight_version=7),
                reason="quality",
                diagnostics=CompletionDiagnostics(
                    original_index=index, metadata={"slot": index}
                ),
            )
            for index in [2, 0, 1]
        ],
    )
    for _ in range(2):
        state.settle(controller, request, state.prepare(controller, request, []))
    assert [event.identity.sequence for event in observed] == [2, 0, 1]
    assert [event.diagnostics.original_index for event in observed] == [2, 0, 1]
    assert all(event.disposition.outcome == "rejected" for event in observed)
    assert status.filter_records["application_rejected"] == 3


def test_cleanup_without_a_primary_error_still_finishes_the_entire_batch():
    packer = Packer()
    packer.on_reward_read_complete = Mock(side_effect=RuntimeError("cleanup"))
    with pytest.raises(RuntimeError, match="cleanup"):
        with reward_read_scope(packer, [payload(), payload()], is_validation=False):
            pass
    assert packer.on_reward_read_complete.call_count == 2


@pytest.mark.parametrize("failure", ["fetch", "shape", "split"])
def test_terminal_remote_response_failure_releases_current_and_other_pending_readers(
    failure,
):
    packer = Packer()
    calculator, pending = remote(packer, payload())
    second = payload()
    second.prompt_idx = 6
    calculator.uuid2payload["later"] = [second]
    calculator.uuid2stage["later"] = "validation"
    original = ValueError("malformed remote response")
    if failure == "fetch":
        calculator.fetch_reward.side_effect = original
    elif failure == "shape":
        calculator.fetch_reward.return_value = torch.zeros(2)
    else:
        calculator.uuid2completions_per_payload["task"] = [1, 2]
    with pytest.raises(ValueError) as error:
        calculator.get_results(pending)
    if failure == "fetch":
        assert error.value is original
    assert [(index, validation) for index, validation, _ in packer.notifications] == [
        (5, False),
        (6, True),
    ]
    assert all(notification[2] is error.value for notification in packer.notifications)
    assert calculator.uuid2payload == {}
    calculator.close()
    assert len(packer.notifications) == 2


def test_remote_dispatcher_shutdown_finishes_pending_readers_once():
    from concurrent.futures import CancelledError

    packer = Packer()
    calculator, _ = remote(packer, payload())
    dispatcher = RewardDispatcher()
    dispatcher.is_remote = True
    dispatcher.reward_calculator = calculator
    dispatcher.shutdown(timeout_s=2)
    dispatcher.shutdown(timeout_s=2)
    assert len(packer.notifications) == 1
    assert isinstance(packer.notifications[0][2], CancelledError)
    assert calculator.uuid2payload == {}


@pytest.mark.parametrize("remote_reward", [False, True])
def test_cancellation_cleanup_failure_fails_shutdown_and_retains_owner(remote_reward):
    from concurrent.futures import Future

    packer = Packer()
    packer.on_reward_read_complete = Mock(side_effect=RuntimeError("cleanup unsafe"))
    dispatcher = RewardDispatcher()
    dispatcher.is_remote = remote_reward
    dispatcher.data_packer = packer
    if remote_reward:
        calculator, _ = remote(packer, payload())
        dispatcher.reward_calculator = calculator
    else:
        future = Future()
        dispatcher.executor = Mock(submit=Mock(return_value=future))
        dispatcher.enqueue_rewards_cal([payload()], False, 7)
        future.cancel()
    with pytest.raises(RuntimeError, match="shutdown failed"):
        dispatcher.shutdown(timeout_s=2)
    if not remote_reward:
        assert dispatcher.executor is not None
    with pytest.raises(RuntimeError, match="shutdown failed"):
        dispatcher.shutdown(timeout_s=2)
    packer.on_reward_read_complete.assert_called_once()


def test_queued_local_cancellation_notifies_submitter_once():
    from concurrent.futures import CancelledError, Future

    packer = Packer()
    dispatcher = RewardDispatcher(payload_per_task=1)
    dispatcher.is_remote = False
    dispatcher.data_packer = packer
    futures = [Future(), Future()]
    dispatcher.executor = Mock(submit=Mock(side_effect=futures))
    first, second = payload(), payload()
    second.prompt_idx = 6
    dispatcher.enqueue_rewards_cal([first, second], False, 7)
    for future in futures:
        assert future.cancel()
        assert future.cancel()  # Future callbacks remain single-shot.
    assert [item[0] for item in packer.notifications] == [5, 6]
    assert all(isinstance(item[2], CancelledError) for item in packer.notifications)


def test_local_submission_failure_finishes_every_not_submitted_group():
    packer = Packer()
    dispatcher = RewardDispatcher(payload_per_task=1)
    dispatcher.is_remote = False
    dispatcher.data_packer = packer
    original = RuntimeError("executor closed")
    dispatcher.executor = Mock(submit=Mock(side_effect=original))
    with pytest.raises(RuntimeError) as error:
        dispatcher.enqueue_rewards_cal([payload(), payload()], False, 7)
    assert error.value is original
    assert len(packer.notifications) == 2
    assert all(item[2] is original for item in packer.notifications)


def test_remote_shutdown_waits_for_active_application_reader():
    import threading

    packer = Packer()
    calculator, pending = remote(packer, payload())
    entered, release = threading.Event(), threading.Event()
    annotate = packer.annotate_reward_result

    def read(*args):
        entered.set()
        assert release.wait(2)
        return annotate(*args)

    packer.annotate_reward_result = read
    reader = threading.Thread(target=calculator.get_results, args=(pending,))
    reader.start()
    dispatcher = RewardDispatcher()
    dispatcher.is_remote = True
    dispatcher.reward_calculator = calculator
    try:
        assert entered.wait(1)
        with pytest.raises(TimeoutError):
            dispatcher.shutdown(timeout_s=0.01)
        assert packer.notifications == []
    finally:
        release.set()
        reader.join(2)
    dispatcher.shutdown(timeout_s=2)
    assert not reader.is_alive()
    assert packer.notifications == [(5, False, None)]


@pytest.mark.parametrize("observer_failure", [False, True])
def test_accepted_and_rejected_diagnostics_follow_identities_and_observer_failure_is_terminal(
    observer_failure,
):
    from cosmos_rl.dispatcher.data.schema import Rollout

    status = PolicyStatusManager()
    status.current_step = 7
    status.samples_on_the_fly = 3
    status.rollout_admission_closed = lambda: False
    status._publish_payload_transport_cleanup = Mock()
    state = CompletionAdmissionState()
    controller = SimpleNamespace(
        policy_status_manager=status,
        rollout_status_manager={
            "source": SimpleNamespace(
                n_atoms_per_replica=lambda: 1, status=SimpleNamespace(ended=False)
            )
        },
    )
    observer = Mock(
        side_effect=RuntimeError("required observer failed")
        if observer_failure
        else None
    )
    status.hook_fns = {"completion_settled": observer}
    request = RolloutRequest(
        src_replica_name="source",
        src_global_rank=0,
        payloads=[],
        completion_identities=[
            CompletionIdentity(sequence=index, weight_version=7) for index in [2, 0]
        ],
        completion_diagnostics=[
            CompletionDiagnostics(original_index=index, metadata={"index": index})
            for index in [2, 0]
        ],
        completion_failures=[
            CompletionFailure(
                identity=CompletionIdentity(sequence=1, weight_version=7),
                reason="quality",
                diagnostics=CompletionDiagnostics(
                    original_index=1, metadata={"index": 1}
                ),
            )
        ],
    )
    rollouts = [Rollout(completion="accepted", weight_version=7)] * 2
    plan = state.prepare(controller, request, rollouts)
    if observer_failure:
        with pytest.raises(RuntimeError, match="required observer failed"):
            state.settle(controller, request, plan)
        assert state.failed
        with pytest.raises(RuntimeError, match="restart required"):
            state.prepare(controller, request, rollouts)
    else:
        assert state.settle(controller, request, plan) == rollouts
        assert (
            state.settle(
                controller, request, state.prepare(controller, request, rollouts)
            )
            == []
        )
    observer.assert_called_once()
    events = observer.call_args.args[0]
    assert [event.identity.sequence for event in events] == [2, 0, 1]
    assert [event.diagnostics.original_index for event in events] == [2, 0, 1]
    assert [event.disposition.outcome for event in events] == [
        "accepted",
        "accepted",
        "rejected",
    ]
    assert status.samples_on_the_fly == 2
