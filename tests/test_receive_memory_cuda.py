# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Standalone Cosmos acceptance tests: real Redis/NCCL and two CUDA devices.

These tests exercise the public consumer contract without NDAS. Explicit fault
injection uses real received CUDA storage; it does not claim a physical device OOM.
"""

import gc
import time
import uuid
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from cosmos_rl.utils.payload_transport.nccl import strategy as nccl
from cosmos_rl.utils.payload_transport.receive_memory import ReceiveMemoryError
from cosmos_rl.utils.trajectory import build_trajectory_schema, schema_layout

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.device_count() < 2,
    reason="requires two real CUDA devices and Redis/NCCL",
)

LENGTHS = (17, 23, 31)
OBS_DIM = 63
MAX_BYTES = schema_layout(
    build_trajectory_schema(dict(max_steps=max(LENGTHS), obs_dim=OBS_DIM, action_dim=3))
)[1]


def eventually(predicate, timeout=10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition did not become true before deadline")


def trajectory(seed, length, device):
    return {
        "observations": torch.arange(
            length * OBS_DIM, device=device, dtype=torch.float32
        ).reshape(length, OBS_DIM)
        / 1000
        + seed / 100,
        "actions": torch.arange(length * 3, device=device, dtype=torch.float32).reshape(
            length, 3
        )
        / 100,
        "rewards": torch.full((length,), 0.5, device=device),
        "episode_length": length - 2,
    }


@pytest.fixture
def endpoint():
    import redis
    import test_nccl_e2e as e2e
    from cosmos_rl.utils.payload_transport.nccl.mixins import NCCLRolloutMixin

    assert e2e._ensure_redis(), "Redis unavailable: refusing an inconclusive GPU test"
    endpoints = []

    def create(budget):
        token = uuid.uuid4().hex
        config = SimpleNamespace(
            logging=SimpleNamespace(experiment_name=f"bounded-acceptance-{token}"),
            custom=dict(
                nccl_receive_budget_bytes=budget, nccl_receive_admission_timeout=15
            ),
        )
        client = redis.Redis(
            host=e2e.REDIS_HOST, port=e2e.REDIS_PORT, decode_responses=True
        )
        producers = []
        packer = e2e._ConsumerPacker()
        endpoints.append((producers, packer))
        torch.cuda.set_device(0)
        for index, length in enumerate(LENGTHS):
            producer = NCCLRolloutMixin()
            producer.setup_nccl(
                replica_id=f"producer-{token}-{index}",
                rollout_idx=index,
                redis_client=client,
                config=config,
                device=torch.device("cuda:0"),
                sender_rank=0,
                max_steps=length,
                obs_dim=OBS_DIM,
                action_dim=3,
                registry_capacity=64,
            )
            producers.append(producer)
        torch.cuda.set_device(1)
        packer._nccl_dp_receiver_replica = f"consumer-{token}"
        packer._setup_nccl_data_packer(
            device=torch.device("cuda:1"),
            redis_client=client,
            config=config,
            prefetch_timeout=30,
            recv_timeout=5,
            first_transfer_timeout=30,
        )

        def batch(count=6, seed=0):
            torch.cuda.set_device(0)
            metas = []
            for i in range(count):
                which = i % len(producers)
                data = trajectory(seed + i, LENGTHS[which], "cuda:0")
                meta = producers[which].write_to_buffer(data)
                assert meta is not None
                metas.append(meta)
            torch.cuda.set_device(1)
            return metas

        return SimpleNamespace(
            packer=packer,
            strategy=packer._transport_strategy,
            producers=producers,
            batch=batch,
        )

    yield create
    for producers, packer in reversed(endpoints):
        torch.cuda.set_device(1)
        packer.shutdown_nccl_data_packer()
        # Tests must drop their aliases themselves. This is failure cleanup only.
        packer.release_prefetch()
        for producer in producers:
            producer.cleanup_nccl()
    torch.cuda.set_device(0)


def collect(env, refs):
    env.packer.start_prefetch(refs)
    env.packer.wait_prefetch()
    assert len(env.packer._prefetch_cache) == len(refs)


def train_step(packer, refs, seed, weight):
    reference_weight = weight.detach().clone().requires_grad_()
    received_losses, reference_losses = [], []
    for i, ref in enumerate(refs):
        payload = packer.get_policy_input(rollout_output=ref)
        assert packer.get_policy_input(rollout_output=ref) is payload
        expected = trajectory(seed + i, LENGTHS[i % len(LENGTHS)], "cuda:1")
        length = expected["episode_length"]
        for name in ("observations", "actions", "rewards"):
            torch.testing.assert_close(
                payload[name], expected[name][:length], rtol=0, atol=0
            )
        received_losses.append(
            (payload["observations"] @ weight - payload["actions"].sum(-1))
            .square()
            .mean()
        )
        reference_losses.append(
            (
                expected["observations"][:length] @ reference_weight
                - expected["actions"][:length].sum(-1)
            )
            .square()
            .mean()
        )
    loss = torch.stack(received_losses).mean()
    reference_loss = torch.stack(reference_losses).mean()
    loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(loss, reference_loss, rtol=0, atol=0)
    torch.testing.assert_close(weight.grad, reference_weight.grad, rtol=0, atol=0)
    with torch.no_grad():
        weight.add_(weight.grad, alpha=-0.0001)
    weight.grad = None
    return loss.item()


@pytest.mark.parametrize("bounded", [False, True])
def test_twenty_training_steps_match_direct_data_loss_and_gradients(endpoint, bounded):
    env = endpoint(8 * MAX_BYTES + 32 if bounded else 0)
    weight = torch.linspace(-0.01, 0.01, OBS_DIM, device="cuda:1", requires_grad=True)
    for step in range(20):
        refs = env.batch(count=6, seed=step * 10)
        collect(env, refs)
        train_step(env.packer, refs, step * 10, weight)
        env.packer.release_prefetch()
        if bounded:
            stats = env.strategy.receive_memory_stats()
            assert stats["reserved_bytes"] == stats["current_tensor_bytes"] == 0
            assert stats["peak_tensor_bytes"] <= stats["budget_bytes"]


def pending_reader(packer, refs):
    stream = torch.cuda.Stream(device=1)
    ready = torch.cuda.Event()
    ready.record()
    with torch.cuda.stream(stream):
        stream.wait_event(ready)
        # Deliberately keep a final reader pending while the next prefetch runs.
        torch.cuda._sleep(3_000_000_000)
        values = [
            packer.get_policy_input(rollout_output=ref)["observations"][::2].sum()
            for ref in refs
        ]
        done = torch.cuda.Event()
        done.record()
    return stream, done, values


def test_prefetch_finishes_while_prior_cuda_reader_is_pending(endpoint):
    env = endpoint(32 * MAX_BYTES + 64)
    # Warm all three producer pairs before making a timing-dependent assertion.
    collect(env, env.batch())
    env.packer.release_prefetch()
    current = env.batch(seed=10)
    following = env.batch(seed=20)
    collect(env, current)
    stream, done, values = pending_reader(env.packer, current)
    env.packer.start_prefetch(following)
    eventually(lambda: not env.packer._prefetch_result_queue.empty())
    assert not done.query(), "next prefetch did not complete during the delayed reader"
    env.packer.release_prefetch(streams=[stream])
    assert done.query()
    assert all(torch.isfinite(value).item() for value in values)
    env.packer.wait_prefetch()
    assert len(env.packer._prefetch_cache) == len(following)
    env.packer.release_prefetch()
    stats = env.strategy.receive_memory_stats()
    assert stats["reserved_bytes"] == stats["current_tensor_bytes"] == 0
    assert stats["admission_waits"] == 0


def test_slow_final_reader_backpressures_then_allows_progress(endpoint):
    env = endpoint(7 * MAX_BYTES + 32)
    current = env.batch(seed=10)
    following = env.batch(seed=20)
    collect(env, current)
    stream, done, values = pending_reader(env.packer, current)
    env.packer.start_prefetch(following)
    eventually(lambda: env.strategy.receive_memory_stats()["admission_waits"] == 1)
    assert env.packer._prefetch_result_queue.empty()
    assert not done.query()
    env.packer.release_prefetch(streams=[stream])
    assert done.query()
    assert all(torch.isfinite(value).item() for value in values)
    env.packer.wait_prefetch()
    env.packer.release_prefetch()
    stats = env.strategy.receive_memory_stats()
    assert stats["reserved_bytes"] == stats["current_tensor_bytes"] == 0
    assert stats["peak_tensor_bytes"] <= stats["budget_bytes"]


def test_shutdown_wakes_real_prefetch_admission_without_releasing_consumer(endpoint):
    env = endpoint(7 * MAX_BYTES + 32)
    collect(env, env.batch())
    leased = env.strategy.receive_memory_stats()["decoded_leased_bytes"]
    env.packer.start_prefetch(env.batch(seed=10))
    eventually(lambda: env.strategy.receive_memory_stats()["admission_waits"] == 1)
    start = time.monotonic()
    env.packer.shutdown_nccl_data_packer()
    assert time.monotonic() - start < 5
    assert env.packer._prefetch_thread is None
    assert env.strategy.receive_memory_stats()["reserved_bytes"] == leased
    env.packer.release_prefetch()
    assert env.strategy.receive_memory_stats()["reserved_bytes"] == 0


def test_oversized_payload_never_contacts_sender(endpoint):
    env = endpoint(1)
    with mock.patch.object(
        env.strategy, "_rendezvous_one", wraps=env.strategy._rendezvous_one
    ) as negotiate:
        env.packer.start_prefetch(env.batch(count=1))
        with pytest.raises(ReceiveMemoryError, match="exceeding"):
            env.packer.wait_prefetch()
        negotiate.assert_not_called()
    assert env.strategy.receive_memory_stats()["reserved_bytes"] == 0


def test_decode_fault_cleans_real_cuda_allocations(endpoint, monkeypatch):
    env = endpoint(32 * MAX_BYTES + 32)
    refs = env.batch()
    torch.cuda.synchronize(1)
    before = torch.cuda.memory_allocated(1)
    original = nccl._unpack
    count = 0

    def faulty_unpack(*args, **kwargs):
        nonlocal count
        payload = original(*args, **kwargs)
        count += 1
        if count == 2:
            raise torch.OutOfMemoryError("injected after real CUDA decode")
        return payload

    monkeypatch.setattr(nccl, "_unpack", faulty_unpack)
    env.packer.start_prefetch(refs)
    with pytest.raises(ReceiveMemoryError, match="injected after real CUDA decode"):
        env.packer.wait_prefetch()
    torch.cuda.synchronize(1)
    gc.collect()
    assert torch.cuda.memory_allocated(1) == before
    stats = env.strategy.receive_memory_stats()
    assert stats["reserved_bytes"] == stats["current_tensor_bytes"] == 0


def test_missing_payload_preserves_other_payloads_and_following_progress(endpoint):
    env = endpoint(32 * MAX_BYTES + 32)
    refs = env.batch(count=3)
    assert env.producers[1]._nccl_registry.free(refs[1]["_transfer_id"])
    env.packer.start_prefetch(refs)
    env.packer.wait_prefetch()
    assert set(env.packer._prefetch_cache) == {refs[i]["_transfer_id"] for i in (0, 2)}
    env.packer.release_prefetch()
    collect(env, env.batch(count=3, seed=10))
    env.packer.release_prefetch()
    assert env.strategy.receive_memory_stats()["reserved_bytes"] == 0


def test_dictionary_mutation_cannot_drop_storage_before_final_cuda_reader(endpoint):
    env = endpoint(32 * MAX_BYTES + 32)
    refs = env.batch()
    collect(env, refs)
    stream, done, values = pending_reader(env.packer, refs)
    allocated = torch.cuda.memory_allocated(1)
    for payload in env.packer._prefetch_cache.values():
        payload.clear()
    del payload
    gc.collect()
    assert not done.query()
    assert torch.cuda.memory_allocated(1) == allocated
    env.packer.release_prefetch(streams=[stream])
    assert done.query()
    assert all(torch.isfinite(value).item() for value in values)
    assert torch.cuda.memory_allocated(1) < allocated
    assert env.strategy.receive_memory_stats()["reserved_bytes"] == 0
