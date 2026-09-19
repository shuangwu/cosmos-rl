"""Regression coverage for synchronous simulator engine cleanup."""

import threading
from types import SimpleNamespace
from unittest.mock import Mock


def test_vla_shutdown_is_safe_before_initialization_and_idempotent():
    from cosmos_rl.rollout.vla_rollout import OpenVLARollout

    rollout = object.__new__(OpenVLARollout)
    rollout.shutdown()
    manager = Mock()
    rollout.env_manager = manager
    rollout.shutdown()
    rollout.shutdown()
    manager.stop_simulator.assert_called_once_with(preserve_state=False)


def test_sync_worker_shuts_down_engine_before_unregister():
    from cosmos_rl.rollout.worker.rollout_control import (
        DisaggregatedRolloutControlWorker,
    )

    events = []
    engine = Mock()
    engine.shutdown.side_effect = lambda: events.append("engine")
    worker = SimpleNamespace(
        shutdown_signal=threading.Event(),
        shutdown_mp_signal=threading.Event(),
        replica_name="test",
        background_thread=None,
        teacher_interact_thread=None,
        scheduler=None,
        heartbeat_thread=None,
        rollout=engine,
        unregister_from_controller=lambda: events.append("unregister"),
    )
    DisaggregatedRolloutControlWorker.handle_shutdown(worker)
    DisaggregatedRolloutControlWorker.handle_shutdown(worker)
    assert events == ["engine", "unregister"]


def test_async_scheduler_retains_engine_shutdown_ownership():
    from cosmos_rl.rollout.worker.rollout_control import (
        DisaggregatedRolloutControlWorker,
    )

    scheduler, engine = Mock(), Mock()
    worker = SimpleNamespace(
        shutdown_signal=threading.Event(),
        shutdown_mp_signal=threading.Event(),
        replica_name="test",
        background_thread=None,
        teacher_interact_thread=None,
        scheduler=scheduler,
        heartbeat_thread=None,
        rollout=engine,
        unregister_from_controller=Mock(),
    )
    DisaggregatedRolloutControlWorker.handle_shutdown(worker)
    scheduler.stop.assert_called_once_with(wait=False)
    engine.shutdown.assert_not_called()
