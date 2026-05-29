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

import os

os.environ["TORCH_CPP_LOG_LEVEL"] = "ERROR"
import unittest
import subprocess
import sys
import time
import toml
import tempfile

from cosmos_rl.utils import network_util


def _wait_all_or_fail(testcase, processes, timeout_s, context):
    """Wait for every process, but bound the wait so a hung worker fails
    fast instead of riding the CI job's multi-hour wall clock.

    Returns once all processes exit with code 0.  On the first process
    that does not finish within ``timeout_s``, kills the whole process
    group and fails the test with ``context`` so the regression is
    attributed to the right place rather than surfacing as an opaque
    ``timeout 2h`` (exit 124) kill of the entire suite.
    """
    deadline = time.monotonic() + timeout_s
    try:
        for process in processes:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(cmd=context, timeout=timeout_s)
            process.communicate(timeout=remaining)
            testcase.assertEqual(
                process.returncode,
                0,
                f"Process failed with code: {process.returncode} ({context})",
            )
    except subprocess.TimeoutExpired:
        for process in processes:
            if process.poll() is None:
                process.kill()
        for process in processes:
            try:
                process.communicate(timeout=30)
            except Exception:
                pass
        testcase.fail(
            f"Timed out after {timeout_s:.0f}s waiting for processes to exit "
            f"cleanly ({context}).  This usually means a multi-rank rollout "
            f"worker did not shut down at genuine end-of-data: either a rank "
            f"left main_loop early and stranded a peer in a cross-rank "
            f"collective, or the controller kept issuing P2R/R2R to an "
            f"already-ended rollout.  See the Option-C drain vote "
            f"(multirank_synchronous_should_self_terminate) in "
            f"rollout_control.py and the status.ended exclusion in "
            f"status.py::trigger_weight_sync (rollout_multirank_shutdown.md)."
        )


class TestProcessFlow(unittest.TestCase):
    def test_process_exit_grpo(self):
        """Test grpo all processes exit cleanly."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        world_size = 2
        port = network_util.find_available_port(8123)
        config_path = os.path.join(
            cur_dir,
            "configs",
            "test_simple_grpo.toml",
        )
        with open(config_path, "r") as f:
            config = toml.load(f)
        config["train"]["epoch"] = 1
        config["train"]["train_policy"]["dataset"]["name"] = os.path.join(
            cur_dir, "data_fixtures", "test_dataset"
        )
        config["train"]["train_policy"]["allowed_outdated_steps"] = 100
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".toml", delete=False
        ) as tmpfile:
            toml.dump(config, tmpfile)
            tmpfile_toml = tmpfile.name
        controller_cmd = f"{sys.executable} -m cosmos_rl.dispatcher.run_web_panel --config {tmpfile_toml}"
        controller_cmd += f" --port {port}"
        env_dict = os.environ.copy()
        env_dict["COSMOS_ROLE"] = "Controller"
        controller_process = subprocess.Popen(
            controller_cmd,
            shell=True,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=env_dict,
        )
        os.environ["COSMOS_CONTROLLER_HOST"] = f"localhost:{port}"
        # Create the Python command for torchrun
        policy_cmd = [
            "torchrun",
            f"--nproc_per_node={world_size}",  # Use 2 GPUs
            "--role=rank",
            "--tee=3",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            os.path.join(cur_dir, "launch_test_worker.py"),
            "--shm_name",
            "-1",
            "--shm_size",
            "-1",
            "--mode",
            "dummy_policy",
        ]
        rollout_cmd = [
            "torchrun",
            f"--nproc_per_node={world_size}",  # Use 2 GPUs
            "--role=rank",
            "--tee=3",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            os.path.join(cur_dir, "launch_test_worker.py"),
            "--shm_name",
            "-1",
            "--shm_size",
            "-1",
            "--mode",
            "dummy_rollout",
        ]
        policy_env = dict(os.environ)
        policy_env["CUDA_VISIBLE_DEVICES"] = "0,1"
        # Start the process
        policy_process = subprocess.Popen(
            policy_cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=policy_env,
        )
        rollout_env = dict(os.environ)
        rollout_env["CUDA_VISIBLE_DEVICES"] = "2,3"
        rollout_process = subprocess.Popen(
            rollout_cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=rollout_env,
        )

        processes = [controller_process, policy_process, rollout_process]

        # Time-bounded wait: this GRPO config (tp=2 -> rollout world_size=2,
        # high allowed_outdated_steps) is exactly the end-of-data shutdown
        # path; an unbounded communicate() here rode the 2h CI wall when the
        # multi-rank shutdown deadlocked.  Bound it so a regression fails
        # fast and attributed.
        _wait_all_or_fail(
            self,
            processes,
            timeout_s=1500,
            context="grpo end-of-data shutdown (tp=2)",
        )

    def _run_multirank_end_of_data(self, rollout_tp_size, context):
        """Drive a multi-rank rollout worker to genuine end-of-data and
        assert every process exits cleanly within a bounded wait.

        Shared body for the ``tp`` (dp==1) and ``dp`` (dp>1) variants
        below.  ``rollout_tp_size`` selects the rollout DP layout for the
        2-GPU rollout worker: ``2`` -> pure TP (all ranks drain on the
        same iteration); ``1`` -> dp_shard=2, so the final round-robin
        batch leaves an *uneven* tail and ranks reach ``prompt_consume_end``
        on different iterations -- the case the Option-C per-iteration
        drain vote must handle without stranding a peer.
        """
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        world_size = 2
        port = network_util.find_available_port(8123)
        config_path = os.path.join(cur_dir, "configs", "test_simple_grpo.toml")
        with open(config_path, "r") as f:
            config = toml.load(f)
        config["train"]["epoch"] = 1
        # Small batch so the last step of the single epoch leaves an
        # uneven prompt tail across the worker's ranks.
        config["train"]["train_batch_per_replica"] = 3
        config["train"]["train_policy"]["dataset"]["name"] = os.path.join(
            cur_dir, "data_fixtures", "test_dataset"
        )
        config["train"]["train_policy"]["allowed_outdated_steps"] = 100
        config["rollout"]["parallelism"]["tp_size"] = rollout_tp_size
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".toml", delete=False
        ) as tmpfile:
            toml.dump(config, tmpfile)
            tmpfile_toml = tmpfile.name
        controller_cmd = f"{sys.executable} -m cosmos_rl.dispatcher.run_web_panel --config {tmpfile_toml}"
        controller_cmd += f" --port {port}"
        env_dict = os.environ.copy()
        env_dict["COSMOS_ROLE"] = "Controller"
        controller_process = subprocess.Popen(
            controller_cmd,
            shell=True,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=env_dict,
        )
        os.environ["COSMOS_CONTROLLER_HOST"] = f"localhost:{port}"
        worker_args = [
            "torchrun",
            f"--nproc_per_node={world_size}",
            "--role=rank",
            "--tee=3",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            os.path.join(cur_dir, "launch_test_worker.py"),
            "--shm_name",
            "-1",
            "--shm_size",
            "-1",
            "--mode",
        ]
        policy_cmd = worker_args + ["dummy_policy"]
        rollout_cmd = worker_args + ["dummy_rollout"]
        policy_env = dict(os.environ)
        policy_env["CUDA_VISIBLE_DEVICES"] = "0,1"
        policy_process = subprocess.Popen(
            policy_cmd, stdout=sys.stderr, stderr=sys.stderr, env=policy_env
        )
        rollout_env = dict(os.environ)
        rollout_env["CUDA_VISIBLE_DEVICES"] = "2,3"
        rollout_process = subprocess.Popen(
            rollout_cmd, stdout=sys.stderr, stderr=sys.stderr, env=rollout_env
        )

        processes = [controller_process, policy_process, rollout_process]
        # Generous bound (model load + a short single-epoch GRPO run) but
        # far below the 2h CI wall, so the consume-end deadlock surfaces
        # as a fast, attributed failure.
        _wait_all_or_fail(self, processes, timeout_s=1500, context=context)

    def test_process_exit_grpo_multirank_end_of_data(self):
        """Regression: a multi-rank rollout worker driven to genuine
        end-of-data must shut down without deadlocking.

        ``tp`` layout (rollout tp_size=2 -> world_size=2, dp==1): all
        ranks reach ``prompt_consume_end`` on the same iteration.  The
        Option-C drain vote self-terminates them in lockstep without
        depending on the controller's stop-carrying R2R broadcast (which
        races the end-of-data signal -- see rollout_multirank_shutdown.md).
        """
        self._run_multirank_end_of_data(
            rollout_tp_size=2,
            context="grpo multi-rank end-of-data shutdown (tp=2, dp=1)",
        )

    def test_process_exit_grpo_multirank_dp_end_of_data(self):
        """Regression (corner E): rollout tp_size=1 -> dp_shard=2, so the
        final round-robin batch is scattered *unevenly* and the two ranks
        reach ``prompt_consume_end`` on *different* iterations.

        This is the case a one-shot ``shutdown_signal.set()`` cannot
        handle (the first rank to drain would strand the other in the next
        collective); the per-iteration drain vote must keep voting until
        the lagging rank also drains, then exit together.
        """
        self._run_multirank_end_of_data(
            rollout_tp_size=1,
            context="grpo multi-rank end-of-data shutdown (tp=1, dp=2)",
        )

    def test_process_exit_sft(self):
        """Test sft all processes exit cleanly."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        world_size = 2
        port = network_util.find_available_port(8123)
        config_path = os.path.join(
            cur_dir,
            "configs",
            "test_simple_sft.toml",
        )
        with open(config_path, "r") as f:
            config = toml.load(f)
        config["train"]["epoch"] = 1
        config["train"]["train_policy"]["dataset"]["name"] = os.path.join(
            cur_dir, "data_fixtures", "test_dataset"
        )
        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".toml", delete=False
        ) as tmpfile:
            toml.dump(config, tmpfile)
            tmpfile_toml = tmpfile.name
        controller_cmd = f"{sys.executable} -m cosmos_rl.dispatcher.run_web_panel --config {tmpfile_toml}"
        controller_cmd += f" --port {port}"
        env_dict = os.environ.copy()
        env_dict["COSMOS_ROLE"] = "Controller"
        controller_process = subprocess.Popen(
            controller_cmd,
            shell=True,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=env_dict,
        )
        os.environ["COSMOS_CONTROLLER_HOST"] = f"localhost:{port}"
        # Create the Python command for torchrun
        policy_cmd = [
            "torchrun",
            f"--nproc_per_node={world_size}",  # Use 2 GPUs
            "--role=rank",
            "--tee=3",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            os.path.join(cur_dir, "launch_test_worker.py"),
            "--shm_name",
            "-1",
            "--shm_size",
            "-1",
            "--mode",
            "dummy_policy",
        ]
        policy_env = dict(os.environ)
        policy_env["CUDA_VISIBLE_DEVICES"] = "0,1"
        # Start the process
        policy_process = subprocess.Popen(
            policy_cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=policy_env,
        )
        processes = [controller_process, policy_process]

        # Time-bounded so an SFT end-of-data hang fails fast and attributed
        # instead of riding the CI job's multi-hour wall.
        _wait_all_or_fail(
            self,
            processes,
            timeout_s=1500,
            context="sft end-of-data shutdown",
        )


class TestValidationFlow(unittest.TestCase):
    def test_train_validation(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        world_size = 4
        # Create the Python command for torchrun
        cmd = [
            "torchrun",
            f"--nproc_per_node={world_size}",  # Use 4 GPUs
            "--role=rank",
            "--tee=3",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            os.path.join(cur_dir, "launch_test_worker.py"),
            "--mode",
            "sft_for_validation",
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=env,
        )
        processes = [process]

        # Wait for process to complete
        for process in processes:
            stdout, stderr = process.communicate()
            # Check if process completed successfully
            assert process.returncode == 0, (
                f"Process failed with code: {process.returncode}"
            )


class TestRewardFlow(unittest.TestCase):
    def test_check_reward(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        world_size = 2
        # Create the Python command for torchrun
        cmd = [
            "torchrun",
            f"--nproc_per_node={world_size}",  # Use 4 GPUs
            "--role=rank",
            "--tee=3",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            os.path.join(cur_dir, "launch_test_worker.py"),
            "--mode",
            "reward_execution_check",
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0,1"
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=env,
        )
        processes = [process]

        # Wait for process to complete
        for process in processes:
            stdout, stderr = process.communicate()
            # Check if process completed successfully
            assert process.returncode == 0, (
                f"Process failed with code: {process.returncode}"
            )


class TestSFTDDPLoadFlow(unittest.TestCase):
    def test_sft_ddp_load(self):
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        world_size = 4
        # Create the Python command for torchrun
        cmd = [
            "torchrun",
            f"--nproc_per_node={world_size}",  # Use 4 GPUs
            "--role=rank",
            "--tee=3",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            os.path.join(cur_dir, "launch_test_worker.py"),
            "--mode",
            "sft_ddp_load_check",
        ]
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
        # Start the process
        process = subprocess.Popen(
            cmd,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=env,
        )
        processes = [process]

        # Wait for process to complete
        for process in processes:
            stdout, stderr = process.communicate()
            # Check if process completed successfully
            assert process.returncode == 0, (
                f"Process failed with code: {process.returncode}"
            )


class TestMultiReplicaSFT(unittest.TestCase):
    def test_multi_replica_sft(self):
        """Test the multi-replica SFT process flow."""
        cur_dir = os.path.dirname(os.path.abspath(__file__))
        world_size = 2
        port = network_util.find_available_port(8123)
        config_path = os.path.join(
            cur_dir,
            "configs",
            "test_simple_sft.toml",
        )
        with open(config_path, "r") as f:
            config = toml.load(f)

        config["train"]["epoch"] = 16
        config["train"]["train_batch_per_replica"] = 4
        config["train"]["train_policy"]["dataset"]["name"] = os.path.join(
            cur_dir, "data_fixtures", "sharegpt52k_small"
        )
        config["policy"]["parallelism"]["tp_size"] = 1
        config["policy"]["parallelism"]["dp_shard_size"] = 2
        config["policy"]["parallelism"]["n_init_replicas"] = 4

        config["validation"]["batch_size"] = 2
        config["validation"]["dataset"]["name"] = os.path.join(
            cur_dir, "data_fixtures", "sharegpt52k_small"
        )

        with tempfile.NamedTemporaryFile(
            mode="w+", suffix=".toml", delete=False
        ) as tmpfile:
            toml.dump(config, tmpfile)
            tmpfile_toml = tmpfile.name
        controller_cmd = f"{sys.executable} -m cosmos_rl.dispatcher.run_web_panel --config {tmpfile_toml}"
        controller_cmd += f" --port {port}"
        env_dict = os.environ.copy()
        env_dict["COSMOS_ROLE"] = "Controller"
        controller_process = subprocess.Popen(
            controller_cmd,
            shell=True,
            stdout=sys.stderr,
            stderr=sys.stderr,
            env=env_dict,
        )
        os.environ["COSMOS_CONTROLLER_HOST"] = f"localhost:{port}"
        # Create the Python command for torchrun
        policy_cmd = [
            "torchrun",
            f"--nproc_per_node={world_size}",  # Use 2 GPUs
            "--role=rank",
            "--tee=3",
            "--rdzv_backend=c10d",
            "--rdzv_endpoint=localhost:0",
            os.path.join(cur_dir, "utils", "mock_policy_entrance.py"),
            "--test",
            "multi_replica_sft",
        ]
        rollout_processes = []
        for dev in ["0,1", "2,3", "4,5", "6,7"]:
            rollout_env = dict(os.environ)
            rollout_env["CUDA_VISIBLE_DEVICES"] = dev
            rollout_processes.append(
                subprocess.Popen(
                    policy_cmd,
                    stdout=sys.stderr,
                    stderr=sys.stderr,
                    env=rollout_env,
                )
            )

        processes = [controller_process] + rollout_processes

        # Wait for process to complete
        for process in processes:
            stdout, stderr = process.communicate()
            # Check if process completed successfully
            assert process.returncode == 0, (
                f"Process failed with code: {process.returncode}"
            )


if __name__ == "__main__":
    unittest.main()
