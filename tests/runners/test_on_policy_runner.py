# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the OnPolicyRunner."""

from __future__ import annotations

import copy
import math
import os
import socket
import tempfile
import torch
import torch.multiprocessing as mp
from tensordict import TensorDict

import pytest

from rsl_rl.env import VecEnv
from rsl_rl.runners import OnPolicyRunner
from tests.algorithms.test_ppo import _build_ppo

NUM_ENVS = 4
OBS_DIM = 8
NUM_ACTIONS = 4
MAX_EP_LEN = 50
IMG_C, IMG_H, IMG_W = 1, 16, 16


class DummyEnv(VecEnv):
    """Minimal VecEnv that returns random observations and rewards."""

    def __init__(self, device: str = "cpu", include_image: bool = False) -> None:  # noqa: D107
        self.num_envs = NUM_ENVS
        self.num_actions = NUM_ACTIONS
        self.max_episode_length = MAX_EP_LEN
        self.episode_length_buf = torch.zeros(NUM_ENVS, dtype=torch.long, device=device)
        self.device = device
        self.cfg = {}
        self._include_image = include_image

    def get_observations(self) -> TensorDict:  # noqa: D102
        data: dict = {"policy": torch.randn(self.num_envs, OBS_DIM, device=self.device)}
        if self._include_image:
            data["image"] = torch.randn(self.num_envs, IMG_C, IMG_H, IMG_W, device=self.device)
        return TensorDict(data, batch_size=[self.num_envs], device=self.device)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:  # noqa: D102
        self.episode_length_buf += 1
        dones = (self.episode_length_buf >= self.max_episode_length).float()
        self.episode_length_buf[dones.bool()] = 0
        obs = self.get_observations()
        rewards = torch.randn(self.num_envs, device=self.device)
        extras = {"time_outs": torch.zeros(self.num_envs, device=self.device)}
        return obs, rewards, dones, extras


def _make_train_cfg(model_type: str = "mlp") -> dict:
    """Return a minimal training configuration for PPO.

    Args:
        model_type: One of ``"mlp"``, ``"rnn"``, or ``"cnn"``.
    """
    cfg: dict = {
        "num_steps_per_env": 8,
        "save_interval": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 2,
            "num_mini_batches": 2,
        },
    }
    if model_type == "rnn":
        cfg["actor"] = {
            "class_name": "RNNModel",
            "hidden_dims": [32],
            "rnn_type": "gru",
            "rnn_hidden_dim": 16,
            "rnn_num_layers": 1,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
            },
        }
        cfg["critic"] = {
            "class_name": "RNNModel",
            "hidden_dims": [32],
            "rnn_type": "gru",
            "rnn_hidden_dim": 16,
            "rnn_num_layers": 1,
        }
    elif model_type == "cnn":
        cfg["obs_groups"] = {
            "actor": ["policy", "image"],
            "critic": ["policy", "image"],
        }
        cnn_cfg = {
            "output_channels": [4],
            "kernel_size": 3,
            "stride": 2,
        }
        cfg["actor"] = {
            "class_name": "CNNModel",
            "hidden_dims": [32],
            "activation": "elu",
            "cnn_cfg": cnn_cfg,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
            },
        }
        cfg["critic"] = {
            "class_name": "CNNModel",
            "hidden_dims": [32],
            "activation": "elu",
            "cnn_cfg": cnn_cfg,
        }
    else:
        cfg["actor"] = {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
            },
        }
        cfg["critic"] = {
            "class_name": "MLPModel",
            "hidden_dims": [32, 32],
            "activation": "elu",
        }
    return cfg


def _build_runner(log_dir: str | None = None, model_type: str = "mlp") -> OnPolicyRunner:
    """Construct a runner with a DummyEnv and minimal config."""
    env = DummyEnv(include_image=(model_type == "cnn"))
    cfg = _make_train_cfg(model_type)
    return OnPolicyRunner(env, cfg, log_dir=log_dir, device="cpu")


class TestRunnerConstruction:
    """Tests for constructing the runner and its components."""

    def test_runner_creates_algorithm(self) -> None:
        """Runner should instantiate a PPO algorithm with actor and critic."""
        runner = _build_runner()
        assert runner.alg is not None
        assert runner.alg.actor is not None
        assert runner.alg.critic is not None

    def test_runner_sets_initial_iteration(self) -> None:
        """Initial learning iteration should be zero."""
        runner = _build_runner()
        assert runner.current_learning_iteration == 0


class TestLearnLoop:
    """Tests that the learn loop runs and updates parameters."""

    def test_learn_runs_without_error(self) -> None:
        """A short learn call should complete without raising."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=2)

    def test_learn_updates_parameters(self) -> None:
        """Actor parameters should change after a learning iteration."""
        runner = _build_runner()
        params_before = {n: p.clone() for n, p in runner.alg.actor.named_parameters()}
        runner.learn(num_learning_iterations=2)
        changed = any(not torch.equal(params_before[n], p) for n, p in runner.alg.actor.named_parameters())
        assert changed, "Actor parameters should have changed after learning"

    def test_learn_advances_iteration_counter(self) -> None:
        """current_learning_iteration should reflect completed iterations."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=3)
        assert runner.current_learning_iteration == 2


class TestSaveLoad:
    """Tests for checkpoint save and load."""

    def test_save_creates_file(self) -> None:
        """save() should create a checkpoint file at the given path."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=1)
        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            data = torch.load(f.name, weights_only=False, map_location="cpu")
            assert "iter" in data

    def test_load_restores_parameters(self) -> None:
        """Loading a checkpoint should restore model parameters exactly."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=2)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            saved_actor = copy.deepcopy(runner.alg.actor.state_dict())

            runner.learn(num_learning_iterations=2)
            assert not all(torch.equal(saved_actor[k], v) for k, v in runner.alg.actor.state_dict().items()), (
                "Parameters should have changed after additional training"
            )

            runner.load(f.name)
            for key, param in runner.alg.actor.state_dict().items():
                assert torch.equal(saved_actor[key], param), f"Parameter '{key}' not restored after load"

    def test_load_restores_iteration(self) -> None:
        """Loading a checkpoint should restore the iteration counter."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=3)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            saved_iter = runner.current_learning_iteration

            runner.learn(num_learning_iterations=2)
            assert runner.current_learning_iteration != saved_iter

            runner.load(f.name)
            assert runner.current_learning_iteration == saved_iter

    def test_load_restores_normalization_stats(self) -> None:
        """Running-mean stats should be identical after save and load."""
        cfg = _make_train_cfg("mlp")
        cfg["actor"]["obs_normalization"] = True
        cfg["critic"]["obs_normalization"] = True

        runner = OnPolicyRunner(DummyEnv(), cfg, log_dir=None, device="cpu")
        runner.learn(num_learning_iterations=2)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            saved_state = copy.deepcopy(runner.alg.actor.state_dict())

            cfg2 = _make_train_cfg("mlp")
            cfg2["actor"]["obs_normalization"] = True
            cfg2["critic"]["obs_normalization"] = True
            runner2 = OnPolicyRunner(DummyEnv(), cfg2, log_dir=None, device="cpu")
            runner2.load(f.name)

            for key, param in runner2.alg.actor.state_dict().items():
                assert torch.equal(saved_state[key], param), f"Normalization stat '{key}' not restored after load"

    def test_curriculum_state_round_trips_through_save_load(self) -> None:
        """runner.save embeds curriculum_state, runner.load calls manager.load_state_dict with it."""

        class _FakeCurriculumManager:
            def __init__(self) -> None:
                self.tensor = torch.tensor([1.0, 2.0, 3.0])
                self.last_loaded: dict | None = None

            def state_dict(self) -> dict:
                return {"my_term": {"x": self.tensor.clone()}}

            def load_state_dict(self, state: dict) -> None:
                self.last_loaded = state
                self.tensor.copy_(state["my_term"]["x"])

        env = DummyEnv()
        env.curriculum_manager = _FakeCurriculumManager()
        runner = OnPolicyRunner(env, _make_train_cfg("mlp"), log_dir=None, device="cpu")
        runner.learn(num_learning_iterations=1)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            saved_dict = torch.load(f.name, weights_only=False, map_location="cpu")
            assert "curriculum_state" in saved_dict
            assert torch.equal(saved_dict["curriculum_state"]["my_term"]["x"], torch.tensor([1.0, 2.0, 3.0]))

            env2 = DummyEnv()
            env2.curriculum_manager = _FakeCurriculumManager()
            env2.curriculum_manager.tensor.copy_(torch.tensor([99.0, 99.0, 99.0]))
            runner2 = OnPolicyRunner(env2, _make_train_cfg("mlp"), log_dir=None, device="cpu")
            runner2.load(f.name)

            assert env2.curriculum_manager.last_loaded is not None, "load_state_dict was never called"
            assert torch.equal(env2.curriculum_manager.tensor, torch.tensor([1.0, 2.0, 3.0]))

    def test_save_curriculum_state_false_omits_section(self) -> None:
        """``save_curriculum_state=False`` should skip the curriculum_state key entirely."""

        class _FakeCurriculumManager:
            def state_dict(self) -> dict:
                return {"my_term": {"x": torch.tensor([1.0])}}

            def load_state_dict(self, state: dict) -> None:
                pass

        env = DummyEnv()
        env.curriculum_manager = _FakeCurriculumManager()
        cfg = _make_train_cfg("mlp")
        cfg["save_curriculum_state"] = False
        runner = OnPolicyRunner(env, cfg, log_dir=None, device="cpu")
        runner.learn(num_learning_iterations=1)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            saved_dict = torch.load(f.name, weights_only=False, map_location="cpu")
            assert "curriculum_state" not in saved_dict

    def test_reset_curriculum_on_load_skips_restore(self) -> None:
        """``reset_curriculum_on_load=True`` should skip calling ``manager.load_state_dict``."""

        class _FakeCurriculumManager:
            def __init__(self) -> None:
                self.tensor = torch.tensor([1.0])
                self.load_calls = 0

            def state_dict(self) -> dict:
                return {"my_term": {"x": self.tensor.clone()}}

            def load_state_dict(self, state: dict) -> None:
                self.load_calls += 1

        env = DummyEnv()
        env.curriculum_manager = _FakeCurriculumManager()
        runner = OnPolicyRunner(env, _make_train_cfg("mlp"), log_dir=None, device="cpu")
        runner.learn(num_learning_iterations=1)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)

            env2 = DummyEnv()
            env2.curriculum_manager = _FakeCurriculumManager()
            cfg2 = _make_train_cfg("mlp")
            cfg2["reset_curriculum_on_load"] = True
            runner2 = OnPolicyRunner(env2, cfg2, log_dir=None, device="cpu")
            runner2.load(f.name)
            assert env2.curriculum_manager.load_calls == 0, (
                "reset_curriculum_on_load=True should skip the manager's load_state_dict"
            )

    def test_event_state_round_trips_through_save_load(self) -> None:
        """runner.save embeds event_state, runner.load calls manager.load_state_dict with it."""

        class _FakeEventManager:
            def __init__(self) -> None:
                self.tensor = torch.tensor([4.0, 5.0, 6.0])
                self.last_loaded: dict | None = None

            def state_dict(self) -> dict:
                return {"reset_term": {"x": self.tensor.clone()}}

            def load_state_dict(self, state: dict) -> None:
                self.last_loaded = state
                self.tensor.copy_(state["reset_term"]["x"])

        env = DummyEnv()
        env.event_manager = _FakeEventManager()
        runner = OnPolicyRunner(env, _make_train_cfg("mlp"), log_dir=None, device="cpu")
        runner.learn(num_learning_iterations=1)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            saved_dict = torch.load(f.name, weights_only=False, map_location="cpu")
            assert "event_state" in saved_dict
            assert torch.equal(saved_dict["event_state"]["reset_term"]["x"], torch.tensor([4.0, 5.0, 6.0]))

            env2 = DummyEnv()
            env2.event_manager = _FakeEventManager()
            env2.event_manager.tensor.copy_(torch.tensor([99.0, 99.0, 99.0]))
            runner2 = OnPolicyRunner(env2, _make_train_cfg("mlp"), log_dir=None, device="cpu")
            runner2.load(f.name)

            assert env2.event_manager.last_loaded is not None, "load_state_dict was never called"
            assert torch.equal(env2.event_manager.tensor, torch.tensor([4.0, 5.0, 6.0]))

    def test_save_event_state_false_omits_section(self) -> None:
        """``save_event_state=False`` should skip the event_state key entirely."""

        class _FakeEventManager:
            def state_dict(self) -> dict:
                return {"reset_term": {"x": torch.tensor([1.0])}}

            def load_state_dict(self, state: dict) -> None:
                pass

        env = DummyEnv()
        env.event_manager = _FakeEventManager()
        cfg = _make_train_cfg("mlp")
        cfg["save_event_state"] = False
        runner = OnPolicyRunner(env, cfg, log_dir=None, device="cpu")
        runner.learn(num_learning_iterations=1)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            saved_dict = torch.load(f.name, weights_only=False, map_location="cpu")
            assert "event_state" not in saved_dict

    def test_reset_event_on_load_skips_restore(self) -> None:
        """``reset_event_on_load=True`` should skip calling ``manager.load_state_dict``."""

        class _FakeEventManager:
            def __init__(self) -> None:
                self.tensor = torch.tensor([1.0])
                self.load_calls = 0

            def state_dict(self) -> dict:
                return {"reset_term": {"x": self.tensor.clone()}}

            def load_state_dict(self, state: dict) -> None:
                self.load_calls += 1

        env = DummyEnv()
        env.event_manager = _FakeEventManager()
        runner = OnPolicyRunner(env, _make_train_cfg("mlp"), log_dir=None, device="cpu")
        runner.learn(num_learning_iterations=1)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)

            env2 = DummyEnv()
            env2.event_manager = _FakeEventManager()
            cfg2 = _make_train_cfg("mlp")
            cfg2["reset_event_on_load"] = True
            runner2 = OnPolicyRunner(env2, cfg2, log_dir=None, device="cpu")
            runner2.load(f.name)
            assert env2.event_manager.load_calls == 0, (
                "reset_event_on_load=True should skip the manager's load_state_dict"
            )

    def test_load_restores_optimizer_state(self) -> None:
        """Loading should restore Adam optimizer state (step, exp_avg, exp_avg_sq)."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=2)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            saved = copy.deepcopy(runner.alg.optimizer.state_dict()["state"])

            runner.learn(num_learning_iterations=2)
            current = runner.alg.optimizer.state_dict()["state"]
            assert any(
                isinstance(saved[pid][field], torch.Tensor)
                and not torch.equal(saved[pid][field], current[pid][field])
                for pid in saved
                for field in ("exp_avg", "exp_avg_sq")
            ), "Optimizer moments should change after additional training"

            runner.load(f.name)
            loaded = runner.alg.optimizer.state_dict()["state"]
            for pid, fields in saved.items():
                for name, val in fields.items():
                    if isinstance(val, torch.Tensor):
                        assert torch.equal(val, loaded[pid][name]), f"optimizer state[{pid}][{name}] not restored"
                    else:
                        assert val == loaded[pid][name], f"optimizer state[{pid}][{name}] not restored"


class TestInferencePolicy:
    """Tests for get_inference_policy and the returned callable."""

    def test_inference_policy_returns_callable(self) -> None:
        """get_inference_policy should return a callable model."""
        runner = _build_runner()
        policy = runner.get_inference_policy()
        assert callable(policy)

    def test_inference_policy_produces_actions(self) -> None:
        """The inference policy should return a tensor with the correct action shape."""
        runner = _build_runner()
        policy = runner.get_inference_policy()
        obs = runner.env.get_observations()
        actions = policy(obs)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_inference_loop(self) -> None:
        """Simulate a replay loop: step the env with policy outputs for several steps."""
        runner = _build_runner()
        runner.learn(num_learning_iterations=1)
        policy = runner.get_inference_policy()

        obs = runner.env.get_observations()
        for _ in range(5):
            actions = policy(obs)
            obs, rewards, _dones, _extras = runner.env.step(actions)
            assert rewards.shape == (NUM_ENVS,)


class TestDeterministicTraining:
    """Two seeded training runs should produce identical results."""

    @staticmethod
    def _seeded_train(seed: int, model_type: str = "mlp") -> dict[str, torch.Tensor]:
        """Run a short training loop with a fixed seed and return actor state_dict."""
        torch.manual_seed(seed)
        runner = _build_runner(model_type=model_type)
        runner.learn(num_learning_iterations=3)
        return {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}

    def test_mlp_reproducibility(self) -> None:
        """Two MLP training runs with the same seed should yield identical parameters."""
        run_a = self._seeded_train(seed=42, model_type="mlp")
        run_b = self._seeded_train(seed=42, model_type="mlp")
        for key in run_a:
            assert torch.equal(run_a[key], run_b[key]), f"MLP param '{key}' differs between seeded runs"

    def test_rnn_reproducibility(self) -> None:
        """Two RNN training runs with the same seed should yield identical parameters."""
        run_a = self._seeded_train(seed=42, model_type="rnn")
        run_b = self._seeded_train(seed=42, model_type="rnn")
        for key in run_a:
            assert torch.equal(run_a[key], run_b[key]), f"RNN param '{key}' differs between seeded runs"

    def test_different_seeds_diverge(self) -> None:
        """Different seeds should produce different parameters."""
        run_a = self._seeded_train(seed=42)
        run_b = self._seeded_train(seed=99)
        any_different = any(not torch.equal(run_a[k], run_b[k]) for k in run_a)
        assert any_different, "Different seeds should produce different parameters"


class TestRNNRunner:
    """Tests that the full learn loop works with an RNN-based actor/critic."""

    def test_rnn_learn_runs_without_error(self) -> None:
        """A short learn call with RNN models should complete without raising."""
        runner = _build_runner(model_type="rnn")
        runner.learn(num_learning_iterations=2)

    def test_rnn_learn_updates_parameters(self) -> None:
        """RNN actor parameters should change after learning."""
        runner = _build_runner(model_type="rnn")
        params_before = {n: p.clone() for n, p in runner.alg.actor.named_parameters()}
        runner.learn(num_learning_iterations=2)
        changed = any(not torch.equal(params_before[n], p) for n, p in runner.alg.actor.named_parameters())
        assert changed, "RNN actor parameters should have changed after learning"

    def test_rnn_inference_produces_actions(self) -> None:
        """Inference policy from an RNN runner should return correct action shape."""
        runner = _build_runner(model_type="rnn")
        policy = runner.get_inference_policy()
        obs = runner.env.get_observations()
        actions = policy(obs)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_rnn_save_load_restores_parameters(self) -> None:
        """Save/load should preserve RNN model parameters (including hidden state shapes)."""
        runner = _build_runner(model_type="rnn")
        runner.learn(num_learning_iterations=2)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            saved_actor = copy.deepcopy(runner.alg.actor.state_dict())

            runner.learn(num_learning_iterations=2)
            runner.load(f.name)

            for key, param in runner.alg.actor.state_dict().items():
                assert torch.equal(saved_actor[key], param), f"RNN parameter '{key}' not restored after load"


class TestCNNRunner:
    """Tests that the full learn loop works with a CNN-based actor/critic."""

    def test_cnn_learn_runs_without_error(self) -> None:
        """A short learn call with CNN models should complete without raising."""
        runner = _build_runner(model_type="cnn")
        runner.learn(num_learning_iterations=2)

    def test_cnn_learn_updates_parameters(self) -> None:
        """CNN actor parameters should change after learning."""
        runner = _build_runner(model_type="cnn")
        params_before = {n: p.clone() for n, p in runner.alg.actor.named_parameters()}
        runner.learn(num_learning_iterations=2)
        changed = any(not torch.equal(params_before[n], p) for n, p in runner.alg.actor.named_parameters())
        assert changed, "CNN actor parameters should have changed after learning"

    def test_cnn_inference_produces_actions(self) -> None:
        """Inference policy from a CNN runner should return correct action shape."""
        runner = _build_runner(model_type="cnn")
        policy = runner.get_inference_policy()
        obs = runner.env.get_observations()
        actions = policy(obs)
        assert actions.shape == (NUM_ENVS, NUM_ACTIONS)

    def test_cnn_save_load_restores_parameters(self) -> None:
        """Save/load should preserve CNN model parameters."""
        runner = _build_runner(model_type="cnn")
        runner.learn(num_learning_iterations=2)

        with tempfile.NamedTemporaryFile(suffix=".pt") as f:
            runner.save(f.name)
            saved_actor = copy.deepcopy(runner.alg.actor.state_dict())

            runner.learn(num_learning_iterations=2)
            runner.load(f.name)

            for key, param in runner.alg.actor.state_dict().items():
                assert torch.equal(saved_actor[key], param), f"CNN parameter '{key}' not restored after load"


class TestGradientNoiseScaleIntegration:
    """Tests for the gradient-noise-scale instrumentation in :class:`PPO`."""

    @staticmethod
    def _build_runner_with_grad_noise_cfg(grad_noise_cfg: dict | None) -> OnPolicyRunner:
        """Build a fresh runner with ``cfg['algorithm']['gradient_noise_scale_cfg']`` set."""
        env = DummyEnv()
        cfg = _make_train_cfg("mlp")
        if grad_noise_cfg is not None:
            cfg["algorithm"]["gradient_noise_scale_cfg"] = grad_noise_cfg
        return OnPolicyRunner(env, cfg, log_dir=None, device="cpu")

    @staticmethod
    def _run_one_update(runner: OnPolicyRunner) -> dict[str, float]:
        """Drive one rollout + PPO update and return the resulting loss_dict."""
        runner.alg.train_mode()
        obs = runner.env.get_observations()
        for _ in range(8):  # matches num_steps_per_env in _make_train_cfg
            actions = runner.alg.act(obs)
            obs, rewards, dones, extras = runner.env.step(actions)
            runner.alg.process_env_step(obs, rewards, dones, extras)
        runner.alg.compute_returns(obs)
        return runner.alg.update()

    def test_metric_absent_when_cfg_is_none(self) -> None:
        """No gradient_noise_scale_cfg ⇒ tracker is None and no noise_scale/* keys appear."""
        runner = self._build_runner_with_grad_noise_cfg(None)
        assert runner.alg.noise_scale_tracker is None
        loss_dict = self._run_one_update(runner)
        assert not any(key.startswith("noise_scale/") for key in loss_dict)

    def test_metric_absent_when_disabled(self) -> None:
        """enabled=False ⇒ tracker is None and no noise_scale/* keys appear."""
        runner = self._build_runner_with_grad_noise_cfg({"enabled": False})
        assert runner.alg.noise_scale_tracker is None
        loss_dict = self._run_one_update(runner)
        assert not any(key.startswith("noise_scale/") for key in loss_dict)

    def test_metric_present_when_enabled(self) -> None:
        """across_minibatches mode populates B_simple, G_sq, sigma_tr in loss_dict."""
        runner = self._build_runner_with_grad_noise_cfg({"enabled": True, "mode": "across_minibatches"})
        assert runner.alg.noise_scale_tracker is not None
        loss_dict = self._run_one_update(runner)
        assert "noise_scale/B_simple" in loss_dict
        assert "noise_scale/G_sq" in loss_dict
        assert "noise_scale/sigma_tr" in loss_dict
        # After one update the denominator EMA must have moved off its initial zero.
        assert loss_dict["noise_scale/G_sq"] != 0.0

    def test_within_minibatch_mode_still_unimplemented(self) -> None:
        """The within_minibatch mode is reserved but not implemented yet."""
        with pytest.raises(NotImplementedError, match="within_minibatch"):
            self._build_runner_with_grad_noise_cfg({"enabled": True, "mode": "within_minibatch"})

    def test_explicit_ddp_native_requires_multi_gpu(self) -> None:
        """mode='ddp_native' must error on single-GPU (b_big == b_small would div-by-zero)."""
        with pytest.raises(ValueError, match="multi-GPU"):
            self._build_runner_with_grad_noise_cfg({"enabled": True, "mode": "ddp_native"})

    def test_auto_resolves_to_across_minibatches_on_single_gpu(self) -> None:
        """With no multi_gpu_cfg, mode='auto' resolves to across_minibatches."""
        runner = self._build_runner_with_grad_noise_cfg({"enabled": True, "mode": "auto"})
        assert runner.alg.noise_scale_tracker is not None
        assert runner.alg.noise_scale_tracker.mode == "across_minibatches"

    def test_auto_resolves_to_ddp_native_when_multi_gpu(self) -> None:
        """With multi_gpu_cfg set, mode='auto' resolves to ddp_native."""
        ppo, _ = _build_ppo(
            gradient_noise_scale_cfg={"enabled": True, "mode": "auto"},
            multi_gpu_cfg={"global_rank": 0, "local_rank": 0, "world_size": 2},
        )
        assert ppo.noise_scale_tracker is not None
        assert ppo.noise_scale_tracker.mode == "ddp_native"
        assert ppo.noise_scale_tracker.gpu_world_size == 2

    def test_unknown_cfg_key_raises(self) -> None:
        """Unrecognized cfg keys should be rejected so typos do not silently no-op."""
        with pytest.raises(ValueError, match="unrecognized"):
            self._build_runner_with_grad_noise_cfg({"enabled": True, "typo_key": 0.5})

    def test_bit_identical_disabled_vs_across_minibatches(self) -> None:
        """Enabling the metric must not perturb the training trajectory.

        Two seeded :meth:`learn` calls — one with the tracker off, one with
        ``across_minibatches`` on — must produce byte-equal actor, critic, and
        optimizer state. This is the load-bearing guarantee that the tracker
        only reads ``p.grad`` and never writes to it.
        """

        def seeded_run(grad_noise_cfg: dict | None) -> dict[str, object]:
            torch.manual_seed(42)
            runner = self._build_runner_with_grad_noise_cfg(grad_noise_cfg)
            runner.learn(num_learning_iterations=3)
            return {
                "actor": {k: v.clone() for k, v in runner.alg.actor.state_dict().items()},
                "critic": {k: v.clone() for k, v in runner.alg.critic.state_dict().items()},
                "optimizer": runner.alg.optimizer.state_dict(),
            }

        state_off = seeded_run(None)
        state_on = seeded_run({"enabled": True, "mode": "across_minibatches"})

        for component in ("actor", "critic"):
            for key in state_off[component]:
                assert torch.equal(state_off[component][key], state_on[component][key]), (
                    f"{component} param '{key}' diverged when noise scale tracker is enabled"
                )

        # Adam optimizer state (exp_avg, exp_avg_sq, step).
        optimizer_state_off = state_off["optimizer"]["state"]
        optimizer_state_on = state_on["optimizer"]["state"]
        assert optimizer_state_off.keys() == optimizer_state_on.keys()
        for param_id in optimizer_state_off:
            for entry_key, entry_val in optimizer_state_off[param_id].items():
                other = optimizer_state_on[param_id][entry_key]
                if isinstance(entry_val, torch.Tensor):
                    assert torch.equal(entry_val, other), (
                        f"optimizer state[{param_id}]['{entry_key}'] diverged when tracker is enabled"
                    )
                else:
                    assert entry_val == other

    @pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.device_count() < 2,
        reason="ddp_native smoke test requires >= 2 CUDA devices",
    )
    def test_ddp_native_smoke_two_ranks(self) -> None:
        """End-to-end DDP smoke: 2 NCCL ranks must produce a finite, rank-agreeing B_simple.

        ``step_ddp_native`` all-reduces the local gradient norm, so after the call
        every rank's EMA inputs are identical and therefore every rank's ``B_simple``
        must be bit-equal up to the order of independent NCCL operations on the same
        deterministic inputs.
        """
        ctx = mp.get_context("spawn")
        result_queue: mp.Queue = ctx.Queue()
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            free_port = probe.getsockname()[1]

        procs = [ctx.Process(target=_ddp_smoke_worker, args=(rank, 2, free_port, result_queue)) for rank in range(2)]
        for proc in procs:
            proc.start()
        for proc in procs:
            proc.join(timeout=180)
            assert proc.exitcode == 0, f"DDP worker (pid={proc.pid}) exited with {proc.exitcode}"

        per_rank: dict[int, dict[str, object]] = {}
        for _ in range(2):
            payload = result_queue.get(timeout=5)
            per_rank[payload["rank"]] = payload
        assert set(per_rank) == {0, 1}
        for rank, payload in per_rank.items():
            assert payload["mode"] == "ddp_native", f"rank {rank} mode={payload['mode']}"
            # The unbiased estimators (G_sq, sigma_tr) can be transiently negative for the
            # first few EMA samples — we only require finiteness here, not positivity.
            assert math.isfinite(payload["B_simple"]), f"rank {rank} B_simple={payload['B_simple']}"
            assert math.isfinite(payload["G_sq"]), f"rank {rank} G_sq={payload['G_sq']}"
            assert math.isfinite(payload["sigma_tr"]), f"rank {rank} sigma_tr={payload['sigma_tr']}"
        # The load-bearing invariant: step_ddp_native all-reduces its inputs, so every
        # rank's EMA inputs are identical and the resulting state must agree across ranks.
        assert per_rank[0]["B_simple"] == pytest.approx(per_rank[1]["B_simple"], rel=1e-5, abs=1e-12)
        assert per_rank[0]["G_sq"] == pytest.approx(per_rank[1]["G_sq"], rel=1e-5, abs=1e-12)
        assert per_rank[0]["sigma_tr"] == pytest.approx(per_rank[1]["sigma_tr"], rel=1e-5, abs=1e-12)


def _ddp_smoke_worker(rank: int, world_size: int, port: int, result_queue: mp.Queue) -> None:
    """One DDP rank: set up rendezvous env vars then let the runner handle init."""
    try:
        # OnPolicyRunner._configure_multi_gpu reads these env vars and calls
        # ``torch.distributed.init_process_group`` itself; we must not double-init.
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(port)
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank)

        device = f"cuda:{rank}"
        env = DummyEnv(device=device)
        cfg = _make_train_cfg("mlp")
        cfg["algorithm"]["gradient_noise_scale_cfg"] = {"enabled": True, "mode": "auto"}
        runner = OnPolicyRunner(env, cfg, log_dir=None, device=device)
        runner.learn(num_learning_iterations=2)

        state = runner.alg.noise_scale_tracker.state()
        result_queue.put({
            "rank": rank,
            "mode": runner.alg.noise_scale_tracker.mode,
            "B_simple": state["B_simple"],
            "G_sq": state["G_sq"],
            "sigma_tr": state["sigma_tr"],
        })
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
