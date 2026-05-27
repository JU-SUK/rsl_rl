# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import os
import time
import torch
from datetime import timedelta

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.utils import check_nan, resolve_callable
from rsl_rl.utils.logger import Logger


class OnPolicyRunner:
    """On-policy runner for reinforcement learning algorithms."""

    alg: PPO
    """The actor-critic algorithm."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        """Construct the runner, algorithm, and logging stack."""
        self.env = env
        self.cfg = train_cfg
        self.device = device

        # Setup multi-GPU training if enabled
        self._configure_multi_gpu()

        # Query observations from the environment for algorithm construction
        obs = self.env.get_observations()

        # Create the algorithm
        alg_class: type[PPO] = resolve_callable(self.cfg["algorithm"]["class_name"])  # type: ignore
        self.alg = alg_class.construct_algorithm(obs, self.env, self.cfg, self.device)

        # Create the logger
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

        self.current_learning_iteration = 0
        self._resumed = False

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run the learning loop for the specified number of iterations."""
        # Randomize initial episode lengths (for exploration)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        # Start learning
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()  # switch to train mode (for dropout for example)

        # Ensure all parameters are in-synced
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Initialize the logging writer
        self.logger.init_logging_writer()

        # Optional warmup: rollouts without gradient updates to fill success monitors.
        warmup_iters = self.cfg.get("resume_warmup_iterations", 0)
        if self._resumed and warmup_iters > 0:
            self._run_warmup(obs, warmup_iters)

        # Start training
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            # Rollout
            with torch.inference_mode():
                # gSDE: re-sample the exploration matrix at the start of every rollout. No-op for non-gSDE actors.
                self.alg.reset_sde_noise(self.env.num_envs)
                for step in range(self.cfg["num_steps_per_env"]):
                    # gSDE: re-sample the exploration matrix every sde_sample_freq env-steps within the rollout.
                    if self.alg.sde_sample_freq > 0 and step > 0 and step % self.alg.sde_sample_freq == 0:
                        self.alg.reset_sde_noise(self.env.num_envs)
                    # Sample actions
                    actions = self.alg.act(obs)
                    # Step the environment
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    # Check for NaN values from the environment
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    # Move to device
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    # Process the step
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    # Extract intrinsic rewards if RND is used (only for logging)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    # Book keeping
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop

                # Compute returns
                self.alg.compute_returns(obs)

                # Per-task policy metrics (action magnitude always; entropy only for
                # heteroscedastic Gaussians since homoscedastic entropy is redundant with
                # Loss/entropy). Computed before update() clears storage; graceful no-op
                # on non-IsaacLab envs — see :meth:`_compute_per_task_policy_metrics`.
                policy_metrics = self._compute_per_task_policy_metrics()

            # Update policy
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            # Log information
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
                policy_metrics=policy_metrics,
            )

            # Save model
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        # Save the final model after training and stop the logging writer
        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()

    def save(self, path: str, infos: dict | None = None) -> None:
        """Save the models and training state to a given path and upload them if external logging is used."""
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        if self.cfg.get("save_curriculum_state", True):
            curriculum_state = self._get_curriculum_state()
            if curriculum_state:
                saved_dict["curriculum_state"] = curriculum_state
        if self.cfg.get("save_event_state", True):
            event_state = self._get_event_state()
            if event_state:
                saved_dict["event_state"] = event_state
        if self.cfg.get("save_logger_state", True):
            saved_dict["logger_state"] = self.logger.state_dict()
        torch.save(saved_dict, path)
        # Upload model to external logging services
        self.logger.save_model(path, self.current_learning_iteration)

    def load(
        self, path: str, load_cfg: dict | None = None, strict: bool = True, map_location: str | None = None
    ) -> dict:
        """Load the models and training state from a given path.

        Args:
            path (str): Path to load the model from.
            load_cfg (dict | None): Optional dictionary that defines what models and states to load. If None, all
                models and states are loaded.
            strict (bool): Whether state_dict loading should be strict.
            map_location (str | None): Device mapping for loading the model.
        """
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
        if load_iteration:
            self.current_learning_iteration = loaded_dict["iter"]
        if not self.cfg.get("reset_curriculum_on_load", False) and "curriculum_state" in loaded_dict:
            self._set_curriculum_state(loaded_dict["curriculum_state"])
        if not self.cfg.get("reset_event_on_load", False) and "event_state" in loaded_dict:
            self._set_event_state(loaded_dict["event_state"])
        if not self.cfg.get("reset_logger_on_load", False) and "logger_state" in loaded_dict:
            self.logger.load_state_dict(loaded_dict["logger_state"])
        self._resumed = True
        return loaded_dict["infos"]

    def _run_warmup(self, obs: torch.Tensor, num_iters: int) -> None:
        """Roll out the policy without gradient updates to fill success monitors."""
        if self.gpu_global_rank == 0:
            print(f"[warmup] Running {num_iters} rollout-only iterations to fill success monitors...")
        num_steps = self.cfg["num_steps_per_env"]
        for w in range(num_iters):
            with torch.inference_mode():
                self.alg.reset_sde_noise(self.env.num_envs)
                for step in range(num_steps):
                    if self.alg.sde_sample_freq > 0 and step > 0 and step % self.alg.sde_sample_freq == 0:
                        self.alg.reset_sde_noise(self.env.num_envs)
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    if self.cfg.get("check_for_nan", True):
                        check_nan(obs, rewards, dones)
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self.logger.process_env_step(rewards, dones, extras, None)
            self.alg.storage.clear()
            if self.gpu_global_rank == 0:
                ep_done = dones.sum().item()
                mean_rew = rewards.mean().item()
                print(f"[warmup] iter {w + 1}/{num_iters}  eps_done={ep_done:.0f}  mean_rew={mean_rew:.3f}")
        if self.gpu_global_rank == 0:
            print(f"[warmup] Done ({num_iters} iterations).")

    def _compute_per_task_policy_metrics(self) -> dict[str, float] | None:
        """Per-terrain-type policy metrics over the just-collected rollout.

        Returns a flat dict mapping ``<metric>/<terrain_name>`` → float. The logger
        prefixes each key with ``Policy/``, so e.g.
        ``per_task_action_magnitude/stepping_stone`` becomes
        ``Policy/per_task_action_magnitude/stepping_stone`` in TB / W&B.

        Always-logged:

        - **Action magnitude.** ``mean(||a||_2)`` per terrain type. Depends on the action
          *mean*, so it varies across envs even when ``std`` is global.

        Conditionally-logged:

        - **Action-distribution entropy.** Only for *heteroscedastic* Gaussians
          (state-dependent ``std``). For homoscedastic Gaussians the per-task split is
          degenerate (entropy depends only on ``std``, which is constant across envs),
          and the same signal already lives in ``Loss/entropy`` — within ~0.003 nat,
          which is just the std drift across the K minibatch update steps.

        Returns ``None`` if the env is not an IsaacLab terrain env or storage hasn't
        been populated yet — graceful no-op outside the locomotion-task setup.
        """
        try:
            terrain = self.env.unwrapped.scene.terrain  # type: ignore[attr-defined]
            terrain_types = terrain.terrain_types  # [num_envs] long
            sub_terrain_names = list(terrain.cfg.terrain_generator.sub_terrains.keys())
        except (AttributeError, KeyError):
            return None

        actions = self.alg.storage.actions
        if actions is None:
            return None

        metrics: dict[str, float] = {}

        # Per-task action magnitude. ``actions`` is [num_steps, num_envs, action_dim].
        action_magnitudes = actions.norm(dim=-1)  # [num_steps, num_envs]
        for terrain_index, name in enumerate(sub_terrain_names):
            mask = terrain_types == terrain_index
            if mask.any():
                metrics[f"per_task_action_magnitude/{name}"] = action_magnitudes[:, mask].mean().item()

        # Per-task entropy — only worth logging if std varies across envs (heteroscedastic).
        params = self.alg.storage.distribution_params
        if params is not None and len(params) == 2:
            mean, std = params
            if mean.ndim == 3 and std.shape == mean.shape:
                std_varies_across_envs = std.std(dim=1).max().item() > 1e-9
                if std_varies_across_envs:
                    entropy_per_step_env = (0.5 * (2.0 * torch.pi * torch.e * std.pow(2)).log()).sum(dim=-1)
                    for terrain_index, name in enumerate(sub_terrain_names):
                        mask = terrain_types == terrain_index
                        if mask.any():
                            metrics[f"per_task_entropy/{name}"] = entropy_per_step_env[:, mask].mean().item()

        return metrics if metrics else None

    def _get_curriculum_state(self) -> dict | None:
        """Return the curriculum manager's serializable state, or ``None`` if unavailable."""
        env = getattr(self.env, "unwrapped", self.env)
        manager = getattr(env, "curriculum_manager", None)
        if manager is None or not hasattr(manager, "state_dict"):
            return None
        return manager.state_dict()

    def _set_curriculum_state(self, state: dict) -> None:
        """Restore curriculum state previously produced by :meth:`_get_curriculum_state`."""
        env = getattr(self.env, "unwrapped", self.env)
        manager = getattr(env, "curriculum_manager", None)
        if manager is None or not hasattr(manager, "load_state_dict"):
            return
        manager.load_state_dict(state)

    def _get_event_state(self) -> dict | None:
        """Return the event manager's serializable state, or ``None`` if unavailable."""
        env = getattr(self.env, "unwrapped", self.env)
        manager = getattr(env, "event_manager", None)
        if manager is None or not hasattr(manager, "state_dict"):
            return None
        return manager.state_dict()

    def _set_event_state(self, state: dict) -> None:
        """Restore event state previously produced by :meth:`_get_event_state`."""
        env = getattr(self.env, "unwrapped", self.env)
        manager = getattr(env, "event_manager", None)
        if manager is None or not hasattr(manager, "load_state_dict"):
            return
        manager.load_state_dict(state)

    def get_inference_policy(self, device: str | None = None) -> MLPModel:
        """Return the policy on the requested device for inference."""
        self.alg.eval_mode()  # Switch to evaluation mode (e.g. for dropout)
        return self.alg.get_policy().to(device)  # type: ignore

    def export_policy_to_jit(self, path: str, filename: str = "policy.pt") -> None:
        """Export the model to a Torch JIT file."""
        jit_model = self.alg.get_policy().as_jit()
        jit_model.to("cpu")

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)

        # Trace and save the model
        traced_model = torch.jit.script(jit_model)
        traced_model.save(save_path)

    def export_policy_to_onnx(self, path: str, filename: str = "policy.onnx", verbose: bool = False) -> None:
        """Export the model into an ONNX file."""
        onnx_model = self.alg.get_policy().as_onnx(verbose=verbose)
        onnx_model.to("cpu")
        onnx_model.eval()

        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        save_path = os.path.join(path, filename)

        # Trace and save the model
        torch.onnx.export(
            onnx_model,
            onnx_model.get_dummy_inputs(),  # type: ignore
            save_path,
            export_params=True,
            opset_version=18,
            verbose=verbose,
            input_names=onnx_model.input_names,  # type: ignore
            output_names=onnx_model.output_names,  # type: ignore
        )

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        """Register a repository path whose git status should be logged."""
        self.logger.git_status_repos.append(repo_file_path)

    def add_file_to_log(self, file_path: str) -> None:
        """Register a file path to upload to W&B/Neptune when the writer initializes."""
        self.logger.log_files.append(file_path)

    def _configure_multi_gpu(self) -> None:
        """Configure multi-gpu training."""
        # Check if distributed training is enabled
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        # If not distributed training, set local and global rank to 0 and return
        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.cfg["multi_gpu"] = None
            return

        # Get rank and world size
        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        # Make a configuration dictionary
        self.cfg["multi_gpu"] = {
            "global_rank": self.gpu_global_rank,  # Rank of the main process
            "local_rank": self.gpu_local_rank,  # Rank of the current process
            "world_size": self.gpu_world_size,  # Total number of processes
        }

        # Check if user has device specified for local rank
        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        # Validate multi-GPU configuration
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        # Initialize torch distributed
        torch.distributed.init_process_group(
            backend="nccl",
            rank=self.gpu_global_rank,
            world_size=self.gpu_world_size,
            timeout=timedelta(minutes=10),
        )
        # Set device to the local rank
        torch.cuda.set_device(self.gpu_local_rank)
