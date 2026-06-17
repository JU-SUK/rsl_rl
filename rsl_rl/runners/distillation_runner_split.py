# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DistillationRunner with a fixed per-env student/teacher pool split.

Ported from pat-gravity's ``uwlab_rl.rsl_rl.distillation_runner_split`` and
re-targeted at feature/manipulation's :class:`rsl_rl.runners.OnPolicyRunner` API
(``self.logger`` instead of ``self.writer`` / ``self._prepare_logging_writer``).

Env layout along the batch axis:

* ``[0, num_student_eval)`` — student-eval (no-grad, student drives). Pure
  inference signal even at tight train cadences where same-step gradients
  create echo-of-teacher bias on the train pool.
* ``[num_student_eval, num_eval)`` — teacher-eval (no-grad, teacher drives).
* ``[num_eval, num_eval + num_student_train)`` — student-train (gradient).
* ``[num_eval + num_student_train, num_envs)`` — teacher-train (gradient).

The masks are injected into the algorithm cfg so :class:`DistillationDAgger`
applies them to action mixing + loss masking, and stashed on
``env.unwrapped.{pool_mask, eval_mask}`` so visual-DR events can target the
no-grad eval pool with OOD textures.
"""

from __future__ import annotations

import os
import statistics
import time
from collections import deque

import torch

from rsl_rl.env import VecEnv
from rsl_rl.runners import DistillationRunner


class DistillationRunnerSplit(DistillationRunner):
    """:class:`DistillationRunner` + fixed student/teacher/eval pool split.

    Subclasses ``DistillationRunner`` (which itself subclasses ``OnPolicyRunner``
    on this branch). Overrides ``__init__`` to compute the per-env mask and
    inject it into the algorithm cfg before :meth:`OnPolicyRunner.__init__`
    constructs the algorithm via :meth:`DistillationDAgger.construct_algorithm`.

    The rollout loop in :meth:`learn` reimplements parent logic so each step
    can route teacher actions on the teacher pool, exclude eval transitions
    from the loss, track per-pool rolling success rates, and log them to
    :attr:`self.logger.writer` alongside the standard PPO metrics.
    """

    # --------------------------------------------------------------------- #
    # Setup                                                                 #
    # --------------------------------------------------------------------- #

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        # Strip split-only keys before the parent sees the cfg.
        self.student_fraction = float(train_cfg.pop("student_fraction", 0.5))
        self.eval_fraction = float(train_cfg.pop("eval_fraction", 0.0))
        self.teacher_eval_fraction = float(train_cfg.pop("teacher_eval_fraction", 0.0))
        for name, v in [
            ("student_fraction", self.student_fraction),
            ("eval_fraction", self.eval_fraction),
            ("teacher_eval_fraction", self.teacher_eval_fraction),
        ]:
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]; got {v}")

        num_envs = env.num_envs
        num_eval = round(num_envs * self.eval_fraction)
        num_teacher_eval = round(num_eval * self.teacher_eval_fraction)
        num_student_eval = num_eval - num_teacher_eval
        num_train = num_envs - num_eval
        num_student_train = round(num_train * self.student_fraction)
        num_teacher_train = num_train - num_student_train

        student_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        student_mask[:num_student_eval] = True
        student_mask[num_eval : num_eval + num_student_train] = True

        eval_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        eval_mask[:num_eval] = True

        self.student_mask = student_mask
        self.eval_mask = eval_mask
        self.num_eval = num_eval
        self.num_student_eval = num_student_eval
        self.num_teacher_eval = num_teacher_eval
        self.num_student_train = num_student_train
        self.num_teacher_train = num_teacher_train

        # Inject masks into the algorithm cfg so DistillationDAgger.__init__ binds them.
        train_cfg["algorithm"] = dict(train_cfg["algorithm"])
        train_cfg["algorithm"]["student_mask"] = student_mask
        train_cfg["algorithm"]["eval_mask"] = eval_mask

        # Expose for env-side visual DR / per-pool obs randomization.
        env.unwrapped.pool_mask = student_mask.to(env.unwrapped.device)
        env.unwrapped.eval_mask = eval_mask.to(env.unwrapped.device)

        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

        # Rolling per-pool success buckets. Cadence-independent (don't rely on
        # the reset event populating extras["log"] on every step).
        self._pool_buf_len = 1024
        self._student_train_success_buf: deque[float] = deque(maxlen=self._pool_buf_len)
        self._teacher_train_success_buf: deque[float] = deque(maxlen=self._pool_buf_len)
        self._student_eval_success_buf: deque[float] = deque(maxlen=self._pool_buf_len)
        self._teacher_eval_success_buf: deque[float] = deque(maxlen=self._pool_buf_len)
        # Per-reset-bucket success: {(pool, tag_name): deque}. pool in
        # student_train/teacher_train/student_eval/teacher_eval; tag_name is the reset
        # strategy (grasp_asset_in_air / start_assembled / start_grasped). Lets us see
        # which reset types the teacher/student succeed on, per pool.
        from collections import defaultdict as _defaultdict
        self._bucket_success = _defaultdict(lambda: deque(maxlen=self._pool_buf_len))
        # MSE(student_action, teacher_action) on the no-grad student-eval pool.
        # Direct sim2real proxy when visual DR fires OOD textures on this pool.
        self._bc_loss_student_eval_buf: deque[float] = deque(maxlen=self._pool_buf_len)

        # Also keep a regular reward + length buffer for parity with
        # OnPolicyRunner.learn (logger reads from logger.rewbuffer, but BC has
        # no advantage rewards so we hand-fill them here for cadence parity).
        self._rewbuffer: deque[float] = deque(maxlen=100)
        self._lenbuffer: deque[float] = deque(maxlen=100)

        print(
            f"[DistillationRunnerSplit] pool split: "
            f"{num_student_eval} student-eval / {num_teacher_eval} teacher-eval (no-grad) | "
            f"{num_student_train} student-train / {num_teacher_train} teacher-train "
            f"(student_fraction={self.student_fraction}, eval_fraction={self.eval_fraction}, "
            f"teacher_eval_fraction={self.teacher_eval_fraction})"
        )

    # --------------------------------------------------------------------- #
    # Training loop                                                         #
    # --------------------------------------------------------------------- #

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:  # type: ignore[override]
        """Rollout + update loop with per-pool success tracking."""
        if not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.policy.train()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        # Initialize the logging writer.
        self.logger.init_logging_writer()

        # Cache the per-env success signal source. Optional — if the env
        # doesn't expose a success termination term we just skip pool tracking.
        try:
            success_term = self.env.unwrapped.termination_manager.get_term("success")
        except (AttributeError, KeyError):
            success_term = None

        # Cache the reset accumulator so we can attribute each done env's success to
        # the reset strategy (bucket) it was reset from (``sampled_tags`` per env).
        try:
            _acc = self.env.unwrapped.event_manager.get_term_cfg("reset_positioning").func
            _bucket_names = list(_acc.names)
        except Exception:
            _acc = None
            _bucket_names = []

        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            # Rollouts in eval mode: encoder skips photometric augs (only the
            # imagenet_norm step applies). Augs fire only during the gradient
            # update below, on the train-pool data. This is what makes the
            # eval pool's success metric a clean signal.
            self.alg.policy.eval()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    # Per-step BC loss on the student-eval pool. Teacher action
                    # was already computed inside alg.act() and stored on
                    # alg.transition.privileged_actions.
                    student_eval_m = self.student_mask & self.eval_mask
                    if student_eval_m.any():
                        teacher_act = self.alg.transition.privileged_actions
                        diff = (actions[student_eval_m] - teacher_act[student_eval_m]) ** 2
                        self._bc_loss_student_eval_buf.append(diff.mean().item())

                    # Snapshot the per-env reset tag BEFORE stepping: env.step auto-
                    # resets done envs in-place (overwriting sampled_tags with the NEXT
                    # episode's tag), so we must capture the ending episode's tag now to
                    # attribute its success to the correct reset bucket.
                    _pre_tags = (
                        _acc.sampled_tags.clone()
                        if _acc is not None and getattr(_acc, "sampled_tags", None) is not None
                        else None
                    )
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)

                    cur_reward_sum += rewards
                    cur_episode_length += 1
                    new_ids = (dones > 0).nonzero(as_tuple=False)
                    if new_ids.numel() > 0:
                        self._rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        self._lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

                    # Per-pool success tracking. Drop ``corrupted_camera``-
                    # terminated envs from the success rate — it's a rendering
                    # glitch, not a policy failure.
                    if success_term is not None:
                        valid_dones = dones
                        try:
                            corrupted = self.env.unwrapped.termination_manager.get_term("corrupted_camera")
                            valid_dones = dones & ~corrupted.to(dones.device)
                        except (KeyError, AttributeError):
                            pass
                        done_ids = valid_dones.view(-1).nonzero(as_tuple=False).view(-1)
                        if done_ids.numel() > 0:
                            succ = success_term[done_ids].float()
                            eval_m = self.eval_mask[done_ids]
                            student_m = self.student_mask[done_ids]
                            self._student_eval_success_buf.extend(
                                succ[student_m & eval_m].detach().cpu().tolist()
                            )
                            self._teacher_eval_success_buf.extend(
                                succ[~student_m & eval_m].detach().cpu().tolist()
                            )
                            self._student_train_success_buf.extend(
                                succ[student_m & ~eval_m].detach().cpu().tolist()
                            )
                            self._teacher_train_success_buf.extend(
                                succ[~student_m & ~eval_m].detach().cpu().tolist()
                            )
                            # Per-reset-bucket breakdown: attribute each done env's
                            # success to its reset strategy (sampled_tags) x pool.
                            if _pre_tags is not None:
                                tags_d = _pre_tags[done_ids].to(succ.device)
                                pools = {
                                    "student_train": student_m & ~eval_m,
                                    "teacher_train": ~student_m & ~eval_m,
                                    "student_eval": student_m & eval_m,
                                    "teacher_eval": ~student_m & eval_m,
                                }
                                for ti, tname in enumerate(_bucket_names):
                                    tag_m = tags_d == ti
                                    if not tag_m.any():
                                        continue
                                    for pool, pmask in pools.items():
                                        sel = pmask & tag_m
                                        if sel.any():
                                            self._bucket_success[(pool, tname)].extend(
                                                succ[sel].detach().cpu().tolist()
                                            )

                    # Also let the new logger pull standard episode info.
                    self.logger.process_env_step(rewards, dones, extras, None)

                collect_time = time.time() - start
                start = time.time()

            # Switch to train mode for the gradient update.
            self.alg.policy.train()
            loss_dict = self.alg.update()
            learn_time = time.time() - start
            self.current_learning_iteration = it

            # Standard PPO-style log; rnd_weight=None (no RND on distillation).
            # ``action_std`` only exists on stochastic policies; for the
            # student-teacher policy stack the closest analogue is the student
            # std, but it's not a model-wide attribute. Fall back to a 1-tensor
            # of ones so the logger's mean/std step doesn't crash.
            action_std = getattr(self.alg.policy, "std", None)
            if action_std is None:
                log_std = getattr(self.alg.policy, "log_std", None)
                action_std = torch.exp(log_std) if log_std is not None else torch.ones(1, device=self.device)
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=action_std,
                rnd_weight=None,
                policy_metrics=None,
            )

            # Append per-pool success metrics directly to the writer (the
            # standard logger pipeline doesn't carry an injection point for
            # arbitrary scalars after ``log`` returns).
            writer = self.logger.writer
            if writer is not None:
                if self._student_train_success_buf:
                    writer.add_scalar(
                        "Metrics/success_student_train",
                        statistics.mean(self._student_train_success_buf),
                        it,
                    )
                if self._teacher_train_success_buf:
                    writer.add_scalar(
                        "Metrics/success_teacher_train",
                        statistics.mean(self._teacher_train_success_buf),
                        it,
                    )
                if self._student_eval_success_buf:
                    writer.add_scalar(
                        "Metrics/success_student_eval",
                        statistics.mean(self._student_eval_success_buf),
                        it,
                    )
                if self._teacher_eval_success_buf:
                    writer.add_scalar(
                        "Metrics/success_teacher_eval",
                        statistics.mean(self._teacher_eval_success_buf),
                        it,
                    )
                # Per-reset-bucket success (pool x strategy), e.g.
                # SuccessBucket/student_train/start_assembled.
                for (pool, tname), buf in self._bucket_success.items():
                    if buf:
                        writer.add_scalar(f"SuccessBucket/{pool}/{tname}", statistics.mean(buf), it)
                if self._bc_loss_student_eval_buf:
                    writer.add_scalar(
                        "Metrics/bc_loss_student_eval",
                        statistics.mean(self._bc_loss_student_eval_buf),
                        it,
                    )

            # Save model
            if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore
            self.logger.stop_logging_writer()
