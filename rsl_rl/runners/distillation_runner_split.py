# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DistillationRunner with a fixed per-env student/teacher pool split.

Assigns the first ``round(num_envs * student_fraction)`` envs to the student
pool and the rest to the teacher pool; the assignment is fixed for the entire
training run. The mask is:

* passed to :class:`DistillationDAgger` via ``student_mask`` so action routing
  honors the split, and
* stashed on ``env.unwrapped.pool_mask`` so the reset event
  (:class:`MultiResetManager`) can log ``Metrics/success_student_only`` and
  ``Metrics/success_teacher_only`` alongside the usual per-task success rates.

Per-pool success/length are also tracked in the runner's rollout loop as a
fallback: IsaacLab's ``ManagerBasedRLEnv._reset_idx`` wipes ``extras["log"]``
after the reset-mode event terms run, so ``MultiResetManager``'s writes are
only preserved by coincidence at certain rollout cadences. Tracking here is
cadence-independent and writes directly to the SummaryWriter each iter.
"""

from __future__ import annotations

from collections import deque

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import Distillation
from rsl_rl.env import VecEnv
from rsl_rl.modules import StudentTeacher
from rsl_rl.runners import DistillationRunner
from rsl_rl.utils import resolve_obs_groups


class DistillationRunnerSplit(DistillationRunner):
    """DistillationRunner that fixes a per-env student/teacher/eval pool split.

    Env layout along the batch axis:

    * envs ``[0, num_eval)``           — eval-only: student drives; transitions
      **excluded from gradient update**. Pure inference signal even at tight
      train cadences where same-step gradients create echo-of-teacher bias.
    * envs ``[num_eval, num_eval+num_student_train)`` — train-student: student
      drives; transitions in gradient.
    * envs ``[num_eval+num_student_train, num_envs)`` — train-teacher: teacher
      drives; transitions in gradient.
    """

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        # Strip split-only keys before the parent sees the cfg.
        self.student_fraction = float(train_cfg.pop("student_fraction", 0.5))
        self.eval_fraction = float(train_cfg.pop("eval_fraction", 0.0))
        # Of the eval pool, what fraction runs teacher actions (no-grad teacher rate).
        # Default 0 = backward-compatible (eval is all student). Set to 0.5 for
        # half-student / half-teacher eval — useful to verify teacher is solving the
        # env at the same time we measure student eval.
        self.teacher_eval_fraction = float(train_cfg.pop("teacher_eval_fraction", 0.0))
        if not 0.0 <= self.student_fraction <= 1.0:
            raise ValueError(f"student_fraction must be in [0, 1]; got {self.student_fraction}")
        if not 0.0 <= self.eval_fraction <= 1.0:
            raise ValueError(f"eval_fraction must be in [0, 1]; got {self.eval_fraction}")
        if not 0.0 <= self.teacher_eval_fraction <= 1.0:
            raise ValueError(f"teacher_eval_fraction must be in [0, 1]; got {self.teacher_eval_fraction}")

        num_envs = env.num_envs
        num_eval = round(num_envs * self.eval_fraction)
        num_teacher_eval = round(num_eval * self.teacher_eval_fraction)
        num_student_eval = num_eval - num_teacher_eval
        num_train = num_envs - num_eval
        num_student_train = round(num_train * self.student_fraction)
        num_teacher_train = num_train - num_student_train

        # Env layout (first→last along batch axis):
        #   [0, num_student_eval)                                         — student-eval (no-grad, student)
        #   [num_student_eval, num_eval)                                  — teacher-eval (no-grad, teacher)
        #   [num_eval, num_eval + num_student_train)                      — student-train (grad, student)
        #   [num_eval + num_student_train, num_envs)                      — teacher-train (grad, teacher)
        student_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        student_mask[:num_student_eval] = True
        student_mask[num_eval : num_eval + num_student_train] = True

        # eval_mask flags envs whose transitions are masked out of the BC gradient
        # (covers both student-eval and teacher-eval).
        eval_mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        eval_mask[:num_eval] = True

        self.student_mask = student_mask
        self.eval_mask = eval_mask
        self.num_eval = num_eval
        self.num_student_eval = num_student_eval
        self.num_teacher_eval = num_teacher_eval
        self.num_student_train = num_student_train
        self.num_teacher = num_teacher_train  # legacy alias
        self.num_teacher_train = num_teacher_train

        # Inject masks into the algorithm cfg so _construct_algorithm passes them
        # to DistillationDAgger.__init__.
        train_cfg["algorithm"] = dict(train_cfg["algorithm"])
        train_cfg["algorithm"]["student_mask"] = student_mask
        train_cfg["algorithm"]["eval_mask"] = eval_mask

        # Expose pool mask to the env for the reset event's per-pool success logging
        # (legacy; may be partially wiped by ManagerBasedRLEnv._reset_idx on certain
        # cadences — this runner also tracks per-pool success internally below).
        env.unwrapped.pool_mask = student_mask.to(env.unwrapped.device)
        # Also expose eval_mask so visual-DR events can apply OOD textures / etc.
        # to no-grad eval envs only.
        env.unwrapped.eval_mask = eval_mask.to(env.unwrapped.device)

        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

        # Rolling per-pool success buffers (cadence-independent). Four pools:
        # student-eval, teacher-eval, student-train, teacher-train.
        self._pool_buf_len = 1024
        self._student_train_success_buf: deque[float] = deque(maxlen=self._pool_buf_len)
        self._teacher_success_buf: deque[float] = deque(maxlen=self._pool_buf_len)  # teacher-train
        self._student_eval_success_buf: deque[float] = deque(maxlen=self._pool_buf_len)
        self._teacher_eval_success_buf: deque[float] = deque(maxlen=self._pool_buf_len)
        # BC loss on the student-eval pool: MSE between the no-grad student action
        # and the teacher action on the same obs. Teacher action is already computed
        # for all envs in self.alg.act() so this is essentially free per step.
        # On OOD-texture eval (per rgb_dagger_cfg) this is a direct sim2real proxy.
        self._bc_loss_student_eval_buf: deque[float] = deque(maxlen=self._pool_buf_len)

        print(
            f"[DistillationRunnerSplit] pool split: "
            f"{num_student_eval} student-eval / {num_teacher_eval} teacher-eval (no-grad) | "
            f"{num_student_train} student-train / {num_teacher_train} teacher-train "
            f"(student_fraction={self.student_fraction}, eval_fraction={self.eval_fraction}, "
            f"teacher_eval_fraction={self.teacher_eval_fraction})"
        )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:  # type: ignore[override]
        """Rollout + update loop with per-pool success tracking injected."""
        import os
        import time
        from collections import deque

        import rsl_rl

        # ``store_code_state`` lives on patrickhaoy/main but not on UW-Lab/feature/manipulation;
        # tolerantly skip if unavailable (git-status repo dump is logging-only, not required).
        try:
            from rsl_rl.utils import store_code_state  # type: ignore
        except ImportError:
            store_code_state = None  # type: ignore

        # Prepare logging (mirrors parent)
        self._prepare_logging_writer()
        if not self.alg.policy.loaded_teacher:
            raise ValueError("Teacher model parameters not loaded. Please load a teacher model to distill.")

        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        ep_infos = []
        rewbuffer: deque[float] = deque(maxlen=100)
        lenbuffer: deque[float] = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_iter = self.current_learning_iteration
        tot_iter = start_iter + num_learning_iterations

        # Cache the underlying env's progress-context reward term for direct per-env success reads.
        # Accessed once; if unavailable, per-pool tracking is skipped for that run.
        reward_mgr = getattr(self.env.unwrapped, "reward_manager", None)
        progress_term = None
        if reward_mgr is not None:
            try:
                progress_term = reward_mgr.get_term_cfg("progress_context").func
            except Exception:
                progress_term = None

        for it in range(start_iter, tot_iter):
            start = time.time()
            # Rollouts in eval mode: encoder skips photometric augs (only
            # imagenet_norm applies). The augs fire only during the gradient
            # update below, on the train-pool data. This is what makes the
            # eval pool's success metric a clean signal.
            self.eval_mode()
            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    actions = self.alg.act(obs)
                    # Per-step BC loss on student-eval envs (no-grad). Teacher
                    # action was already computed inside self.alg.act() and stored
                    # on self.alg.transition.privileged_actions.
                    student_eval_m = self.student_mask.to(actions.device) & self.eval_mask.to(actions.device)
                    if student_eval_m.any():
                        teacher_act = self.alg.transition.privileged_actions
                        diff = (actions[student_eval_m] - teacher_act[student_eval_m]) ** 2
                        self._bc_loss_student_eval_buf.append(diff.mean().item())
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)

                    # Per-pool success tracking (cadence-independent).
                    # Four categories: {student, teacher} × {eval, train}.
                    # ``corrupted_camera``-terminated envs are EXCLUDED from the
                    # success metric: it's a rendering glitch (camera std<10),
                    # not a policy failure, and shouldn't pollute success-rate.
                    if progress_term is not None:
                        valid_dones = dones
                        try:
                            corrupted = self.env.unwrapped.termination_manager.get_term("corrupted_camera")
                            valid_dones = dones & ~corrupted.to(dones.device)
                        except (KeyError, AttributeError):
                            pass  # corrupted_camera term not present — fall through
                        done_ids = valid_dones.view(-1).nonzero(as_tuple=False).view(-1)
                        if done_ids.numel() > 0:
                            succ = progress_term.success[done_ids].float()
                            eval_m = self.eval_mask.to(dones.device)[done_ids]
                            student_m = self.student_mask.to(dones.device)[done_ids]
                            student_eval_m = student_m & eval_m
                            teacher_eval_m = ~student_m & eval_m
                            student_train_m = student_m & ~eval_m
                            teacher_train_m = ~student_m & ~eval_m
                            self._student_eval_success_buf.extend(succ[student_eval_m].detach().cpu().tolist())
                            self._teacher_eval_success_buf.extend(succ[teacher_eval_m].detach().cpu().tolist())
                            self._student_train_success_buf.extend(succ[student_train_m].detach().cpu().tolist())
                            self._teacher_success_buf.extend(succ[teacher_train_m].detach().cpu().tolist())

                    if self.log_dir is not None:
                        if "episode" in extras:
                            ep_infos.append(extras["episode"])
                        elif "log" in extras:
                            ep_infos.append(extras["log"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                stop = time.time()
                collection_time = stop - start
                start = stop

            # Switch to train mode for the gradient update so the encoder's
            # photometric augs fire on the training batch (eval-pool transitions
            # are masked out of the loss by ``eval_mask``).
            self.train_mode()
            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            if self.log_dir is not None and not self.disable_logs:
                # Inject rolling per-pool success into ep_infos so it logs alongside
                # other Metrics/* keys (rsl_rl's log() uses the first ep_info's keys
                # as the schema, so inject into every entry for safety).
                pool_extras = {}
                if len(self._student_train_success_buf) > 0:
                    pool_extras["Metrics/success_student_train"] = sum(self._student_train_success_buf) / len(
                        self._student_train_success_buf
                    )
                if len(self._teacher_success_buf) > 0:
                    pool_extras["Metrics/success_teacher_train"] = sum(self._teacher_success_buf) / len(
                        self._teacher_success_buf
                    )
                if len(self._student_eval_success_buf) > 0:
                    pool_extras["Metrics/success_student_eval"] = sum(self._student_eval_success_buf) / len(
                        self._student_eval_success_buf
                    )
                if len(self._teacher_eval_success_buf) > 0:
                    pool_extras["Metrics/success_teacher_eval"] = sum(self._teacher_eval_success_buf) / len(
                        self._teacher_eval_success_buf
                    )
                if len(self._bc_loss_student_eval_buf) > 0:
                    pool_extras["Metrics/bc_loss_student_eval"] = sum(self._bc_loss_student_eval_buf) / len(
                        self._bc_loss_student_eval_buf
                    )
                if pool_extras:
                    if not ep_infos:
                        ep_infos.append(pool_extras)
                    else:
                        for ep in ep_infos:
                            ep.update(pool_extras)

                self.log(locals())
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

            ep_infos.clear()
            if it == start_iter and not self.disable_logs and store_code_state is not None:
                git_file_paths = store_code_state(self.log_dir, self.git_status_repos)
                if self.logger_type in ["wandb", "neptune"] and git_file_paths:
                    for path in git_file_paths:
                        self.writer.save_file(path)

        if self.log_dir is not None and not self.disable_logs:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
