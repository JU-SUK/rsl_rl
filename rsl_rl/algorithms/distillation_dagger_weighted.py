# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""DEXTRAH-style inverse-variance-weighted distillation loss.

Replaces the plain MSE-on-mean loss of :class:`DistillationDAgger` with::

    loss = weighted_l2(μ_student, μ_teacher, weights = (1/σ_teacher)^2)
         + l2(σ_student, σ_teacher)

where:

* ``weighted_l2(m, t, w) = sqrt(Σ_i w_i (m_i - t_i)^2)`` — per-sample L2 norm
  with per-action-dim weights, matching
  ``dextrah_lab/distillation/distillation.py:53-54``.
* ``l2(m, t) = ||m - t||_2`` — unweighted per-sample L2 norm.

Teacher ``σ_teacher`` is detached from the weighting factor (we don't want the
weighting itself to create a gradient path back into the teacher — it's frozen
anyway, but explicit detach preserves semantic clarity).

Requires:

* Policy built with ``teacher_returns_std=True`` + teacher JIT that returns
  ``(mean, std)`` (see :func:`scripts_v2/tools/convert_state_expert_to_jit.py`
  with ``--std``).
* Policy built with ``predict_std=True`` — adds a student ``std_head`` head.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict

from .distillation_dagger import DistillationDAgger


def _weighted_l2(model: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Per-sample weighted L2 norm. ``sqrt(Σ_i w_i (m_i - t_i)²)``. Returns (B,)."""
    return torch.sqrt(torch.sum(weights * (model - target) ** 2, dim=-1))


def _l2(model: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample L2 norm. ``||m - t||_2``. Returns (B,)."""
    return torch.norm(model - target, p=2, dim=-1)


class DistillationDAggerWeighted(DistillationDAgger):
    """DAgger with inverse-variance-weighted L2 on mean + L2 on std.

    Optional separate loss treatment for the binary gripper dim (last action
    dim) — addresses the failure mode where MSE'd binary gripper command
    smooths out into indecisive sub-threshold values that mistime the grasp.
    """

    def __init__(
        self,
        *args,
        gripper_loss_type: str = "shared",
        gripper_loss_weight: float = 1.0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # gripper_loss_type:
        #   "shared" — treat gripper dim like arm dims in weighted_l2 (original)
        #   "mse"    — split arm vs gripper; add λ·MSE(student_grip, teacher_grip)
        #   "bce"    — split arm vs gripper; add λ·BCE_with_logits(student_grip,
        #              (teacher_grip > 0).float()) — treats binary action as
        #              classification, more faithful to BinaryJointPositionAction
        if gripper_loss_type not in ("shared", "mse", "bce"):
            raise ValueError(f"Unknown gripper_loss_type: {gripper_loss_type}")
        self.gripper_loss_type = gripper_loss_type
        self.gripper_loss_weight = float(gripper_loss_weight)

    # Uses parent's ``act`` unchanged — teacher mean is stored as ``privileged_actions``.
    # Teacher std is re-computed on demand in ``update`` via a fresh ``evaluate_with_std``
    # call (teacher JIT is frozen + small, cheap per-step forward).

    def update(self) -> dict[str, float]:
        """Weighted BC update. Mirrors the aux-branch of :class:`DistillationDAgger.update`
        but replaces ``behavior_loss = mse(μ_s, μ_t)`` with the DEXTRAH-style
        weighted L2 + σ-term loss.

        Must re-run the student forward per step inside this loop to get
        ``(μ_s, σ_s)``; cannot reuse ``privileged_actions`` alone since σ
        supervision needs both ends.
        """
        # Unfreeze vision backbone once the warmup window ends.
        if hasattr(self.policy, "maybe_unfreeze_backbone"):
            self.policy.maybe_unfreeze_backbone(self.num_updates + 1)

        self.num_updates += 1
        mean_mu_loss = 0.0
        mean_sigma_loss = 0.0
        loss = 0
        cnt = 0

        aux_enabled = bool(getattr(self.policy, "aux_enabled", False))
        # Per-key aux loss accumulators discovered from the policy (no
        # hard-coded key list — keys come from AuxTargetCfg via the policy's
        # ``aux_heads`` ModuleDict).
        aux_keys: list[str] = list(getattr(self.policy, "aux_keys", []))
        sum_aux_train: dict[str, float] = {k: 0.0 for k in aux_keys}
        sum_aux_eval: dict[str, float] = {k: 0.0 for k in aux_keys}
        cnt_aux_train = 0
        cnt_aux_eval = 0
        train_mask = (~self.eval_mask).to(self.device) if self.eval_mask is not None else None
        eval_mask = self.eval_mask.to(self.device) if self.eval_mask is not None else None
        aux_target_group = getattr(self.policy, "aux_target_group", "aux_target")
        aux_coeff = float(getattr(self, "aux_coeff", 1.0))

        for epoch in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            for obs, _, privileged_mu, dones in self.storage.generator():
                # Student forward. Single encoder pass returns (μ, σ, aux_pred);
                # aux_pred is None when aux_enabled=False.
                if aux_enabled:
                    student_mu, student_sigma, student_aux = self.policy.forward_all_heads(obs)
                else:
                    student_mu, student_sigma = self.policy.act_inference_with_std(obs)
                    student_aux = None

                # Re-run teacher to get σ (teacher μ is already in privileged_actions).
                # Frozen JIT forward — cheap (215d → 7d MLP + gSDE head).
                _, teacher_sigma = self.policy.evaluate_with_std(obs)
                teacher_mu = privileged_mu

                # Apply eval mask (mask rows out of the gradient).
                if train_mask is not None:
                    s_mu = student_mu[train_mask]
                    s_sig = student_sigma[train_mask]
                    t_mu = teacher_mu[train_mask]
                    t_sig = teacher_sigma[train_mask]
                else:
                    s_mu, s_sig = student_mu, student_sigma
                    t_mu, t_sig = teacher_mu, teacher_sigma

                # weights = (1/σ_t)² per dim, detached.
                weights = (1.0 / t_sig.detach().clamp_min(1e-6)) ** 2
                if self.gripper_loss_type == "shared":
                    mu_loss = _weighted_l2(s_mu, t_mu, weights).mean()
                    arm_loss_val = mu_loss.item()
                    gripper_loss_val = 0.0
                else:
                    # Split arm dims (0:6) from gripper dim (6).
                    arm_loss = _weighted_l2(s_mu[:, :6], t_mu[:, :6], weights[:, :6]).mean()
                    if self.gripper_loss_type == "mse":
                        gripper_loss = (s_mu[:, 6] - t_mu[:, 6]).pow(2).mean()
                    else:  # bce
                        # Teacher's binary intent: gripper output > 0 means CLOSE.
                        binary_target = (t_mu[:, 6] > 0).float()
                        gripper_loss = nn.functional.binary_cross_entropy_with_logits(
                            s_mu[:, 6], binary_target
                        )
                    mu_loss = arm_loss + self.gripper_loss_weight * gripper_loss
                    arm_loss_val = arm_loss.item()
                    gripper_loss_val = gripper_loss.item()
                sigma_loss = _l2(s_sig, t_sig).mean()
                step_loss = mu_loss + sigma_loss

                # ---- aux pose-reconstruction loss (DEXTRAH/DP-style) ----------
                # Train pool contributes to gradient via aux_coeff * Σ_k MSE_k.
                # Eval pool is no-grad — its per-key MSE is logged as the
                # encoder-generalization metric (forced through visual DR + OOD
                # textures on the eval pool, this measures sim2real proxy).
                if student_aux is not None:
                    # ``obs[aux_target_group]`` is a flat tensor (concatenate_terms=True)
                    # because rsl_rl's RolloutStorage doesn't preserve nested
                    # TensorDicts. Slice it per-key using policy.aux_dim_per_key.
                    aux_target_flat = obs[aux_target_group]
                    dim_per_key = int(self.policy.aux_dim_per_key)
                    train_contributed = False
                    eval_contributed = False
                    for i, k in enumerate(aux_keys):
                        pred_k = student_aux[k]
                        lo, hi = i * dim_per_key, (i + 1) * dim_per_key
                        target_k = aux_target_flat[..., lo:hi]
                        if train_mask is not None:
                            s_tr, t_tr = pred_k[train_mask], target_k[train_mask]
                        else:
                            s_tr, t_tr = pred_k, target_k
                        if s_tr.shape[0] > 0:
                            mse_tr = F.mse_loss(s_tr, t_tr)
                            # Sum (not mean) over keys — matches DP form
                            # ``aux_loss = Σ_k MSE_k`` with aux_coeff=1.0.
                            step_loss = step_loss + aux_coeff * mse_tr
                            sum_aux_train[k] += mse_tr.detach().item()
                            train_contributed = True
                        if eval_mask is not None:
                            s_ev, t_ev = pred_k[eval_mask], target_k[eval_mask]
                            if s_ev.shape[0] > 0:
                                with torch.no_grad():
                                    sum_aux_eval[k] += F.mse_loss(s_ev, t_ev).item()
                                eval_contributed = True
                    if train_contributed:
                        cnt_aux_train += 1
                    if eval_contributed:
                        cnt_aux_eval += 1

                loss = loss + step_loss
                mean_mu_loss += mu_loss.item()
                mean_sigma_loss += sigma_loss.item()
                if not hasattr(self, "_mean_arm_loss"):
                    self._mean_arm_loss = 0.0
                    self._mean_gripper_loss = 0.0
                self._mean_arm_loss += arm_loss_val
                self._mean_gripper_loss += gripper_loss_val
                cnt += 1

                if cnt % self.gradient_length == 0:
                    self.optimizer.zero_grad()
                    loss.backward()
                    if self.is_multi_gpu:
                        self.reduce_parameters()
                    if self.max_grad_norm:
                        nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    self.policy.detach_hidden_states()
                    loss = 0

                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

        mean_mu_loss /= cnt
        mean_sigma_loss /= cnt
        mean_arm_loss = self._mean_arm_loss / cnt if hasattr(self, "_mean_arm_loss") else 0.0
        mean_gripper_loss = self._mean_gripper_loss / cnt if hasattr(self, "_mean_gripper_loss") else 0.0
        if hasattr(self, "_mean_arm_loss"):
            self._mean_arm_loss = 0.0
            self._mean_gripper_loss = 0.0
        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        loss_dict = {
            # Use "behavior" as the shared key for rsl_rl's logger / our plot script.
            "behavior": mean_mu_loss + mean_sigma_loss,
            "behavior_mu": mean_mu_loss,
            "behavior_sigma": mean_sigma_loss,
            "behavior_arm": mean_arm_loss,
            "behavior_gripper": mean_gripper_loss,
        }
        if aux_enabled and cnt_aux_train > 0:
            train_total = 0.0
            for k in aux_keys:
                v = sum_aux_train[k] / cnt_aux_train
                loss_dict[f"aux_train_{k}"] = v
                train_total += v
            loss_dict["aux_train_total"] = train_total
        if aux_enabled and cnt_aux_eval > 0:
            eval_total = 0.0
            for k in aux_keys:
                v = sum_aux_eval[k] / cnt_aux_eval
                loss_dict[f"aux_eval_{k}"] = v
                eval_total += v
            loss_dict["aux_eval_total"] = eval_total
        if self.student_mask is None:
            loss_dict["beta"] = self.beta
        else:
            loss_dict["student_fraction"] = float(self.student_mask.float().mean().item())
        if self.eval_mask is not None:
            loss_dict["eval_fraction"] = float(self.eval_mask.float().mean().item())
        return loss_dict
