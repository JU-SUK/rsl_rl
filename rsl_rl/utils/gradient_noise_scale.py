# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Online estimator for the gradient noise scale of McCandlish et al. 2018."""

from __future__ import annotations

import torch
import torch.distributed as dist
from collections.abc import Iterable
from typing import Literal

Mode = Literal["ddp_native", "across_minibatches", "within_minibatch"]
_VALID_MODES: tuple[str, ...] = ("ddp_native", "across_minibatches", "within_minibatch")


class GradientNoiseScaleTracker:
    """Online estimator for the *simple* gradient noise scale ``B_simple``.

    Tracks the McCandlish et al. 2018 statistic
    ``B_simple = tr(Sigma) / |G|^2`` (Eq. 2.8) using EMAs of the unbiased
    numerator and denominator estimators of Eq. A.2:

    .. code-block:: text

        |G|^2  ~  ( B_big  * |G_{B_big}|^2  - B_small * |G_{B_small}|^2 ) / (B_big - B_small)
        tr(S)  ~  ( |G_{B_small}|^2 - |G_{B_big}|^2 ) / (1/B_small - 1/B_big)

    Three estimator modes differ only in how the small- and big-batch
    gradient norms are obtained:

    - ``"ddp_native"``: per-rank vs. globally-reduced gradient norms (paper's
      canonical no-overhead recipe). Bit-identical with metric-disabled.
    - ``"across_minibatches"``: per-minibatch vs. averaged-over-K-minibatches
      gradient norms; epoch 0 only. Bit-identical with metric-disabled, with
      a small parameter-drift bias since the optimizer steps between
      minibatches.
    - ``"within_minibatch"``: split each minibatch into M shards and run M
      backwards. Unbiased w.r.t. parameters but **not** bit-identical with
      the disabled baseline because the float-order of the summed shard
      grads differs from a single full-batch backward.

    The tracker only ever *reads* ``p.grad``; it never writes to it. EMAs
    are stored as device-resident scalars to avoid host syncs each step.

    Args:
        mode: Which estimator to use.
        ema_decay: Decay factor for the running averages of the numerator
            and denominator estimators. Must lie in ``(0, 1)``.
        is_multi_gpu: Whether ``torch.distributed`` is initialized; controls
            whether :meth:`step_ddp_native` performs the cross-rank all-reduce.
        gpu_world_size: World size; only used when ``mode='ddp_native'``.
        device: Device for the EMA state tensors.
        eps: Lower clamp on ``EMA(|G|^2)`` to prevent divide-by-zero in
            :meth:`state`.
    """

    def __init__(
        self,
        mode: Mode,
        ema_decay: float = 0.99,
        is_multi_gpu: bool = False,
        gpu_world_size: int = 1,
        device: str | torch.device = "cpu",
        eps: float = 1e-12,
    ) -> None:
        """Initialize the tracker; see class docstring for the argument contract."""
        if mode not in _VALID_MODES:
            raise ValueError(f"mode={mode!r} not in {_VALID_MODES}.")
        if not 0.0 < ema_decay < 1.0:
            raise ValueError(f"ema_decay must be in (0, 1), got {ema_decay}.")
        if gpu_world_size < 1:
            raise ValueError(f"gpu_world_size must be >= 1, got {gpu_world_size}.")

        self.mode: Mode = mode
        self.ema_decay = float(ema_decay)
        self.is_multi_gpu = bool(is_multi_gpu)
        self.gpu_world_size = int(gpu_world_size)
        self.device = torch.device(device)
        self.eps = float(eps)

        self.ema_g_sq = torch.zeros((), device=self.device)
        self.ema_sigma_tr = torch.zeros((), device=self.device)
        self.num_updates = 0

        # accumulators for ``across_minibatches`` mode (lazy-allocated)
        self._sum_norm_sq: torch.Tensor | None = None
        self._sum_grad: list[torch.Tensor | None] | None = None

    # ---- public API: shared ---------------------------------------------

    def grad_norm_sq(self, params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
        """Return ``sum_p |p.grad|^2`` as a scalar tensor on the grad device."""
        return self._grad_norm_sq(params)

    def state(self) -> dict[str, float]:
        """Return current EMA state as a dict of plain floats for logging.

        Returns ``B_simple``, ``G_sq``, and ``sigma_tr``. ``B_simple`` is the
        ratio of EMAs (with the denominator clamped at ``eps`` and the
        numerator at zero) — *not* an EMA of per-step ratios, since the
        ratio of expectations is the well-defined target.
        """
        denominator = self.ema_g_sq.clamp_min(self.eps)
        numerator = self.ema_sigma_tr.clamp_min(0.0)
        b_simple = (numerator / denominator).item()
        return {
            "B_simple": b_simple,
            "G_sq": self.ema_g_sq.item(),
            "sigma_tr": self.ema_sigma_tr.item(),
        }

    # ---- public API: ddp_native -----------------------------------------

    def step_ddp_native(
        self,
        local_norm_sq: torch.Tensor,
        global_norm_sq: torch.Tensor,
        b_small: int,
        b_big: int,
    ) -> None:
        """All-reduce the local norm across ranks then update EMAs.

        Call after ``loss.backward()``: pass ``|p.grad|^2`` *before* the
        gradient all-reduce as ``local_norm_sq`` and *after* as
        ``global_norm_sq``. When :attr:`is_multi_gpu` is False (or
        ``torch.distributed`` is not initialized) the all-reduce is skipped
        and ``local_norm_sq`` is used as-is — useful for unit tests.
        """
        local = local_norm_sq.detach().to(self.device).clone()
        if self.is_multi_gpu and dist.is_available() and dist.is_initialized():
            dist.all_reduce(local, op=dist.ReduceOp.SUM)
            local = local / self.gpu_world_size
        big = global_norm_sq.detach().to(self.device)
        g_sq, sigma_tr = self._unbiased(local, big, b_small, b_big)
        self._update_emas(g_sq, sigma_tr)

    # ---- public API: across_minibatches ---------------------------------

    def begin_iteration(self) -> None:
        """Zero accumulators ahead of the first epoch's minibatch sweep."""
        self._sum_norm_sq = torch.zeros((), device=self.device)
        self._sum_grad = None  # shape known on first accumulate call

    def accumulate_minibatch(self, params: Iterable[torch.nn.Parameter]) -> None:
        """Add this minibatch's gradient norm and gradient to the accumulators.

        Reads ``p.grad`` from each parameter — never writes to it.
        """
        if self._sum_norm_sq is None:
            raise RuntimeError("call begin_iteration() before accumulate_minibatch().")
        params = list(params)
        sq = self._grad_norm_sq(params).to(self.device)
        self._sum_norm_sq = self._sum_norm_sq + sq
        if self._sum_grad is None:
            self._sum_grad = [
                torch.zeros_like(p.grad, device=self.device) if p.grad is not None else None for p in params
            ]
        for buf, p in zip(self._sum_grad, params):
            if buf is None or p.grad is None:
                continue
            buf.add_(p.grad.detach().to(self.device))

    def step_across_minibatches(self, b_small: int, num_mini_batches: int) -> None:
        """Finalize EMAs from the K accumulated minibatches and free buffers."""
        if self._sum_norm_sq is None or self._sum_grad is None:
            raise RuntimeError("call begin_iteration() and accumulate_minibatch() first.")
        small_norm_sq = self._sum_norm_sq / num_mini_batches
        big_norm_sq = torch.zeros((), device=self.device)
        for buf in self._sum_grad:
            if buf is None:
                continue
            big_norm_sq = big_norm_sq + (buf / num_mini_batches).pow(2).sum()
        b_big = b_small * num_mini_batches
        g_sq, sigma_tr = self._unbiased(small_norm_sq, big_norm_sq, b_small, b_big)
        self._update_emas(g_sq, sigma_tr)
        self._sum_norm_sq = None
        self._sum_grad = None

    # ---- public API: save/load ------------------------------------------

    def state_dict(self) -> dict:
        """Return a snapshot of the persistent EMA counters."""
        return {
            "ema_g_sq": self.ema_g_sq.detach().cpu().clone(),
            "ema_sigma_tr": self.ema_sigma_tr.detach().cpu().clone(),
            "num_updates": self.num_updates,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore the persistent EMA counters from :meth:`state_dict` output."""
        self.ema_g_sq.copy_(state["ema_g_sq"].to(self.ema_g_sq.device))
        self.ema_sigma_tr.copy_(state["ema_sigma_tr"].to(self.ema_sigma_tr.device))
        self.num_updates = int(state["num_updates"])

    # ---- public API: within_minibatch -----------------------------------

    def step_within_minibatch(
        self,
        shard_norm_sqs: list[torch.Tensor],
        big_norm_sq: torch.Tensor,
        b_small: int,
        b_big: int,
    ) -> None:
        """Update EMAs from per-shard ``|g_k|^2`` and the averaged ``|G|^2``."""
        if not shard_norm_sqs:
            raise ValueError("shard_norm_sqs must be non-empty.")
        small = torch.stack([s.detach().to(self.device) for s in shard_norm_sqs]).mean()
        big = big_norm_sq.detach().to(self.device)
        g_sq, sigma_tr = self._unbiased(small, big, b_small, b_big)
        self._update_emas(g_sq, sigma_tr)

    # ---- internals ------------------------------------------------------

    @staticmethod
    def _grad_norm_sq(params: Iterable[torch.nn.Parameter]) -> torch.Tensor:
        total: torch.Tensor | None = None
        for p in params:
            if p.grad is None:
                continue
            sq = p.grad.detach().pow(2).sum()
            total = sq if total is None else total + sq
        if total is None:
            raise RuntimeError("no gradients found on the supplied parameters.")
        return total

    @staticmethod
    def _unbiased(
        small_norm_sq: torch.Tensor,
        big_norm_sq: torch.Tensor,
        b_small: int,
        b_big: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not (b_big > b_small > 0):
            raise ValueError(f"need b_big > b_small > 0; got b_small={b_small}, b_big={b_big}.")
        denom_g = float(b_big - b_small)
        denom_sigma = (1.0 / b_small) - (1.0 / b_big)
        g_sq = (b_big * big_norm_sq - b_small * small_norm_sq) / denom_g
        sigma_tr = (small_norm_sq - big_norm_sq) / denom_sigma
        return g_sq, sigma_tr

    def _update_emas(self, g_sq: torch.Tensor, sigma_tr: torch.Tensor) -> None:
        d = self.ema_decay
        self.ema_g_sq = d * self.ema_g_sq + (1.0 - d) * g_sq
        self.ema_sigma_tr = d * self.ema_sigma_tr + (1.0 - d) * sigma_tr
        self.num_updates += 1
