# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Regression tests for the TanhGaussianDistribution NaN blow-up (cluster runs died iters 9-61).

Root cause chain: saturated stored actions -> atanh lands far from the updated mean ->
Jacobian-corrected log-prob differences exceed the fp32 exp() limit (~88) -> PPO ratio = inf
-> inf surrogate loss -> NaN gradients -> NaN network weights -> ``Normal`` raises
"normal expects all elements of std >= 0.0" at the next sample.

Two-layer fix under test:
1. PPO clamps the log-ratio to [-20, 20] before exponentiation (ppo.py).
2. ``TanhGaussianDistribution.update`` sanitizes non-finite actor outputs (distribution.py).
"""

import math

import pytest
import torch

from rsl_rl.modules.distribution import TanhGaussianDistribution


def _make_dist(mean: torch.Tensor, log_std: torch.Tensor) -> TanhGaussianDistribution:
    dist = TanhGaussianDistribution(
        output_dim=mean.shape[-1], std_type="log", std_range=(1e-3, 1e6), init_std=1.0
    )
    dist.update(torch.stack([mean, log_std], dim=-2))
    return dist


def test_log_ratio_overflow_reproduced_then_clamped():
    """The exact overflow that killed the cluster runs: unclamped exp() -> inf; clamped -> finite."""
    n, d = 4, 6
    # collection-time policy: wide std, mean 0 -> modest old log-prob
    old = _make_dist(torch.zeros(n, d), torch.zeros(n, d))
    # saturated stored actions (the policy pushed |u| big, tanh ~ +-1)
    actions = torch.full((n, d), 1.0 - 1e-7)
    old_logp = old.log_prob(actions)
    # post-update policy: mean moved onto atanh(a), std collapsed to the floor
    u = torch.atanh(actions.clamp(-1 + 1e-6, 1 - 1e-6))
    new = _make_dist(u, torch.full((n, d), math.log(1e-3)))
    new_logp = new.log_prob(actions)

    log_ratio = new_logp - old_logp
    assert torch.isfinite(log_ratio).all(), "log-probs themselves must stay finite"
    assert (log_ratio > 88.0).any(), "regression precondition lost: log-ratio no longer exceeds the fp32 exp limit"

    # OLD code path (pre-fix): direct exponentiation overflows to inf -> inf loss -> NaN grads.
    ratio_unclamped = torch.exp(log_ratio)
    assert torch.isinf(ratio_unclamped).any(), "this is the bug being regression-tested"

    # NEW code path (the ppo.py fix): clamped log-ratio stays finite.
    ratio_clamped = torch.exp(torch.clamp(log_ratio, -20.0, 20.0))
    assert torch.isfinite(ratio_clamped).all()
    surrogate = (-torch.ones(n) * ratio_clamped).max()
    assert torch.isfinite(surrogate)


def test_update_sanitizes_non_finite_actor_outputs():
    """NaN/inf actor outputs must not produce a crashing ``Normal`` (the 'std >= 0' error)."""
    n, d = 3, 6
    mean = torch.zeros(n, d)
    log_std = torch.zeros(n, d)
    mean[0, 0] = float("nan")
    mean[1, 1] = float("inf")
    log_std[2, 2] = float("nan")  # clamp() would pass this through; exp(nan)=nan std -> Normal raises on sample
    dist = _make_dist(mean, log_std)
    a = dist.sample()
    assert torch.isfinite(a).all()
    assert (a.abs() <= 1.0).all()
    assert torch.isfinite(dist.log_prob(a)).all()


def test_entropy_and_logprob_bounded_at_saturation():
    """Sampled entropy and log-probs stay finite even with saturated means at the std floor."""
    n, d = 8, 6
    dist = _make_dist(torch.full((n, d), 10.0), torch.full((n, d), math.log(1e-3)))
    assert torch.isfinite(dist.entropy).all()
    a = dist.sample()
    assert torch.isfinite(dist.log_prob(a)).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
