# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the gradient noise scale tracker (McCandlish et al. 2018, B_simple)."""

from __future__ import annotations

import torch

import pytest

from rsl_rl.utils import GradientNoiseScaleTracker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simulate_shards(
    rng: torch.Generator,
    mu: torch.Tensor,
    sigma: float,
    b_big: int,
    b_small: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Sample ``b_big`` per-sample grads ~ N(mu, sigma^2 * I) and split into shards.

    Returns ``(big_mean, shard_means)`` where ``big_mean`` is the average of
    all ``b_big`` samples and ``shard_means`` is a list of M = b_big // b_small
    shard means, each averaged over ``b_small`` consecutive samples.
    """
    if b_big % b_small != 0:
        raise ValueError("b_big must be a multiple of b_small")
    dim = mu.numel()
    samples = mu + sigma * torch.randn(b_big, dim, generator=rng)
    n_shards = b_big // b_small
    shards = samples.view(n_shards, b_small, dim)
    shard_means = shards.mean(dim=1)
    big_mean = samples.mean(dim=0)
    return big_mean, [shard_means[i].clone() for i in range(n_shards)]


def _step_mode(
    tracker: GradientNoiseScaleTracker,
    big_mean: torch.Tensor,
    shard_means: list[torch.Tensor],
    b_small: int,
) -> None:
    """Drive one EMA update through whichever mode the tracker is configured for.

    All three modes are mathematically equivalent given the same shard means;
    only the entry point differs. This helper hides the wiring so the
    convergence tests are identical across modes.
    """
    n_shards = len(shard_means)
    b_big = b_small * n_shards
    if tracker.mode == "within_minibatch":
        shard_norm_sqs = [s.pow(2).sum() for s in shard_means]
        big_norm_sq = big_mean.pow(2).sum()
        tracker.step_within_minibatch(shard_norm_sqs, big_norm_sq, b_small, b_big)
    elif tracker.mode == "ddp_native":
        # without dist init, step_ddp_native uses local_norm_sq as-is. Pass
        # the across-rank average ourselves so the math matches the other modes.
        local = torch.stack([s.pow(2).sum() for s in shard_means]).mean()
        big = big_mean.pow(2).sum()
        tracker.step_ddp_native(local, big, b_small, b_big)
    elif tracker.mode == "across_minibatches":
        dim = shard_means[0].numel()
        param = torch.nn.Parameter(torch.zeros(dim))
        tracker.begin_iteration()
        for s in shard_means:
            param.grad = s.clone()
            tracker.accumulate_minibatch([param])
        tracker.step_across_minibatches(b_small, n_shards)
    else:
        raise AssertionError(f"unhandled mode {tracker.mode}")


def _run_known_gaussian(
    mode: str,
    *,
    dim: int = 16,
    mu_value: float = 0.5,
    sigma: float = 1.0,
    b_small: int = 4,
    b_big: int = 32,
    n_iter: int = 1000,
    ema_decay: float = 0.9,
    seed: int = 0,
) -> tuple[float, float]:
    """Run a closed-form-Gaussian convergence simulation for one mode.

    Returns ``(estimated_B_simple, expected_B_simple)`` where
    ``expected = dim * sigma^2 / |mu|^2``.
    """
    mu = torch.full((dim,), mu_value)
    expected = (dim * sigma**2) / mu.pow(2).sum().item()
    rng = torch.Generator()
    rng.manual_seed(seed)
    tracker = GradientNoiseScaleTracker(mode=mode, ema_decay=ema_decay)
    for _ in range(n_iter):
        big_mean, shard_means = _simulate_shards(rng, mu, sigma, b_big, b_small)
        _step_mode(tracker, big_mean, shard_means, b_small)
    return tracker.state()["B_simple"], expected


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Constructor argument validation and default state."""

    def test_invalid_mode_raises(self) -> None:
        """Unknown mode strings should be rejected up front."""
        with pytest.raises(ValueError, match="mode="):
            GradientNoiseScaleTracker(mode="bogus")  # type: ignore[arg-type]

    @pytest.mark.parametrize("decay", [0.0, 1.0, -0.1, 1.5])
    def test_invalid_ema_decay_raises(self, decay: float) -> None:
        """``ema_decay`` outside the open interval (0, 1) is invalid."""
        with pytest.raises(ValueError, match="ema_decay"):
            GradientNoiseScaleTracker(mode="across_minibatches", ema_decay=decay)

    def test_invalid_world_size_raises(self) -> None:
        """``gpu_world_size`` < 1 is invalid."""
        with pytest.raises(ValueError, match="gpu_world_size"):
            GradientNoiseScaleTracker(mode="ddp_native", gpu_world_size=0)

    def test_defaults_present(self) -> None:
        """Constructor sets EMAs to zero and ``num_updates`` to zero."""
        t = GradientNoiseScaleTracker(mode="within_minibatch")
        assert t.num_updates == 0
        assert torch.equal(t.ema_g_sq, torch.zeros(()))
        assert torch.equal(t.ema_sigma_tr, torch.zeros(()))


# ---------------------------------------------------------------------------
# state()
# ---------------------------------------------------------------------------


class TestState:
    """Reporting interface."""

    def test_state_keys(self) -> None:
        """``state()`` returns a dict with B_simple, G_sq, sigma_tr as floats."""
        t = GradientNoiseScaleTracker(mode="within_minibatch")
        s = t.state()
        assert set(s) == {"B_simple", "G_sq", "sigma_tr"}
        assert all(isinstance(v, float) for v in s.values())

    def test_state_zero_at_init(self) -> None:
        """Before any updates, B_simple is 0 (numerator clamped at 0)."""
        t = GradientNoiseScaleTracker(mode="within_minibatch")
        assert t.state()["B_simple"] == 0.0


# ---------------------------------------------------------------------------
# No-noise: identical small and big norms ⇒ tr(Σ) → 0 ⇒ B_simple → 0
# ---------------------------------------------------------------------------


class TestNoNoiseConvergence:
    """Identical per-shard gradients should drive ``B_simple`` to zero."""

    @pytest.mark.parametrize("mode", ["ddp_native", "across_minibatches", "within_minibatch"])
    def test_no_noise_collapses_to_zero(self, mode: str) -> None:
        """A constant gradient field has tr(Σ)=0 so the noise scale is 0."""
        dim, b_small, n_shards = 8, 4, 4
        mu = torch.full((dim,), 0.3)
        # zero noise: every shard mean equals mu, big mean equals mu.
        big_mean = mu.clone()
        shard_means = [mu.clone() for _ in range(n_shards)]
        tracker = GradientNoiseScaleTracker(mode=mode, ema_decay=0.9)
        for _ in range(200):
            _step_mode(tracker, big_mean, shard_means, b_small)
        s = tracker.state()
        assert s["B_simple"] == pytest.approx(0.0, abs=1e-6)
        # numerator should be tiny (tr(Σ) estimator); denominator should equal |mu|^2
        assert s["sigma_tr"] == pytest.approx(0.0, abs=1e-5)
        assert s["G_sq"] == pytest.approx(mu.pow(2).sum().item(), rel=1e-4)


# ---------------------------------------------------------------------------
# Closed-form: B_simple ≈ D * σ² / |μ|² when per-sample grads ~ N(μ, σ²·I)
# ---------------------------------------------------------------------------


class TestKnownGaussianConvergence:
    """Each mode's EMA must approach the closed-form noise scale."""

    @pytest.mark.parametrize(
        ("mode", "seed"),
        [
            ("ddp_native", 0),
            ("across_minibatches", 0),
            ("within_minibatch", 0),
        ],
    )
    def test_known_gaussian_matches_closed_form(self, mode: str, seed: int) -> None:
        """Per-sample N(μ, σ²·I) gradients give the textbook B_simple."""
        estimated, expected = _run_known_gaussian(mode=mode, seed=seed)
        # 30% relative tolerance: 1000 iters at decay=0.9 -> N_eff ~ 100,
        # standard error ~ 10%, so 30% leaves comfortable margin for flakes.
        assert estimated == pytest.approx(expected, rel=0.3), (
            f"{mode}: estimated B_simple={estimated:.3f}, expected={expected:.3f}"
        )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same seed twice ⇒ identical EMA state across the run."""

    @pytest.mark.parametrize("mode", ["ddp_native", "across_minibatches", "within_minibatch"])
    def test_same_seed_same_emas(self, mode: str) -> None:
        """Two runs with the same RNG seed produce bit-identical EMAs."""

        def run() -> tuple[torch.Tensor, torch.Tensor, int]:
            mu = torch.full((8,), 0.5)
            rng = torch.Generator()
            rng.manual_seed(42)
            tracker = GradientNoiseScaleTracker(mode=mode, ema_decay=0.9)
            for _ in range(50):
                big_mean, shard_means = _simulate_shards(rng, mu, sigma=1.0, b_big=16, b_small=4)
                _step_mode(tracker, big_mean, shard_means, 4)
            return tracker.ema_g_sq.clone(), tracker.ema_sigma_tr.clone(), tracker.num_updates

        g1, s1, n1 = run()
        g2, s2, n2 = run()
        assert torch.equal(g1, g2)
        assert torch.equal(s1, s2)
        assert n1 == n2


# ---------------------------------------------------------------------------
# accumulate_minibatch error path
# ---------------------------------------------------------------------------


class TestAccumulateGuards:
    """``across_minibatches`` mode rejects out-of-order calls."""

    def test_accumulate_without_begin_raises(self) -> None:
        """``accumulate_minibatch`` must follow ``begin_iteration``."""
        t = GradientNoiseScaleTracker(mode="across_minibatches")
        param = torch.nn.Parameter(torch.zeros(4))
        param.grad = torch.ones(4)
        with pytest.raises(RuntimeError, match="begin_iteration"):
            t.accumulate_minibatch([param])

    def test_step_without_accumulate_raises(self) -> None:
        """``step_across_minibatches`` requires at least one accumulate call."""
        t = GradientNoiseScaleTracker(mode="across_minibatches")
        t.begin_iteration()
        # accumulate_minibatch was never called, so _sum_grad is still None.
        with pytest.raises(RuntimeError, match="accumulate_minibatch"):
            t.step_across_minibatches(b_small=4, num_mini_batches=2)


# ---------------------------------------------------------------------------
# Save / load round trip
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """``state_dict`` / ``load_state_dict`` preserve the persistent EMA counters."""

    def test_round_trip_preserves_ema_state(self) -> None:
        """A non-trivial EMA history round-trips through save+load into a fresh tracker."""
        mu = torch.full((8,), 0.5)
        rng = torch.Generator()
        rng.manual_seed(42)
        src = GradientNoiseScaleTracker(mode="ddp_native", ema_decay=0.9)
        for _ in range(25):
            big_mean, shard_means = _simulate_shards(rng, mu, sigma=1.0, b_big=16, b_small=4)
            _step_mode(src, big_mean, shard_means, 4)

        assert src.num_updates == 25
        snapshot = src.state_dict()

        dst = GradientNoiseScaleTracker(mode="ddp_native", ema_decay=0.9)
        ema_g_sq_ref = dst.ema_g_sq
        ema_sigma_tr_ref = dst.ema_sigma_tr
        dst.load_state_dict(snapshot)

        assert dst.num_updates == 25
        assert torch.equal(dst.ema_g_sq, src.ema_g_sq)
        assert torch.equal(dst.ema_sigma_tr, src.ema_sigma_tr)
        # In-place restore must keep the original buffer tensor objects alive.
        assert dst.ema_g_sq is ema_g_sq_ref
        assert dst.ema_sigma_tr is ema_sigma_tr_ref

    def test_snapshot_decoupled_from_source(self) -> None:
        """Continuing to update the source after ``state_dict`` does not mutate the snapshot."""
        mu = torch.full((4,), 0.0)
        rng = torch.Generator()
        rng.manual_seed(1)
        src = GradientNoiseScaleTracker(mode="ddp_native", ema_decay=0.9)
        for _ in range(5):
            big_mean, shard_means = _simulate_shards(rng, mu, sigma=1.0, b_big=8, b_small=2)
            _step_mode(src, big_mean, shard_means, 2)
        snapshot = src.state_dict()

        for _ in range(5):
            big_mean, shard_means = _simulate_shards(rng, mu, sigma=1.0, b_big=8, b_small=2)
            _step_mode(src, big_mean, shard_means, 2)

        # Source has advanced, but the snapshot must still reflect the earlier state.
        assert src.num_updates == 10
        assert snapshot["num_updates"] == 5
        assert not torch.equal(snapshot["ema_g_sq"], src.ema_g_sq)

    def test_resumed_updates_match_uninterrupted_run(self) -> None:
        """A save+load mid-stream produces the same final EMAs as a single-shot run."""
        mu = torch.full((4,), 0.1)

        # Reference: 8 updates on one tracker.
        rng_ref = torch.Generator()
        rng_ref.manual_seed(7)
        ref = GradientNoiseScaleTracker(mode="ddp_native", ema_decay=0.9)
        for _ in range(8):
            big_mean, shard_means = _simulate_shards(rng_ref, mu, sigma=1.0, b_big=8, b_small=2)
            _step_mode(ref, big_mean, shard_means, 2)

        # Run: 4 updates, save, then 4 more on a fresh tracker that loaded the snapshot.
        # Share the same RNG so the simulated minibatches are identical.
        rng_run = torch.Generator()
        rng_run.manual_seed(7)
        run = GradientNoiseScaleTracker(mode="ddp_native", ema_decay=0.9)
        for _ in range(4):
            big_mean, shard_means = _simulate_shards(rng_run, mu, sigma=1.0, b_big=8, b_small=2)
            _step_mode(run, big_mean, shard_means, 2)
        snapshot = run.state_dict()

        resumed = GradientNoiseScaleTracker(mode="ddp_native", ema_decay=0.9)
        resumed.load_state_dict(snapshot)
        for _ in range(4):
            big_mean, shard_means = _simulate_shards(rng_run, mu, sigma=1.0, b_big=8, b_small=2)
            _step_mode(resumed, big_mean, shard_means, 2)

        assert torch.equal(resumed.ema_g_sq, ref.ema_g_sq)
        assert torch.equal(resumed.ema_sigma_tr, ref.ema_sigma_tr)
        assert resumed.num_updates == ref.num_updates
