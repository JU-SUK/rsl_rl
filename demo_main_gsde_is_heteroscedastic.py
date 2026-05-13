# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Standalone demonstration script; not part of the library API. The first class is a
# verbatim copy of an upstream implementation, so its signatures are not under our control.
# ruff: noqa

"""Demonstration: ``GSDENoiseDistribution`` from commit ``cae640292c`` is, in
practice, a heteroscedastic Gaussian — not generalized State Dependent
Exploration.

gSDE (Raffin et al. 2020, arXiv:2005.05719) requires that the exploration noise
be a fixed linear function of policy features::

    a_t = mu(s_t) + phi(s_t) @ epsilon,   with epsilon ~ N(0, W^2) FIXED across t

The exploration tensor ``epsilon`` is resampled only every ``sde_sample_freq``
env-steps. Because ``epsilon`` is held fixed and ``phi(s_t)`` varies smoothly
along a trajectory, the noise sequence ``phi(s_t) @ epsilon`` is temporally
CORRELATED. That is the property that makes gSDE useful for hardware-realistic
exploration — it produces smooth motor commands instead of high-frequency
jitter.

The commit's ``GSDENoiseDistribution`` (rsl_rl/modules/actor_critic.py on
``main``):

* builds the correct *marginal* ``Normal(mu, sqrt(phi^2 @ exp(2W)))`` — good;
* then ``sample()``/``rsample()`` falls through to
  ``_base_distribution.rsample()``, drawing FRESH i.i.d. noise from that
  marginal each call;
* never invokes the ``get_noise()`` method that would actually compute
  ``phi @ epsilon`` — it is dead code;
* calls ``sample_weights()`` once in ``__init__``, freezing ``epsilon`` to a
  value that is then never used anyway.

Net effect: ``a_t = mu(s_t) + xi_t * sqrt(phi(s_t)^2 @ exp(2W))`` with
``xi_t ~ N(0, I)`` i.i.d.. That is *exactly* a heteroscedastic Gaussian whose
sigma depends on the state through the ``phi^2 @ exp(2W)`` quadratic. There is
no temporal correlation in the action noise.

This script proves it with three checks, all under fixed RNG:

* **A.** Broken gSDE's ``mean`` and ``stddev`` equal those of the equivalent
  heteroscedastic Gaussian.
* **B.** Their ``sample()`` outputs are bit-identical.
* **C.** Across a smooth state trajectory, the noise from a *correct* gSDE has
  high lag-1 autocorrelation; the noise from the broken gSDE does not.

Run::

    python demo_main_gsde_is_heteroscedastic.py
"""

from __future__ import annotations

import torch
from torch.distributions import Distribution, Normal, constraints


# ============================================================================
# Verbatim copy of cae640292c (rsl_rl/modules/actor_critic.py).
# DO NOT EDIT — this is the implementation we are characterizing.
# ============================================================================
class BrokenGSDENoiseDistribution(Distribution):
    """Verbatim copy of ``GSDENoiseDistribution`` from commit ``cae640292c``."""

    has_rsample = True
    arg_constraints = {
        "mean_actions": constraints.real,
        "log_std": constraints.real,
        "latent_features": constraints.real,
    }
    _validate_args = False

    def __init__(
        self,
        action_dim,
        epsilon=1e-6,
        batch_shape=torch.Size(),
        event_shape=torch.Size(),
        validate_args=None,
    ):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self._base_distribution = None
        self._latent_features = None
        self._exploration_matrix = None
        self._exploration_matrices = None
        self._weights_distribution = None
        super().__init__(batch_shape, event_shape, validate_args)

    def _std_from_log_std(self, log_std):
        return torch.exp(log_std)

    def sample_weights(self, log_std, batch_size=1):
        std = self._std_from_log_std(log_std)
        weights_distribution = Normal(torch.zeros_like(std), std)
        self._weights_distribution = weights_distribution
        self._exploration_matrix = weights_distribution.rsample()
        self._exploration_matrices = weights_distribution.rsample((batch_size,))

    def proba_distribution(self, mean_actions, log_std, latent_features):
        self._latent_features = latent_features
        if self._exploration_matrix is not None:
            self._exploration_matrix = self._exploration_matrix.to(latent_features.device)
        if self._exploration_matrices is not None:
            self._exploration_matrices = self._exploration_matrices.to(latent_features.device)
        variance = torch.mm(latent_features**2, self._std_from_log_std(log_std) ** 2)
        self._base_distribution = Normal(mean_actions, torch.sqrt(variance + self.epsilon))
        return self

    def sample(self, sample_shape=torch.Size()):
        with torch.no_grad():
            return self.rsample(sample_shape)

    def rsample(self, sample_shape=torch.Size()):
        return self._base_distribution.rsample(sample_shape)

    def get_noise(self, latent_features):
        # Defined in the original, but never called by sample()/rsample().
        if (
            self._exploration_matrices is None
            or len(latent_features) == 1
            or len(latent_features) != len(self._exploration_matrices)
        ):
            return torch.mm(latent_features, self._exploration_matrix)
        latent_features = latent_features.unsqueeze(dim=1)
        noise = torch.bmm(latent_features, self._exploration_matrices)
        return noise.squeeze(dim=1)


# ============================================================================
# A correct gSDE for reference. Mirrors SB3's StateDependentNoiseDistribution:
# epsilon is sampled once per sample_weights() call and reused by sample()
# until the next sample_weights().
# ============================================================================
class CorrectGSDE:
    """Minimal correct gSDE; matches stable-baselines3 semantics."""

    def __init__(self, action_dim, epsilon=1e-6):
        self.action_dim = action_dim
        self.epsilon = epsilon
        self.exploration_matrix = None
        self.exploration_matrices = None
        self._mean = None
        self._log_std = None
        self._latent = None

    def sample_weights(self, log_std, batch_size=1):
        std = torch.exp(log_std)
        w_dist = Normal(torch.zeros_like(std), std)
        self.exploration_matrix = w_dist.rsample()
        self.exploration_matrices = w_dist.rsample((batch_size,))

    def update(self, mean, log_std, latent):
        self._mean, self._log_std, self._latent = mean, log_std, latent

    def _noise(self, latent):
        if self.exploration_matrices is None or len(latent) == 1 or len(latent) != len(self.exploration_matrices):
            return torch.mm(latent, self.exploration_matrix)
        return torch.bmm(latent.unsqueeze(1), self.exploration_matrices).squeeze(1)

    def sample(self):
        return self._mean + self._noise(self._latent)


# ============================================================================
# Heteroscedastic Gaussian with sigma(s) = sqrt(phi(s)^2 @ exp(2*log_std)).
# This is exactly the distribution BrokenGSDENoiseDistribution.sample() draws from.
# ============================================================================
class HeteroscedasticGaussian:
    """A diagonal Gaussian whose stddev is computed from latent features."""

    def __init__(self, epsilon=1e-6):
        self.epsilon = epsilon
        self._dist = None

    def update(self, mean, log_std, latent):
        variance = torch.mm(latent**2, torch.exp(log_std) ** 2)
        self._dist = Normal(mean, torch.sqrt(variance + self.epsilon))

    def sample(self):
        return self._dist.rsample()


def _banner(title):
    print("=" * 78)
    print(title)
    print("=" * 78)


def run_experiment_a():
    _banner("Experiment A: broken gSDE has the same mean/stddev as a")
    print("              heteroscedastic Gaussian with sigma = sqrt(phi^2 @ exp(2W)).")
    print()

    torch.manual_seed(0)
    batch, n_features, n_actions = 4, 8, 3
    mean = torch.randn(batch, n_actions)
    latent = torch.randn(batch, n_features)
    log_std = torch.randn(n_features, n_actions)

    broken = BrokenGSDENoiseDistribution(action_dim=n_actions)
    broken.proba_distribution(mean, log_std, latent)

    hetero = HeteroscedasticGaussian()
    hetero.update(mean, log_std, latent)

    mu_diff = (broken._base_distribution.mean - hetero._dist.mean).abs().max().item()
    std_diff = (broken._base_distribution.stddev - hetero._dist.stddev).abs().max().item()
    print(f"  max |delta mean|   = {mu_diff:.3e}")
    print(f"  max |delta stddev| = {std_diff:.3e}")
    passed = mu_diff == 0.0 and std_diff == 0.0
    print(f"  {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def run_experiment_b():
    _banner("Experiment B: under fixed RNG, broken gSDE and the heteroscedastic")
    print("              Gaussian produce bit-identical action samples.")
    print()

    torch.manual_seed(0)
    batch, n_features, n_actions = 4, 8, 3
    mean = torch.randn(batch, n_actions)
    latent = torch.randn(batch, n_features)
    log_std = torch.randn(n_features, n_actions)

    broken = BrokenGSDENoiseDistribution(action_dim=n_actions)
    broken.proba_distribution(mean, log_std, latent)

    hetero = HeteroscedasticGaussian()
    hetero.update(mean, log_std, latent)

    torch.manual_seed(42)
    a_broken = torch.stack([broken.sample() for _ in range(5)])

    torch.manual_seed(42)
    a_hetero = torch.stack([hetero.sample() for _ in range(5)])

    diff = (a_broken - a_hetero).abs().max().item()
    print(f"  max |delta action| over 5 samples x {batch} envs x {n_actions} dims: {diff:.3e}")
    passed = diff == 0.0
    print(f"  {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def _lag1_autocorr(noise):
    """Return mean lag-1 autocorrelation across action dims of a ``[T, A]`` tensor."""
    centered = noise - noise.mean(dim=0, keepdim=True)
    numerator = (centered[:-1] * centered[1:]).mean(dim=0)
    denominator = centered.var(dim=0, unbiased=False)
    return (numerator / denominator).mean().item()


def run_experiment_c():
    _banner("Experiment C: lag-1 noise autocorrelation along a smooth trajectory.")
    print("              Correct gSDE -> high autocorrelation (smooth noise);")
    print("              broken gSDE  -> ~0 autocorrelation (i.i.d. noise per step).")
    print()

    n_steps, n_envs, n_features, n_actions = 400, 1, 8, 3

    # Smooth feature trajectory: a random walk in feature space stands in for
    # the smooth phi(s_t) sequence you would see from a real continuous-state
    # rollout.
    gen = torch.Generator().manual_seed(1)
    deltas = torch.randn(n_steps, n_envs, n_features, generator=gen) * 0.05
    phis = deltas.cumsum(dim=0)  # [T, B, F]

    mean = torch.zeros(n_envs, n_actions)
    log_std = torch.full((n_features, n_actions), -1.5)  # std ~= 0.22

    torch.manual_seed(7)
    broken = BrokenGSDENoiseDistribution(action_dim=n_actions)
    broken.sample_weights(log_std, batch_size=n_envs)
    noises_broken = []
    for t in range(n_steps):
        broken.proba_distribution(mean, log_std, phis[t])
        noises_broken.append(broken.sample() - mean)
    noises_broken = torch.stack(noises_broken).squeeze(1)  # [T, A]

    torch.manual_seed(7)
    correct = CorrectGSDE(action_dim=n_actions)
    correct.sample_weights(log_std, batch_size=n_envs)
    noises_correct = []
    for t in range(n_steps):
        correct.update(mean, log_std, phis[t])
        noises_correct.append(correct.sample() - mean)
    noises_correct = torch.stack(noises_correct).squeeze(1)  # [T, A]

    r_broken = _lag1_autocorr(noises_broken)
    r_correct = _lag1_autocorr(noises_correct)
    print(f"  Lag-1 autocorrelation, broken gSDE:  {r_broken:+.4f}")
    print(f"  Lag-1 autocorrelation, correct gSDE: {r_correct:+.4f}")
    passed = r_correct > 0.5 and abs(r_broken) < 0.2
    print(f"  {'PASS' if passed else 'FAIL'}")
    print()
    return passed


def main():
    a = run_experiment_a()
    b = run_experiment_b()
    c = run_experiment_c()
    _banner("Summary")
    print(f"  A: {'PASS' if a else 'FAIL'} - broken gSDE has the same marginal as a heteroscedastic Gaussian")
    print(f"  B: {'PASS' if b else 'FAIL'} - broken gSDE samples are bit-identical to that heteroscedastic Gaussian")
    print(f"  C: {'PASS' if c else 'FAIL'} - broken gSDE noise is temporally uncorrelated; correct gSDE is correlated")
    print()
    if a and b and c:
        print("Conclusion: the cae640292c GSDENoiseDistribution is functionally identical")
        print("to a heteroscedastic Gaussian. The state-dependent exploration that gSDE is")
        print("supposed to provide (temporally correlated noise via fixed epsilon) is absent.")


if __name__ == "__main__":
    main()
