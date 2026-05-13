# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for distribution modules."""

import math
import numpy as np
import torch

from rsl_rl.modules import MLP
from rsl_rl.modules.distribution import (
    GaussianDistribution,
    GSDEGaussianDistribution,
    HeteroscedasticGaussianDistribution,
)


class TestGaussianDistribution:
    """Tests for ``GaussianDistribution``."""

    def test_log_prob_standard_normal(self) -> None:
        """log_prob at the mean of N(0,1) should equal -0.5*log(2*pi) per dimension, summed."""
        dim = 4
        dist = GaussianDistribution(output_dim=dim, init_std=1.0, std_type="scalar")
        mean = torch.zeros(1, dim)
        dist.update(mean)

        log_p = dist.log_prob(torch.zeros(1, dim))
        expected = -0.5 * math.log(2 * math.pi) * dim
        assert torch.allclose(log_p, torch.tensor([expected]), atol=1e-5)

    def test_log_prob_nonzero_mean(self) -> None:
        """log_prob should decrease as the sample moves away from the mean."""
        dist = GaussianDistribution(output_dim=2, init_std=1.0, std_type="scalar")
        mean = torch.tensor([[3.0, 3.0]])
        dist.update(mean)

        lp_at_mean = dist.log_prob(mean)
        lp_far = dist.log_prob(mean + 5.0)
        assert lp_at_mean > lp_far, "log_prob should be higher at the mean"

    def test_entropy_analytical(self) -> None:
        """Entropy should match the analytical formula 0.5 * sum(log(2*pi*e*std^2))."""
        dim = 3
        std_val = 2.0
        dist = GaussianDistribution(output_dim=dim, init_std=std_val, std_type="scalar")
        dist.update(torch.zeros(1, dim))

        expected = 0.5 * dim * math.log(2 * math.pi * math.e * std_val**2)
        assert torch.allclose(dist.entropy, torch.tensor([expected]), atol=1e-4)

    def test_kl_divergence_analytical(self) -> None:
        """KL(N(0,1) || N(mu,sigma)) should match the closed-form KL for univariate Gaussians."""
        dist = GaussianDistribution(output_dim=1, init_std=1.0, std_type="scalar")
        mu_old, sigma_old = 0.0, 1.0
        mu_new, sigma_new = 1.0, 2.0

        old_params = (torch.tensor([[mu_old]]), torch.tensor([[sigma_old]]))
        new_params = (torch.tensor([[mu_new]]), torch.tensor([[sigma_new]]))

        kl = dist.kl_divergence(old_params, new_params)

        # Analytical KL: log(s2/s1) + (s1^2 + (m1-m2)^2) / (2*s2^2) - 0.5
        expected = math.log(sigma_new / sigma_old) + (sigma_old**2 + (mu_old - mu_new) ** 2) / (2 * sigma_new**2) - 0.5
        assert torch.allclose(kl, torch.tensor([expected]), atol=1e-5)

    def test_kl_divergence_identical_is_zero(self) -> None:
        """KL(p || p) should be zero."""
        dist = GaussianDistribution(output_dim=4, init_std=1.5, std_type="scalar")
        params = (torch.zeros(1, 4), torch.full((1, 4), 1.5))
        kl = dist.kl_divergence(params, params)
        assert torch.allclose(kl, torch.zeros(1), atol=1e-6)

    def test_scalar_vs_log_std_equivalence(self) -> None:
        """Scalar and log parameterizations should give identical results for the same effective std."""
        dim = 3
        std_val = 1.5
        dist_scalar = GaussianDistribution(output_dim=dim, init_std=std_val, std_type="scalar")
        dist_log = GaussianDistribution(output_dim=dim, init_std=std_val, std_type="log")

        mean = torch.randn(2, dim)
        dist_scalar.update(mean)
        dist_log.update(mean)

        sample_point = torch.randn(2, dim)

        assert torch.allclose(dist_scalar.log_prob(sample_point), dist_log.log_prob(sample_point), atol=1e-5)
        assert torch.allclose(dist_scalar.entropy, dist_log.entropy, atol=1e-5)

    def test_log_prob_gradient_flows_to_mean(self) -> None:
        """log_prob should allow gradient flow back to the distribution mean."""
        dim = 3
        dist = GaussianDistribution(output_dim=dim, init_std=1.0, std_type="scalar")
        mean = torch.randn(1, dim, requires_grad=True)
        dist.update(mean)

        sample = dist.sample().detach()
        log_p = dist.log_prob(sample)
        log_p.sum().backward()
        assert mean.grad is not None, "Gradient should flow from log_prob to mean"
        assert not torch.all(mean.grad == 0), "Gradient should be non-zero"

    def test_std_clamped_to_range_scalar(self) -> None:
        """The std should be clamped to both bounds of std_range for std_type='scalar'."""
        dim = 2
        std_range = (0.1, 2.0)
        # Above the upper bound.
        dist_high = GaussianDistribution(output_dim=dim, init_std=10.0, std_type="scalar", std_range=std_range)
        dist_high.update(torch.zeros(1, dim))
        assert torch.allclose(dist_high.std, torch.full((1, dim), std_range[1]), atol=1e-6)
        # Below the lower bound.
        dist_low = GaussianDistribution(output_dim=dim, init_std=0.01, std_type="scalar", std_range=std_range)
        dist_low.update(torch.zeros(1, dim))
        assert torch.allclose(dist_low.std, torch.full((1, dim), std_range[0]), atol=1e-6)

    def test_std_clamped_to_range_log(self) -> None:
        """The std should be clamped to both bounds of std_range for std_type='log'."""
        dim = 2
        std_range = (0.1, 2.0)
        # Above the upper bound.
        dist_high = GaussianDistribution(output_dim=dim, init_std=10.0, std_type="log", std_range=std_range)
        dist_high.update(torch.zeros(1, dim))
        assert torch.allclose(dist_high.std, torch.full((1, dim), std_range[1]), atol=1e-6)
        # Below the lower bound.
        dist_low = GaussianDistribution(output_dim=dim, init_std=0.01, std_type="log", std_range=std_range)
        dist_low.update(torch.zeros(1, dim))
        assert torch.allclose(dist_low.std, torch.full((1, dim), std_range[0]), atol=1e-6)

    def test_std_range_min_floor(self) -> None:
        """The minimum of std_range should be floored to 1e-6 for numerical stability."""
        dist = GaussianDistribution(output_dim=2, init_std=1.0, std_type="scalar", std_range=(0.0, 10.0))
        assert dist.std_range[0] == 1e-6

    def test_learn_std_scalar(self) -> None:
        """learn_std should control whether the scalar std parameter is learnable."""
        dim = 3
        init_std = 0.7
        # learn_std=True: parameter is trainable and receives non-zero gradient.
        dist_learn = GaussianDistribution(output_dim=dim, init_std=init_std, std_type="scalar", learn_std=True)
        assert dist_learn.std_param.requires_grad is True
        dist_learn.update(torch.randn(2, dim))
        sample = dist_learn.sample().detach()
        dist_learn.log_prob(sample).sum().backward()
        assert dist_learn.std_param.grad is not None and not torch.all(dist_learn.std_param.grad == 0)
        # learn_std=False: parameter is frozen and receives no gradient.
        dist_fixed = GaussianDistribution(output_dim=dim, init_std=init_std, std_type="scalar", learn_std=False)
        assert dist_fixed.std_param.requires_grad is False
        mean = torch.randn(2, dim, requires_grad=True)
        dist_fixed.update(mean)
        sample = dist_fixed.sample().detach()
        dist_fixed.log_prob(sample).sum().backward()
        assert dist_fixed.std_param.grad is None, "Non-learnable std should not receive gradients"
        assert torch.allclose(dist_fixed.std_param, torch.full((dim,), init_std), atol=1e-6)

    def test_learn_std_log(self) -> None:
        """learn_std should control whether the log std parameter is learnable."""
        dim = 3
        init_std = 0.7
        # learn_std=True: parameter is trainable and receives non-zero gradient.
        dist_learn = GaussianDistribution(output_dim=dim, init_std=init_std, std_type="log", learn_std=True)
        assert dist_learn.log_std_param.requires_grad is True
        dist_learn.update(torch.randn(2, dim))
        sample = dist_learn.sample().detach()
        dist_learn.log_prob(sample).sum().backward()
        assert dist_learn.log_std_param.grad is not None and not torch.all(dist_learn.log_std_param.grad == 0)
        # learn_std=False: parameter is frozen and receives no gradient.
        dist_fixed = GaussianDistribution(output_dim=dim, init_std=init_std, std_type="log", learn_std=False)
        assert dist_fixed.log_std_param.requires_grad is False
        mean = torch.randn(2, dim, requires_grad=True)
        dist_fixed.update(mean)
        sample = dist_fixed.sample().detach()
        dist_fixed.log_prob(sample).sum().backward()
        assert dist_fixed.log_std_param.grad is None, "Non-learnable log std should not receive gradients"
        assert torch.allclose(dist_fixed.log_std_param, torch.log(torch.full((dim,), init_std)), atol=1e-6)


class TestHeteroscedasticGaussianDistribution:
    """Tests for ``HeteroscedasticGaussianDistribution``."""

    def test_update_splits_mean_and_std(self) -> None:
        """update() should parse MLP output into separate mean and std."""
        dim = 4
        dist = HeteroscedasticGaussianDistribution(output_dim=dim, init_std=1.0, std_type="scalar")

        mean_val = torch.randn(2, dim)
        std_val = torch.abs(torch.randn(2, dim)) + 0.1
        mlp_output = torch.stack([mean_val, std_val], dim=-2)

        dist.update(mlp_output)
        assert torch.allclose(dist.mean, mean_val, atol=1e-6)
        assert torch.allclose(dist.std, std_val, atol=1e-6)

    def test_deterministic_output_returns_mean(self) -> None:
        """deterministic_output() should extract the mean from the MLP output."""
        dim = 3
        dist = HeteroscedasticGaussianDistribution(output_dim=dim, init_std=1.0, std_type="scalar")

        mean_val = torch.tensor([[1.0, 2.0, 3.0]])
        std_val = torch.tensor([[0.5, 0.5, 0.5]])
        mlp_output = torch.stack([mean_val, std_val], dim=-2)

        result = dist.deterministic_output(mlp_output)
        assert torch.allclose(result, mean_val)

    def test_log_std_parameterization(self) -> None:
        """With std_type='log', the second slice should be treated as log(std)."""
        dim = 2
        dist = HeteroscedasticGaussianDistribution(output_dim=dim, init_std=1.0, std_type="log")

        mean_val = torch.zeros(1, dim)
        log_std_val = torch.zeros(1, dim)  # log(1) = 0, so std = 1
        mlp_output = torch.stack([mean_val, log_std_val], dim=-2)

        dist.update(mlp_output)
        assert torch.allclose(dist.std, torch.ones(1, dim), atol=1e-6)

    def test_input_dim_is_pair(self) -> None:
        """input_dim should be [2, output_dim] to accommodate mean and std."""
        dim = 5
        dist = HeteroscedasticGaussianDistribution(output_dim=dim)
        assert dist.input_dim == [2, dim]

    def test_std_clamped_to_range_scalar(self) -> None:
        """The state-dependent std should be clamped to std_range for std_type='scalar'."""
        dim = 3
        dist = HeteroscedasticGaussianDistribution(
            output_dim=dim, init_std=1.0, std_type="scalar", std_range=(0.1, 2.0)
        )

        mean_val = torch.zeros(1, dim)
        std_val = torch.tensor([[10.0, 0.01, 1.0]])  # above, below, inside
        mlp_output = torch.stack([mean_val, std_val], dim=-2)

        dist.update(mlp_output)
        expected = torch.tensor([[2.0, 0.1, 1.0]])
        assert torch.allclose(dist.std, expected, atol=1e-6)

    def test_std_clamped_to_range_log(self) -> None:
        """The state-dependent std should be clamped to std_range for std_type='log'."""
        dim = 3
        dist = HeteroscedasticGaussianDistribution(output_dim=dim, init_std=1.0, std_type="log", std_range=(0.1, 2.0))

        mean_val = torch.zeros(1, dim)
        # log values: log(10) above, log(0.01) below, log(1) inside
        log_std_val = torch.log(torch.tensor([[10.0, 0.01, 1.0]]))
        mlp_output = torch.stack([mean_val, log_std_val], dim=-2)

        dist.update(mlp_output)
        expected = torch.tensor([[2.0, 0.1, 1.0]])
        assert torch.allclose(dist.std, expected, atol=1e-6)

    def test_std_range_min_floor(self) -> None:
        """The minimum of std_range should be floored to 1e-6 for numerical stability."""
        dist = HeteroscedasticGaussianDistribution(output_dim=2, init_std=1.0, std_type="scalar", std_range=(0.0, 10.0))
        assert dist.std_range[0] == 1e-6


def _build_gsde(
    output_dim: int = 3,
    input_dim: int = 6,
    hidden_dims: tuple[int, ...] = (8, 8),
    **gsde_kwargs: object,
) -> tuple[GSDEGaussianDistribution, MLP]:
    """Build an ``MLP`` paired with an initialised ``GSDEGaussianDistribution``."""
    dist = GSDEGaussianDistribution(output_dim=output_dim, **gsde_kwargs)
    mlp = MLP(input_dim, output_dim, hidden_dims=hidden_dims)
    dist.init_mlp_weights(mlp)
    return dist, mlp


def _forward_with_latent(mlp: MLP, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Run ``mlp`` and return ``(mean, latent_sde)`` via :meth:`MLP.forward_with_features`."""
    return mlp.forward_with_features(x)


class TestGSDEGaussianDistribution:
    """Tests for ``GSDEGaussianDistribution``."""

    def test_requires_latent_sde(self) -> None:
        """The class must flag itself as needing the penultimate MLP activation."""
        dist = GSDEGaussianDistribution(output_dim=3)
        assert dist.requires_latent_sde is True

    def test_input_dim_is_action_dim(self) -> None:
        """MLP outputs only the mean; std is owned by the distribution."""
        dim = 5
        dist = GSDEGaussianDistribution(output_dim=dim)
        assert dist.input_dim == dim

    def test_log_std_shape_full_std(self) -> None:
        """With full_std=True, log_std_param has shape [latent_sde_dim, output_dim]."""
        dist, _ = _build_gsde(output_dim=3, hidden_dims=(16, 12), full_std=True)
        assert dist.latent_sde_dim == 12
        assert dist.log_std_param.shape == (12, 3)

    def test_log_std_shape_reduced(self) -> None:
        """With full_std=False, log_std_param has shape [latent_sde_dim, 1]."""
        dist, _ = _build_gsde(output_dim=3, hidden_dims=(16, 12), full_std=False)
        assert dist.log_std_param.shape == (12, 1)

    def test_marginal_variance_matches_quadratic_formula(self) -> None:
        """The marginal std must equal sqrt(phi^2 @ sigma^2 + epsilon) exactly."""
        torch.manual_seed(0)
        dist, mlp = _build_gsde(output_dim=3, hidden_dims=(8, 8), init_std=0.4, full_std=True)
        dist.sample_weights(batch_size=4)

        x = torch.randn(4, 6)
        mean, latent_sde = _forward_with_latent(mlp, x)
        dist.update(mean, latent_sde=latent_sde)

        sigma = torch.exp(dist.log_std_param)
        expected_variance = torch.mm(latent_sde**2, sigma**2)
        expected_std = torch.sqrt(expected_variance + dist.epsilon)
        assert torch.allclose(dist.std, expected_std, atol=1e-6)

    def test_temporal_correlation_property(self) -> None:
        """Two sample() calls with the same state and no re-sample_weights give identical actions.

        This is the core property gSDE provides and that ``main``'s implementation lacks.
        """
        torch.manual_seed(0)
        dist, mlp = _build_gsde(output_dim=3, hidden_dims=(8, 8))
        dist.sample_weights(batch_size=4)

        x = torch.randn(4, 6)
        mean, latent_sde = _forward_with_latent(mlp, x)
        dist.update(mean, latent_sde=latent_sde)
        a1 = dist.sample()
        a2 = dist.sample()
        assert torch.equal(a1, a2), "gSDE must return identical actions for identical state with fixed epsilon"

    def test_resample_weights_changes_noise(self) -> None:
        """After sample_weights(), a new epsilon is drawn and the noise changes."""
        torch.manual_seed(0)
        dist, mlp = _build_gsde(output_dim=3, hidden_dims=(8, 8))
        dist.sample_weights(batch_size=4)

        x = torch.randn(4, 6)
        mean, latent_sde = _forward_with_latent(mlp, x)
        dist.update(mean, latent_sde=latent_sde)
        a_before = dist.sample()

        dist.sample_weights(batch_size=4)
        dist.update(mean, latent_sde=latent_sde)
        a_after = dist.sample()
        assert not torch.equal(a_before, a_after)

    def test_per_env_epsilon_independence(self) -> None:
        """Per-env epsilon yields different noise across envs even with identical features."""
        torch.manual_seed(0)
        dist, _ = _build_gsde(output_dim=3, hidden_dims=(8, 8))
        n_envs = 5
        dist.sample_weights(batch_size=n_envs)

        # Identical latent across envs — any difference in noise must come from per-env epsilon.
        same_latent = torch.randn(1, dist.latent_sde_dim).expand(n_envs, dist.latent_sde_dim).contiguous()
        same_mean = torch.zeros(n_envs, 3)
        dist.update(same_mean, latent_sde=same_latent)
        actions = dist.sample()
        # No two rows should be identical: per-env epsilon differs for each row.
        for i in range(n_envs):
            for j in range(i + 1, n_envs):
                assert not torch.equal(actions[i], actions[j])

    def test_sample_without_sample_weights_raises(self) -> None:
        """sample() before sample_weights() must raise — runner is responsible for the reset."""
        torch.manual_seed(0)
        dist, mlp = _build_gsde(output_dim=3, hidden_dims=(8, 8))

        x = torch.randn(4, 6)
        mean, latent_sde = _forward_with_latent(mlp, x)
        dist.update(mean, latent_sde=latent_sde)
        try:
            dist.sample()
        except RuntimeError:
            return
        raise AssertionError("sample() should have raised RuntimeError because epsilon was never sampled.")

    def test_learn_features_false_blocks_feature_gradient(self) -> None:
        """With learn_features=False the variance term is detached from the feature backbone."""
        torch.manual_seed(0)
        dist, mlp = _build_gsde(output_dim=3, hidden_dims=(8, 8), learn_features=False)
        dist.sample_weights(batch_size=4)

        x = torch.randn(4, 6)
        mean, latent_sde = _forward_with_latent(mlp, x)
        latent_sde_leaf = latent_sde.detach().clone().requires_grad_(True)
        dist.update(mean.detach(), latent_sde=latent_sde_leaf)

        # Loss depending only on the marginal std (so any gradient on latent_sde_leaf would come
        # exclusively from the variance term, which is detached when learn_features=False).
        dist.std.sum().backward()
        assert latent_sde_leaf.grad is None or torch.all(latent_sde_leaf.grad == 0)

    def test_learn_features_true_allows_feature_gradient(self) -> None:
        """With learn_features=True, the variance backprops into the features."""
        torch.manual_seed(0)
        dist, mlp = _build_gsde(output_dim=3, hidden_dims=(8, 8), learn_features=True)
        dist.sample_weights(batch_size=4)

        x = torch.randn(4, 6)
        mean, latent_sde = _forward_with_latent(mlp, x)
        latent_sde_leaf = latent_sde.detach().clone().requires_grad_(True)
        dist.update(mean.detach(), latent_sde=latent_sde_leaf)

        dist.std.sum().backward()
        assert latent_sde_leaf.grad is not None and not torch.all(latent_sde_leaf.grad == 0)

    def test_kl_divergence_identical_is_zero(self) -> None:
        """KL(p || p) under the marginal Gaussian should be zero."""
        torch.manual_seed(0)
        dist, mlp = _build_gsde(output_dim=4, hidden_dims=(8, 8))
        dist.sample_weights(batch_size=2)

        x = torch.randn(2, 6)
        mean, latent_sde = _forward_with_latent(mlp, x)
        dist.update(mean, latent_sde=latent_sde)
        kl = dist.kl_divergence(dist.params, dist.params)
        assert torch.allclose(kl, torch.zeros_like(kl), atol=1e-6)

    def test_log_std_param_gets_gradient(self) -> None:
        """log_prob must backprop into log_std_param so PPO can learn the variance."""
        torch.manual_seed(0)
        dist, mlp = _build_gsde(output_dim=3, hidden_dims=(8, 8))
        dist.sample_weights(batch_size=4)

        x = torch.randn(4, 6)
        mean, latent_sde = _forward_with_latent(mlp, x)
        dist.update(mean, latent_sde=latent_sde)

        # Evaluate log_prob at a fixed off-mean point (PPO uses the stored action, not a fresh sample,
        # so the reparam-trick cancellation does not apply).
        log_p = dist.log_prob(torch.zeros(4, 3))
        log_p.sum().backward()
        assert dist.log_std_param.grad is not None
        assert not torch.all(dist.log_std_param.grad == 0)

    def test_std_range_clamping(self) -> None:
        """The implied scalar std should be clamped to std_range."""
        dist, _ = _build_gsde(output_dim=2, hidden_dims=(4, 4), init_std=0.5, std_range=(0.1, 0.4))
        # Force the param outside the upper bound — the clamp should bring std down to 0.4.
        with torch.no_grad():
            dist.log_std_param.fill_(float(np.log(10.0)))
        std = dist._get_std()
        assert torch.allclose(std, torch.full_like(std, 0.4), atol=1e-6)
        # And below the lower bound — clamp brings it up to 0.1.
        with torch.no_grad():
            dist.log_std_param.fill_(float(np.log(1e-9)))
        std = dist._get_std()
        assert torch.allclose(std, torch.full_like(std, 0.1), atol=1e-6)

    def test_use_expln_yields_different_std(self) -> None:
        """use_expln=True must give a different std curve than plain exp for positive log_std."""
        torch.manual_seed(0)
        dist_exp, _ = _build_gsde(output_dim=3, hidden_dims=(8,), use_expln=False)
        dist_expln, _ = _build_gsde(output_dim=3, hidden_dims=(8,), use_expln=True)
        # Make log_std positive so the two branches diverge.
        with torch.no_grad():
            dist_exp.log_std_param.fill_(1.5)
            dist_expln.log_std_param.fill_(1.5)
        assert not torch.allclose(dist_exp._get_std(), dist_expln._get_std())

    def test_exploration_matrix_not_persisted(self) -> None:
        """The exploration tensor must not be serialized in state_dict; it is transient by design."""
        dist, _ = _build_gsde(output_dim=3, hidden_dims=(8, 8))
        dist.sample_weights(batch_size=4)
        state = dist.state_dict()
        assert dist.exploration_matrix is not None  # actually drawn...
        # ...but not under the canonical names in state_dict.
        for key in state:
            assert "exploration_matrix" not in key
            assert "exploration_matrices" not in key

    def test_log_std_persists_through_state_dict_roundtrip(self) -> None:
        """log_std_param must round-trip cleanly through state_dict save/load."""
        dist_a, _ = _build_gsde(output_dim=3, hidden_dims=(8, 8))
        dist_b, _ = _build_gsde(output_dim=3, hidden_dims=(8, 8))
        with torch.no_grad():
            dist_a.log_std_param.copy_(torch.randn_like(dist_a.log_std_param))
        dist_b.load_state_dict(dist_a.state_dict())
        assert torch.equal(dist_a.log_std_param, dist_b.log_std_param)
