# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class Distribution(nn.Module):
    """Base class for distribution modules.

    Distribution modules encapsulate the stochastic output of a neural model. They define the output structure expected
    from the MLP, manage learnable distribution parameters, and provide methods for sampling, log probability
    computation, and entropy calculation.

    Subclasses must implement all abstract methods and properties to define a specific distribution type.
    """

    requires_latent_sde: bool = False
    """Whether :meth:`update` needs the penultimate MLP activation as ``latent_sde``.

    Set to ``True`` by :class:`GSDEGaussianDistribution`. Owning models check this flag and, when ``True``, route the
    forward pass through :meth:`rsl_rl.modules.MLP.forward_with_features` to obtain both the MLP output and the
    penultimate activation, then pass both into :meth:`update`.
    """

    def __init__(self, output_dim: int) -> None:
        """Initialize the distribution module.

        Args:
            output_dim: Dimension of the action/output space.
        """
        super().__init__()
        self.output_dim = output_dim

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the distribution parameters given the MLP output.

        Args:
            mlp_output: Raw output from the MLP.
        """
        raise NotImplementedError

    def sample(self) -> torch.Tensor:
        """Sample from the distribution.

        Returns:
            Sampled values.
        """
        raise NotImplementedError

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Extract the deterministic (mean) output from the raw MLP output.

        Args:
            mlp_output: Raw output from the MLP.

        Returns:
            The deterministic output (typically the distribution mean).
        """
        raise NotImplementedError

    def as_deterministic_output_module(self) -> nn.Module:
        """Return an export-friendly module that extracts the deterministic output from the MLP output."""
        raise NotImplementedError

    @property
    def input_dim(self) -> int | list[int]:
        """Return the input dimension required by the distribution."""
        raise NotImplementedError

    @property
    def mean(self) -> torch.Tensor:
        """Return the mean of the distribution."""
        raise NotImplementedError

    @property
    def std(self) -> torch.Tensor:
        """Return the standard deviation (or spread measure) of the distribution."""
        raise NotImplementedError

    @property
    def entropy(self) -> torch.Tensor:
        """Return the entropy of the distribution, summed over the last dimension."""
        raise NotImplementedError

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Return the distribution parameters as a tuple of tensors.

        These are the distribution-specific parameters needed to reconstruct the distribution (e.g., mean and std for
        Gaussian, alpha and beta for Beta). They are stored during rollouts and used for KL divergence computation.
        """
        raise NotImplementedError

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute the log probability of the given outputs, summed over the last dimension.

        Args:
            outputs: Values to compute the log probability for.

        Returns:
            Log probability summed over the last dimension.
        """
        raise NotImplementedError

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute the KL divergence KL(old || new) between two distributions of this type.

        The KL divergence measures how the old distribution diverges from the new distribution.
        This is used for adaptive learning rate scheduling in policy optimization.

        Args:
            old_params: Parameters of the old distribution (as returned by :attr:`params`).
            new_params: Parameters of the new distribution (as returned by :attr:`params`).

        Returns:
            KL divergence summed over the last dimension.
        """
        raise NotImplementedError

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        """Initialize distribution-specific weights in the MLP.

        This is called after MLP creation to set up any special weight initialization
        required by the distribution (e.g., initializing std head weights).

        Args:
            mlp: The MLP module whose weights may need initialization.
        """
        pass


class GaussianDistribution(Distribution):
    """Gaussian distribution module with state-independent standard deviation.

    This distribution parameterizes stochastic outputs using a multivariate Gaussian with diagonal covariance. The
    standard deviation can be a learnable parameter or a constant. It can be parameterized in either "scalar" space or
    "log" space and is clamped to a specified range.

    .. note::
        If the standard deviation type is set to "log", the provided arguments are still interpreted in scalar space,
        and converted to log space internally.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_range: tuple[float, float] = (1e-6, 1e6),
        std_type: str = "scalar",
        learn_std: bool = True,
    ) -> None:
        """Initialize the Gaussian distribution module.

        Args:
            output_dim: Dimension of the action/output space.
            init_std: Initial standard deviation.
            std_range: Range for the standard deviation. Should be a tuple of (min, max) values for clamping.
            std_type: Parameterization of the standard deviation: "scalar" or "log".
            learn_std: Whether the standard deviation should be learnable. If False, it will be fixed to `init_std`.
        """
        super().__init__(output_dim)
        self.std_type = std_type

        # Learnable std parameters
        if std_type == "scalar":
            self.std_param = nn.Parameter(init_std * torch.ones(output_dim), requires_grad=learn_std)
        elif std_type == "log":
            self.log_std_param = nn.Parameter(torch.log(init_std * torch.ones(output_dim)), requires_grad=learn_std)
        else:
            raise ValueError(f"Unknown standard deviation type: {std_type}. Should be 'scalar' or 'log'.")

        # Clamp the std range to ensure numerical stability and store log space range if needed
        self.std_range = list(std_range)
        self.std_range[0] = max(self.std_range[0], 1e-6)  # Avoid zero std for numerical stability
        self.log_std_range = [float(np.log(self.std_range[0])), float(np.log(self.std_range[1]))]

        # Internal torch distribution (populated by update())
        self._distribution: Normal | None = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the Gaussian distribution from MLP output."""
        mean = mlp_output
        if self.std_type == "scalar":
            std = self.std_param.clamp(self.std_range[0], self.std_range[1])
        elif self.std_type == "log":
            log_std = self.log_std_param.clamp(self.log_std_range[0], self.log_std_range[1])
            std = torch.exp(log_std)
        self._distribution = Normal(mean, std)

    def sample(self) -> torch.Tensor:
        """Sample from the Gaussian distribution."""
        return self._distribution.sample()  # type: ignore

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Extract the mean from the MLP output."""
        return mlp_output

    def as_deterministic_output_module(self) -> nn.Module:
        """Return an export-friendly module that extracts the mean from the MLP output."""
        return _IdentityDeterministicOutput()

    @property
    def input_dim(self) -> int:
        """Return the input dimension required by the distribution."""
        return self.output_dim

    @property
    def mean(self) -> torch.Tensor:
        """Return the mean of the Gaussian distribution."""
        return self._distribution.mean  # type: ignore

    @property
    def std(self) -> torch.Tensor:
        """Return the standard deviation of the Gaussian distribution."""
        return self._distribution.stddev  # type: ignore

    @property
    def entropy(self) -> torch.Tensor:
        """Return the entropy of the Gaussian distribution, summed over the last dimension."""
        return self._distribution.entropy().sum(dim=-1)  # type: ignore

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Return (mean, std) of the current Gaussian distribution."""
        return (self.mean, self.std)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute the log probability under the Gaussian, summed over the last dimension."""
        return self._distribution.log_prob(outputs).sum(dim=-1)  # type: ignore

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        """Compute KL(old || new) between two Gaussian distributions using torch.distributions."""
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        old_dist = Normal(old_mean, old_std)
        new_dist = Normal(new_mean, new_std)
        return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)


class HeteroscedasticGaussianDistribution(GaussianDistribution):
    """Gaussian distribution module with state-dependent standard deviation.

    This distribution parameterizes stochastic outputs using a multivariate Gaussian with diagonal covariance. The
    standard deviation is output by the MLP alongside the mean, making it state-dependent. It can be parameterized in
    either "scalar" space or "log" space, and is clamped to a specified range.

    .. note::
        If the standard deviation type is set to "log", the provided arguments are still interpreted in scalar space,
        and converted to log space internally.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_range: tuple[float, float] = (1e-6, 1e6),
        std_type: str = "scalar",
    ) -> None:
        """Initialize the heteroscedastic Gaussian distribution module.

        Args:
            output_dim: Dimension of the action/output space.
            init_std: Initial standard deviation (used to initialize the MLP's std head bias).
            std_range: Range for the standard deviation. Should be a tuple of (min, max) values for clamping.
            std_type: Parameterization of the standard deviation: "scalar" or "log".
        """
        # Skip GaussianDistribution.__init__ to avoid creating unnecessary learnable std parameters.
        Distribution.__init__(self, output_dim)
        self.std_type = std_type
        self.init_std = init_std

        if std_type not in ("scalar", "log"):
            raise ValueError(f"Unknown standard deviation type: {std_type}. Should be 'scalar' or 'log'.")

        # Clamp the std range to ensure numerical stability and store log space range if needed
        self.std_range = list(std_range)
        self.std_range[0] = max(self.std_range[0], 1e-6)  # Avoid zero std for numerical stability
        self.log_std_range = [float(np.log(self.std_range[0])), float(np.log(self.std_range[1]))]

        # Internal torch distribution (populated by update())
        self._distribution: Normal | None = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the Gaussian distribution from MLP output."""
        if self.std_type == "scalar":
            mean, std = torch.unbind(mlp_output, dim=-2)
            std = torch.clamp(std, self.std_range[0], self.std_range[1])
        elif self.std_type == "log":
            mean, log_std = torch.unbind(mlp_output, dim=-2)
            log_std = torch.clamp(log_std, self.log_std_range[0], self.log_std_range[1])
            std = torch.exp(log_std)
        self._distribution = Normal(mean, std)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Extract the mean from the MLP output (first slice of the second-to-last dimension)."""
        return mlp_output[..., 0, :]

    def as_deterministic_output_module(self) -> nn.Module:
        """Return export-friendly module that extracts the mean from the MLP output."""
        return _MeanSliceDeterministicOutput()

    @property
    def input_dim(self) -> list[int]:
        """Return the input dimension required by the distribution.

        The MLP must output a tensor of shape ``[..., 2, output_dim]`` where the first slice along the second-to-last
        dimension is the mean and the second is the standard deviation (or log standard deviation).
        """
        return [2, self.output_dim]

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        """Initialize the std head weights in the MLP."""
        # Initialize weights and biases for the std portion of the last layer
        torch.nn.init.zeros_(mlp[-2].weight[self.output_dim :])  # type: ignore
        if self.std_type == "scalar":
            torch.nn.init.constant_(mlp[-2].bias[self.output_dim :], self.init_std)  # type: ignore
        elif self.std_type == "log":
            init_std_log = torch.log(torch.tensor(self.init_std + 1e-7))
            torch.nn.init.constant_(mlp[-2].bias[self.output_dim :], init_std_log)  # type: ignore


class GSDEGaussianDistribution(Distribution):
    r"""Gaussian distribution module with generalized State Dependent Exploration (gSDE).

    Reference: Raffin et al., "Smooth Exploration for Robotic Reinforcement Learning",
    `arXiv:2005.05719 <https://arxiv.org/abs/2005.05719>`_.

    Exploration noise is parameterized as :math:`\phi(s) \cdot \varepsilon`, where
    :math:`\phi(s)` is the penultimate activation of the policy MLP (the ``latent_sde``) and
    :math:`\varepsilon \sim \mathcal{N}(0, W^2)` is sampled once via :meth:`sample_weights`
    and held fixed across many policy steps. With :math:`\varepsilon` fixed, the noise
    :math:`\phi(s) \cdot \varepsilon` varies smoothly along a trajectory, producing
    temporally correlated (rather than i.i.d.) action noise — the property gSDE is designed
    for in real-robot exploration.

    The action is sampled as :math:`a = \mu(s) + \phi(s) \cdot \varepsilon`. The matching
    marginal distribution, used for log-probability, entropy, and KL divergence, is
    :math:`\mathcal{N}(\mu(s), \operatorname{diag}(\phi(s)^2 \cdot \sigma^2))` where
    :math:`\sigma` is the elementwise std implied by :attr:`log_std_param`.

    The MLP output is the mean :math:`\mu(s)`; this distribution owns its own ``log_std_param``
    of shape ``[latent_sde_dim, output_dim]`` (or ``[latent_sde_dim, 1]`` when
    ``full_std=False``). The penultimate activation is obtained from the MLP via
    :meth:`rsl_rl.modules.MLP.forward_with_features` by owning models that set
    ``requires_latent_sde = True`` is observed.

    .. note::
        The initial marginal stddev at a state :math:`s` scales as
        :math:`\text{init\_std} \cdot \lVert \phi(s) \rVert`. Because :math:`\lVert
        \phi(s) \rVert` depends on the hidden width and activation, the default
        :math:`\text{init\_std} = 0.135` matches the SB3 default
        (:math:`\log\text{-init} = -2`) which yields reasonable initial exploration for
        typical post-activation features.
    """

    requires_latent_sde: bool = True

    def __init__(
        self,
        output_dim: int,
        init_std: float = 0.135,
        std_range: tuple[float, float] = (1e-6, 1e6),
        full_std: bool = True,
        learn_features: bool = False,
        use_expln: bool = False,
        epsilon: float = 1e-6,
    ) -> None:
        """Initialize the gSDE distribution module.

        Args:
            output_dim: Dimension of the action/output space.
            init_std: Initial value used to populate every entry of :attr:`log_std_param` in
                scalar space (converted to log space internally). See the class note for how
                this maps to the initial marginal stddev.
            std_range: Range used to clamp the implied scalar std before exponentiation.
            full_std: If ``True``, :attr:`log_std_param` has shape ``[latent_sde_dim,
                output_dim]``. If ``False``, it has shape ``[latent_sde_dim, 1]`` (broadcast
                across actions) which reduces parameter count when ``output_dim`` is large.
            learn_features: If ``False`` (default, matches gSDE paper and SB3), the
                ``latent_sde`` is detached before being used to compute the variance and
                noise so gSDE gradients do not flow into the policy backbone.
            use_expln: If ``True``, use the ``expln`` smooth-positive transform instead of
                ``exp`` to keep variance from growing too fast. See the gSDE paper.
            epsilon: Small constant added under the variance square-root for numerical
                stability.
        """
        super().__init__(output_dim)

        self.init_std = init_std
        self.full_std = full_std
        self.learn_features = learn_features
        self.use_expln = use_expln
        self.epsilon = epsilon

        # Clamp range (scalar space); also store the log-space range used to clamp log_std_param.
        std_range = (max(float(std_range[0]), 1e-6), float(std_range[1]))
        self.std_range = list(std_range)
        self.log_std_range = [float(np.log(std_range[0])), float(np.log(std_range[1]))]

        # log_std_param is created lazily in init_mlp_weights once the latent_sde dim is known.
        self.log_std_param: nn.Parameter | None = None
        self.latent_sde_dim: int | None = None

        # Exploration tensor epsilon. Plain attributes (not buffers) so they're not persisted
        # in state_dict; sample_weights places them on log_std_param's device. They're
        # re-sampled at the start of every rollout (and every sde_sample_freq env steps).
        self.exploration_matrix: torch.Tensor | None = None
        self.exploration_matrices: torch.Tensor | None = None

        # Mean and latent cached from the most recent update() call.
        self._mean: torch.Tensor | None = None
        self._latent_sde: torch.Tensor | None = None
        self._distribution: Normal | None = None

        # Disable args validation for speedup
        Normal.set_default_validate_args(False)

    @property
    def input_dim(self) -> int:
        r"""The MLP outputs the mean :math:`\mu(s)`; the std is owned by this distribution."""
        return self.output_dim

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        """Allocate :attr:`log_std_param` once the MLP's penultimate dimension is known.

        ``latent_sde_dim`` is read from the ``in_features`` of the last :class:`torch.nn.Linear`
        in ``mlp`` — that activation is the one passed to this distribution as ``latent_sde``.
        """
        last_linear: nn.Linear | None = None
        for module in mlp.modules():
            if isinstance(module, nn.Linear):
                last_linear = module
        if last_linear is None:
            raise ValueError("GSDEGaussianDistribution: MLP must contain at least one nn.Linear layer.")

        self.latent_sde_dim = int(last_linear.in_features)
        action_dim = self.output_dim if self.full_std else 1
        log_std_init = float(np.log(self.init_std))
        log_std = torch.full((self.latent_sde_dim, action_dim), log_std_init, device=last_linear.weight.device)
        self.log_std_param = nn.Parameter(log_std)

    def _get_std(self) -> torch.Tensor:
        """Return the elementwise std of :attr:`log_std_param`, broadcast to ``[latent_sde_dim, output_dim]``."""
        assert self.log_std_param is not None and self.latent_sde_dim is not None
        log_std = self.log_std_param.clamp(self.log_std_range[0], self.log_std_range[1])
        if self.use_expln:
            # Smooth positive transform: exp(x) for x <= 0, log1p(x)+1 for x > 0.
            below = torch.exp(log_std) * (log_std <= 0)
            safe_pos = log_std * (log_std > 0) + self.epsilon
            above = (torch.log1p(safe_pos) + 1.0) * (log_std > 0)
            std = below + above
        else:
            std = torch.exp(log_std)
        if not self.full_std:
            std = std.expand(self.latent_sde_dim, self.output_dim)
        return std

    def sample_weights(self, batch_size: int) -> None:
        r"""Re-sample the exploration tensor :math:`\varepsilon \sim \mathcal{N}(0, W^2)`.

        Called by the runner before each rollout and optionally every ``sde_sample_freq`` env
        steps. ``batch_size`` should be the number of parallel environments so each env gets
        its own per-env :math:`\varepsilon` slice in :attr:`exploration_matrices`.

        Args:
            batch_size: Number of per-env exploration matrices to draw.
        """
        std = self._get_std()
        # Use the reparametrization trick so this method participates in a backward pass if
        # the caller wants — but in practice callers wrap this with no_grad before rollouts.
        w_dist = Normal(torch.zeros_like(std), std)
        self.exploration_matrix = w_dist.rsample()
        self.exploration_matrices = w_dist.rsample((batch_size,))

    def update(self, mlp_output: torch.Tensor, latent_sde: torch.Tensor | None = None) -> None:
        r"""Cache the mean and ``latent_sde`` and build the marginal Gaussian distribution.

        Args:
            mlp_output: The MLP output, interpreted as the mean :math:`\mu(s)`. Shape
                ``[batch, output_dim]``.
            latent_sde: The penultimate MLP activation :math:`\phi(s)`. Shape ``[batch,
                latent_sde_dim]``. When :attr:`learn_features` is ``False`` (default), it is
                detached before use so gSDE gradients do not flow into the policy backbone.
        """
        if latent_sde is None:
            raise ValueError("GSDEGaussianDistribution.update requires latent_sde.")

        latent_sde = latent_sde if self.learn_features else latent_sde.detach()
        std = self._get_std()
        variance = torch.mm(latent_sde**2, std**2)
        marginal_std = torch.sqrt(variance + self.epsilon)

        self._mean = mlp_output
        self._latent_sde = latent_sde
        self._distribution = Normal(mlp_output, marginal_std)

    def _get_noise(self, latent_sde: torch.Tensor) -> torch.Tensor:
        r"""Return :math:`\phi(s) \cdot \varepsilon`, using per-env :math:`\varepsilon` when shapes match."""
        if self.exploration_matrix is None:
            raise RuntimeError(
                "GSDEGaussianDistribution.sample_weights must be called before sampling. "
                "The runner should invoke this at the start of every rollout."
            )
        if (
            self.exploration_matrices is None
            or latent_sde.shape[0] == 1
            or latent_sde.shape[0] != self.exploration_matrices.shape[0]
        ):
            # Fallback to the shared single-matrix path (used when the batch size doesn't
            # match the per-env stack, e.g. during PPO update over shuffled minibatches —
            # the sample() return value is discarded there, only the marginal log_prob is used).
            return torch.mm(latent_sde, self.exploration_matrix)
        return torch.bmm(latent_sde.unsqueeze(1), self.exploration_matrices).squeeze(1)

    def sample(self) -> torch.Tensor:
        r"""Sample :math:`a = \mu(s) + \phi(s) \cdot \varepsilon` with the current fixed :math:`\varepsilon`."""
        assert self._mean is not None and self._latent_sde is not None
        return self._mean + self._get_noise(self._latent_sde)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Return the MLP output unchanged; the mean is the deterministic output."""
        return mlp_output

    def as_deterministic_output_module(self) -> nn.Module:
        """Return an export-friendly module that returns the MLP output (the mean) unchanged."""
        return _IdentityDeterministicOutput()

    @property
    def mean(self) -> torch.Tensor:
        r"""Return the mean :math:`\mu(s)` of the current marginal Gaussian."""
        assert self._distribution is not None
        return self._distribution.mean

    @property
    def std(self) -> torch.Tensor:
        r"""Return the marginal stddev :math:`\sqrt{\phi(s)^2 \cdot \sigma^2}` of the current Gaussian."""
        assert self._distribution is not None
        return self._distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        """Return the entropy of the current marginal Gaussian, summed over the action dim."""
        assert self._distribution is not None
        return self._distribution.entropy().sum(dim=-1)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        """Return ``(mean, marginal_std)`` of the current Gaussian for rollout storage and KL."""
        return (self.mean, self.std)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute the log probability of ``outputs`` under the marginal Gaussian, summed over the action dim."""
        assert self._distribution is not None
        return self._distribution.log_prob(outputs).sum(dim=-1)

    def kl_divergence(self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]) -> torch.Tensor:
        r"""Compute :math:`\mathrm{KL}(\text{old} \,\Vert\, \text{new})` between two marginal Gaussians."""
        old_mean, old_std = old_params
        new_mean, new_std = new_params
        old_dist = Normal(old_mean, old_std)
        new_dist = Normal(new_mean, new_std)
        return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)


class _IdentityDeterministicOutput(nn.Module):
    """Exportable module that returns the MLP output as is."""

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output


class _MeanSliceDeterministicOutput(nn.Module):
    """Exportable module that extracts the mean from the MLP output (first slice of the second-to-last dimension)."""

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output[..., 0, :]
