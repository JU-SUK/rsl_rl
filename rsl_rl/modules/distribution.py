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


class TanhGaussianDistribution(HeteroscedasticGaussianDistribution):
    """Tanh-squashed Gaussian (Brax ``NormalTanhDistribution`` style) for bounded actions in ``[-1, 1]``.

    Inherits the heteroscedastic (state-dependent std) base Gaussian over the pre-squash variable ``u``; the
    executed action is ``a = tanh(u)``. Compared to squashing in the environment, doing it in the policy is the
    design deployed for sim-to-real (Brax / MuJoCo Playground):

    * :meth:`log_prob` applies the change-of-variables correction ``- sum log(1 - a^2 + eps)``.
    * :attr:`entropy` is estimated from a reparameterized sample (the squashed entropy has no closed form). The
      sampled entropy is *bounded*, so the entropy bonus stops paying for larger std once actions saturate —
      the std self-regulates, instead of inflating to a bang-bang policy (the failure mode of env-side tanh).
    * :attr:`params` / :attr:`std` / :meth:`kl_divergence` use the raw base Gaussian, so the adaptive-LR KL is
      computed on the unsquashed distribution (correct and stable).

    Numerically guarded with a clamped ``atanh`` and a ``min_std`` floor (set via ``std_range[0]``; use ~1e-3).
    """

    _TANH_EPS = 1e-6

    def update(self, mlp_output: torch.Tensor) -> None:
        """Update the base Gaussian with non-finite network outputs sanitized first.

        ``torch.clamp`` passes NaN through and ``Normal`` raises on a NaN/inf std at sample
        time, which hard-crashes a multi-GPU run. A transient inf/NaN in the actor output
        (e.g. one bad minibatch elsewhere) is mapped to a safe value instead: NaN -> 0,
        +-inf -> +-10 (a pre-squash mean of +-10 already saturates tanh; a log-std of +-10
        is clamped by ``log_std_range``/``std_range`` in the base update).
        """
        super().update(torch.nan_to_num(mlp_output, nan=0.0, posinf=10.0, neginf=-10.0))

    def sample(self) -> torch.Tensor:
        """Sample ``u ~ N(mean, std)`` and return the squashed action ``tanh(u)``."""
        return torch.tanh(self._distribution.sample())  # type: ignore

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        """Squashed mean: ``tanh`` of the heteroscedastic mean slice."""
        return torch.tanh(super().deterministic_output(mlp_output))

    def as_deterministic_output_module(self) -> nn.Module:
        """Export-friendly module returning ``tanh(mean_slice)``."""
        return _TanhMeanSliceDeterministicOutput()

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Squashed log-prob: base Gaussian log-prob of ``atanh(a)`` minus the tanh Jacobian, summed."""
        a = outputs.clamp(-1.0 + self._TANH_EPS, 1.0 - self._TANH_EPS)
        u = torch.atanh(a)
        base = self._distribution.log_prob(u).sum(dim=-1)  # type: ignore
        correction = torch.log(1.0 - a.pow(2) + self._TANH_EPS).sum(dim=-1)
        return base - correction

    @property
    def entropy(self) -> torch.Tensor:
        """Sampled entropy of the squashed distribution, ``-E_u[log p(tanh(u))]`` (bounded, differentiable)."""
        u = self._distribution.rsample()  # type: ignore  # reparameterized so the entropy bonus has a gradient
        base = self._distribution.log_prob(u).sum(dim=-1)  # type: ignore
        correction = torch.log(1.0 - torch.tanh(u).pow(2) + self._TANH_EPS).sum(dim=-1)
        return -(base - correction)


class GsdeDistribution(GaussianDistribution):
    """Generalized State-Dependent Exploration (gSDE) Gaussian distribution.

    The exploration noise is correlated across time within a rollout because it
    is a deterministic linear function of the policy's penultimate features and
    a sampled weight matrix that is held fixed for many consecutive steps.

    Per Raffin et al., "Smooth Exploration for Robotic Reinforcement Learning"
    (https://arxiv.org/abs/2005.05719):

    * ``log_std`` is a 2-D learnable parameter of shape
      ``(latent_dim, output_dim)`` (vs. 1-D for the homoscedastic Gaussian).
    * Per-action stddev is ``sqrt(phi(s)**2 @ exp(log_std)**2 + eps)`` where
      ``phi(s)`` are the features from the last hidden layer of the policy MLP.
    * Sampled action is ``mean + phi(s) @ W``, with the weight matrix
      ``W ~ N(0, exp(log_std)**2)`` resampled periodically (typically once per
      rollout) via :meth:`sample_weights`. Between resamples the noise is a
      smooth state-dependent function — exactly the gSDE property.

    Use with :class:`rsl_rl.models.MLPModel` (or any subclass) which now
    automatically calls :meth:`set_features` with the penultimate-layer
    activations before :meth:`update`. The model also invokes
    :meth:`init_mlp_weights` to lazily allocate ``log_std`` once the MLP's
    latent dim is known.
    """

    def __init__(
        self,
        output_dim: int,
        init_std: float = 1.0,
        std_range: tuple[float, float] = (1e-6, 1e6),
        std_type: str = "log",
        epsilon: float = 1e-6,
        learn_std: bool = True,
    ) -> None:
        """Initialize the gSDE distribution module.

        Args:
            output_dim: Action / output dimension.
            init_std: Initial scalar standard deviation; broadcasted to all entries of ``log_std``.
            std_range: ``(min, max)`` clamp range applied to the exp-of-log std.
            std_type: Accepted for compatibility with :class:`GaussianDistribution` 's cfg
                serialization; gSDE always parameterizes the std in log-space internally,
                so this argument is ignored apart from a value check.
            epsilon: Numerical stabilization added inside the sqrt for variance.
            learn_std: Whether ``log_std`` is learnable. ``False`` fixes it to ``log(init_std)``.
        """
        if std_type not in ("log", "scalar"):
            raise ValueError(f"GsdeDistribution: unknown std_type={std_type!r}; expected 'log' or 'scalar'.")
        # Sidestep GaussianDistribution.__init__ — it allocates a 1-D
        # log_std_param we don't use. Replicate the bits we need.
        Distribution.__init__(self, output_dim)
        self.std_type = "gsde"
        self.epsilon = epsilon
        self._init_std = float(init_std)
        self._learn_std = bool(learn_std)
        self.std_range = list(std_range)
        self.std_range[0] = max(self.std_range[0], 1e-6)
        self.log_std_range = [float(np.log(self.std_range[0])), float(np.log(self.std_range[1]))]

        # Lazily allocated: needs latent_dim from MLPModel.init_mlp_weights().
        self.log_std_param: nn.Parameter | None = None
        # Buffer storing the resampled exploration weight matrix; shape
        # ``(latent_dim, output_dim)``. Lazily allocated alongside log_std.
        self.register_buffer("_exploration_matrix", torch.zeros(0))
        # Features cache: set by MLPModel.forward() before each update().
        self._cached_features: torch.Tensor | None = None
        # Distribution populated by update(); used for log_prob / kl_divergence.
        self._distribution: Normal | None = None

        Normal.set_default_validate_args(False)

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        """Allocate the 2-D ``log_std`` parameter once the MLP is built.

        Infers ``latent_dim`` from the penultimate ``nn.Linear`` layer's output
        dimension (the layer feeding the final output linear of the MLP).
        """
        if self.log_std_param is not None:
            return
        latent_dim = None
        # Iterate from the end: first Linear is the output head; second Linear
        # we hit is the last hidden layer whose ``out_features`` is the
        # latent_dim that gets passed as features to this distribution.
        seen_output = False
        for module in reversed(list(mlp.modules())):
            if isinstance(module, nn.Linear):
                if not seen_output:
                    seen_output = True
                    continue
                latent_dim = module.out_features
                break
        if latent_dim is None:
            raise RuntimeError(
                "gSDE: could not infer latent_dim from the MLP. The MLP must contain at"
                " least two ``nn.Linear`` layers (a hidden layer + an output layer)."
            )
        device = next(mlp.parameters()).device if any(True for _ in mlp.parameters()) else torch.device("cpu")
        log_init = float(np.log(self._init_std + 1e-7))
        self.log_std_param = nn.Parameter(
            log_init * torch.ones(latent_dim, self.output_dim, device=device),
            requires_grad=self._learn_std,
        )
        self._exploration_matrix = torch.zeros(latent_dim, self.output_dim, device=device)
        self.sample_weights()

    @property
    def latent_dim(self) -> int:
        if self.log_std_param is None:
            raise RuntimeError("gSDE: latent_dim is undefined until init_mlp_weights() runs.")
        return self.log_std_param.shape[0]

    def sample_weights(self) -> None:
        """Resample the exploration weight matrix from ``N(0, std**2)``.

        Call once per rollout (or every K steps) to refresh the correlated
        noise direction. Between resamples, the noise applied to every
        action is a smooth function of the latent features.
        """
        if self.log_std_param is None:
            return
        log_std = self.log_std_param.detach().clamp(self.log_std_range[0], self.log_std_range[1])
        std = torch.exp(log_std)
        self._exploration_matrix = std * torch.randn_like(std)

    def set_features(self, features: torch.Tensor) -> None:
        """Cache the penultimate-layer features for the next :meth:`update` and :meth:`sample`."""
        self._cached_features = features

    def update(self, mlp_output: torch.Tensor) -> None:
        """Build a per-batch Gaussian with state-dependent stddev.

        Stddev for action ``i`` in batch entry ``b`` is
        ``sqrt(sum_j features[b, j]^2 * exp(log_std[j, i])^2 + epsilon)``.
        """
        if self.log_std_param is None:
            raise RuntimeError(
                "GsdeDistribution.update called before init_mlp_weights(). MLPModel should"
                " trigger this automatically once the actor is constructed."
            )
        if self._cached_features is None:
            raise RuntimeError(
                "GsdeDistribution.update called before set_features(). The wrapping model"
                " must call set_features() with the penultimate-layer activations."
            )
        log_std = self.log_std_param.clamp(self.log_std_range[0], self.log_std_range[1])
        std_2d = torch.exp(log_std)
        variance = self._cached_features.pow(2) @ std_2d.pow(2)
        stddev = torch.sqrt(variance + self.epsilon)
        self._distribution = Normal(mlp_output, stddev)

    def sample(self) -> torch.Tensor:
        """Sample an action by adding state-dependent correlated noise to the mean.

        Noise is ``features @ exploration_matrix``; deterministic given the
        current features and the most recently sampled weight matrix.
        """
        if self._cached_features is None or self._exploration_matrix.numel() == 0:
            raise RuntimeError("GsdeDistribution.sample called before update().")
        noise = self._cached_features @ self._exploration_matrix.to(self._cached_features.device)
        assert self._distribution is not None
        return self._distribution.mean + noise

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        return (self.mean, self.std)


class _IdentityDeterministicOutput(nn.Module):
    """Exportable module that returns the MLP output as is."""

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output


class _MeanSliceDeterministicOutput(nn.Module):
    """Exportable module that extracts the mean from the MLP output (first slice of the second-to-last dimension)."""

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return mlp_output[..., 0, :]


class _TanhMeanSliceDeterministicOutput(nn.Module):
    """Exportable module returning ``tanh`` of the mean slice (for :class:`TanhGaussianDistribution`)."""

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        return torch.tanh(mlp_output[..., 0, :])
