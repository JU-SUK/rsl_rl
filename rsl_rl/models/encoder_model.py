# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict
from typing import Any

from rsl_rl.models.mlp_model import MLPModel
from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState


class MLPEncoderModel(MLPModel):
    """MLP-based model with dedicated MLP encoders for selected 1D observation groups.

    Analogous to :class:`rsl_rl.models.CNNModel`, but operates on 1D observations only. The
    active observation set is partitioned into:

    * *encoded* groups - routed through per-group :class:`rsl_rl.modules.MLP` encoders that
      produce a compact feature vector;
    * *passthrough* groups - concatenated directly with the encoder outputs and fed into the
      main MLP head.

    This is useful when a subset of the observation is high-dimensional and benefits from
    its own sub-network (e.g., a flat height-scan, lidar scan, or point-cloud). Each
    instance builds its own encoders; actor and critic do not share encoder parameters.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        encoder_normalization: bool = False,
        head_layer_norm: bool = True,
        distribution_cfg: dict | None = None,
        encoder_cfg: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        """Initialize the MLP-encoder model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., ``"actor"`` or ``"critic"``).
            output_dim: Dimension of the output.
            hidden_dims: Hidden dimensions of the main MLP head.
            activation: Activation function of the main MLP head.
            obs_normalization: Whether to apply running-statistic :class:`EmpiricalNormalization`
                to the *passthrough* observation groups before they are concatenated with encoder
                features and fed to the main MLP head. Controls only the passthrough path; the
                encoder input path is controlled by :paramref:`encoder_normalization`.
            encoder_normalization: Whether to apply running-statistic :class:`EmpiricalNormalization`
                to each *encoded* observation group before it enters its per-group encoder MLP. Each
                encoded group gets its own normalizer, updated in lockstep from :meth:`update_normalization`.
            head_layer_norm: Whether to apply :class:`torch.nn.LayerNorm` on the concatenated latent
                ``[passthrough || encoder_features]`` right before the main MLP head. Per-sample
                normalization with no running statistics; safe across train/eval modes and distributed
                ranks. Recommended default in line with modern ML practice (transformers, multi-modal
                fusion, MLP-Mixer, recent stable-RL work such as SimBa, 2024). Defaults to True.
            distribution_cfg: Configuration dictionary for the output distribution. If provided, the model outputs
                stochastic values sampled from the distribution.
            encoder_cfg: Mapping from observation group name to a per-group encoder MLP configuration. Each value is
                a dictionary forwarded to :class:`rsl_rl.modules.MLP` (at minimum ``output_dim`` and ``hidden_dims``).
                The keys of this mapping define which observation groups are routed through encoders; all other
                groups in ``obs_groups[obs_set]`` are fed directly to the main MLP head.
        """
        if encoder_cfg is None:
            raise ValueError("MLPEncoderModel requires 'encoder_cfg' to be provided.")

        # Determine which observation groups are routed through encoders. This must happen before calling
        # ``_get_obs_dim`` (which relies on ``self._encoded_obs_group_keys``) and before ``super().__init__``
        # (which in turn triggers ``_get_obs_dim`` via the parent's constructor).
        self._encoded_obs_group_keys: list[str] = list(encoder_cfg.keys())

        # Populate ``self.obs_groups_encoded`` and ``self.obs_dims_encoded`` for the encoded subset.
        self._get_obs_dim(obs, obs_groups, obs_set)

        if set(encoder_cfg.keys()) != set(self.obs_groups_encoded):
            raise ValueError(
                "The number and names of encoder configurations must match the encoded observation groups."
            )

        # Build the per-group encoders.
        encoders: dict[str, nn.Module] = {}
        for idx, obs_group in enumerate(self.obs_groups_encoded):
            encoders[obs_group] = MLP(input_dim=self.obs_dims_encoded[idx], **encoder_cfg[obs_group])

        # Compute total encoded latent dim by inspecting each encoder's last Linear layer.
        self.encoder_latent_dim = sum(_mlp_output_dim(enc) for enc in encoders.values())

        # Initialize the parent MLP model. The parent's ``__init__`` re-invokes ``self._get_obs_dim``, which now
        # returns only the passthrough subset (``(passthrough_groups, passthrough_dim)``), and builds the
        # passthrough :class:`EmpiricalNormalization` when ``obs_normalization=True``.
        super().__init__(
            obs, obs_groups, obs_set, output_dim, hidden_dims, activation, obs_normalization, distribution_cfg
        )

        # Register encoders after super().__init__ so the parent MLP head is built first.
        self.encoders = nn.ModuleDict(encoders)

        # Per-encoded-group input normalizers. Controlled by ``encoder_normalization``, independent of the
        # passthrough normalization flag. When disabled, use Identity so ``get_latent`` need not branch.
        self.encoder_normalization = encoder_normalization
        if encoder_normalization:
            self.encoder_normalizers = nn.ModuleDict(
                {g: EmpiricalNormalization(self.obs_dims_encoded[i]) for i, g in enumerate(self.obs_groups_encoded)}
            )
        else:
            self.encoder_normalizers = nn.ModuleDict({g: nn.Identity() for g in self.obs_groups_encoded})

        # Post-concat LayerNorm on the main-MLP-head input. Per-sample, no running stats, no train/eval
        # mismatch. Stabilizes the head against scale drift in encoder outputs during training.
        latent_dim = self._get_latent_dim()
        self.head_norm: nn.Module = nn.LayerNorm(latent_dim) if head_layer_norm else nn.Identity()

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent by concatenating normalized passthrough obs with encoder outputs.

        Encoder inputs are normalized by their per-group :class:`EmpiricalNormalization` before the
        encoder MLP runs, mirroring the passthrough normalization handled by the parent class. The
        concatenated latent then goes through :attr:`head_norm` (either :class:`~torch.nn.LayerNorm`
        or :class:`~torch.nn.Identity`) to stabilize the main MLP head input.
        """
        # Passthrough obs -> concatenated + normalized by parent implementation
        latent_passthrough = super().get_latent(obs, masks, hidden_state)
        # Encoded obs -> flatten trailing dims (so 2D images / multi-channel maps work as encoder
        # inputs), then per-group normalize, then per-group MLP encoder, then concat.
        latent_encoded_list = [
            self.encoders[obs_group](self.encoder_normalizers[obs_group](obs[obs_group].flatten(start_dim=1)))
            for obs_group in self.obs_groups_encoded
        ]
        latent_encoded = torch.cat(latent_encoded_list, dim=-1)
        latent = torch.cat([latent_passthrough, latent_encoded], dim=-1)
        # Per-sample LayerNorm (no-op when ``head_layer_norm=False``) on the main MLP head input.
        return self.head_norm(latent)

    def get_encoder_latents(self, obs: TensorDict) -> dict[str, torch.Tensor]:
        """Return per-encoded-group encoder output (post per-group normalization, post encoder MLP).

        Useful for diagnostics and offline analysis of the compact representation actually consumed
        by the main MLP head. Does not apply :attr:`head_norm`, which acts on the concatenated
        ``[passthrough || encoder_features]`` latent.
        """
        return {
            obs_group: self.encoders[obs_group](
                self.encoder_normalizers[obs_group](obs[obs_group].flatten(start_dim=1))
            )
            for obs_group in self.obs_groups_encoded
        }

    def update_normalization(self, obs: TensorDict) -> None:
        """Update running statistics of both the passthrough and per-encoder-group normalizers."""
        # Update the passthrough normalizer (handled by parent, gated on ``obs_normalization``).
        super().update_normalization(obs)
        # Update the per-group encoder-input normalizers (gated on ``encoder_normalization``).
        # Flatten trailing dims to match the layout the normalizer was constructed for —
        # mirrors the flatten in :meth:`get_latent` that feeds the encoder MLP.
        if self.encoder_normalization:
            for obs_group in self.obs_groups_encoded:
                self.encoder_normalizers[obs_group].update(obs[obs_group].flatten(start_dim=1))  # type: ignore[union-attr]

    def as_jit(self) -> nn.Module:
        """Return a TorchScript-friendly copy of the model for JIT export."""
        return _TorchMLPEncoderModel(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """Return an ONNX-export wrapper around the model."""
        return _OnnxMLPEncoderModel(self, verbose)

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Partition active observation groups into encoded and passthrough subsets.

        Populates ``self.obs_groups_encoded`` and ``self.obs_dims_encoded`` for the encoded subset, and returns
        ``(passthrough_groups, passthrough_dim)`` for the parent :class:`MLPModel` constructor to consume.
        """
        active_obs_groups = obs_groups[obs_set]
        passthrough_groups: list[str] = []
        passthrough_dim = 0
        encoded_groups: list[str] = []
        encoded_dims: list[int] = []

        for obs_group in active_obs_groups:
            shape = obs[obs_group].shape
            if len(shape) < 2:
                raise ValueError(
                    f"Observation '{obs_group}' has shape {tuple(shape)}; expected at least"
                    " (batch, ...) with one feature dim."
                )
            # Flatten product of all non-batch dims so the encoder MLP can ingest 2D images,
            # multi-channel maps, etc. without a separate Flatten layer in the cfg. ``height_scan``
            # at (B, 1, 76, 126) becomes a 9576-dim feature vector for the encoder MLP.
            feature_dim = 1
            for d in shape[1:]:
                feature_dim *= int(d)
            if obs_group in self._encoded_obs_group_keys:
                encoded_groups.append(obs_group)
                encoded_dims.append(feature_dim)
            else:
                # Pass-through (raw concat) only makes sense for 1D obs. If a 2D+ obs is in
                # passthrough mode it'd silently be misinterpreted by ``torch.cat(..., dim=-1)``,
                # so reject loudly.
                if len(shape) != 2:
                    raise ValueError(
                        f"Observation '{obs_group}' has shape {tuple(shape)} (>1D) but is in"
                        " passthrough mode. Either route it through an encoder via ``encoder_cfg``"
                        " or reduce it to 1D before reaching the model."
                    )
                passthrough_groups.append(obs_group)
                passthrough_dim += int(shape[-1])

        missing = set(self._encoded_obs_group_keys) - set(encoded_groups)
        if missing:
            raise ValueError(
                f"Encoder configured for obs groups {sorted(missing)} but they are not present in the active"
                f" observation set {active_obs_groups}."
            )
        if not encoded_groups:
            raise ValueError(
                "No observation groups were routed through encoders. If no groups need encoding, use MLPModel"
                " directly instead of MLPEncoderModel."
            )

        # Store encoded-group metadata as attributes so ``__init__`` can build the encoders and
        # ``_get_latent_dim`` can include the encoder latent in the main MLP input dim.
        self.obs_groups_encoded = encoded_groups
        self.obs_dims_encoded = encoded_dims
        # Return passthrough subset for the parent MLP model.
        return passthrough_groups, passthrough_dim

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the main MLP head."""
        return self.obs_dim + self.encoder_latent_dim


def _mlp_output_dim(mlp: nn.Module) -> int:
    """Return the output dimension of a :class:`rsl_rl.modules.MLP` by scanning for the last Linear layer."""
    last_linear: nn.Linear | None = None
    for module in mlp.modules():
        if isinstance(module, nn.Linear):
            last_linear = module
    if last_linear is None:
        raise ValueError("Could not determine MLP output dimension: no torch.nn.Linear layer found.")
    return last_linear.out_features


class _NormalizedEncoder(nn.Module):
    """Per-group ``EmpiricalNormalization`` followed by an MLP encoder.

    Bundled as one module so the export wrappers iterate a single
    :class:`torch.nn.ModuleList` (TorchScript cannot index a ModuleList with a non-literal).
    """

    def __init__(self, normalizer: nn.Module, encoder: nn.Module) -> None:
        super().__init__()
        self.normalizer = normalizer
        self.encoder = encoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(self.normalizer(x))


def _build_encoder_blocks(model: MLPEncoderModel) -> nn.ModuleList:
    return nn.ModuleList(
        [
            _NormalizedEncoder(copy.deepcopy(model.encoder_normalizers[g]), copy.deepcopy(model.encoders[g]))
            for g in model.obs_groups_encoded
        ]
    )


class _TorchMLPEncoderModel(nn.Module):
    """Exportable MLP encoder model for JIT.

    The exported forward takes the passthrough observations as a single pre-concatenated
    ``obs_passthrough`` tensor (in :attr:`MLPEncoderModel.obs_groups` order) and the encoded
    observations as a list of per-group tensors (in :attr:`MLPEncoderModel.obs_groups_encoded`
    order, each pre-flattened along non-batch dims).
    """

    is_recurrent: bool = False

    def __init__(self, model: MLPEncoderModel) -> None:
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoder_blocks = _build_encoder_blocks(model)
        self.head_norm = copy.deepcopy(model.head_norm)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, obs_passthrough: torch.Tensor, obs_encoded: list[torch.Tensor]) -> torch.Tensor:
        """Run deterministic inference from separated passthrough and per-group encoded inputs."""
        latent_passthrough = self.obs_normalizer(obs_passthrough)
        latent_encoded_list: list[torch.Tensor] = []
        for i, block in enumerate(self.encoder_blocks):
            latent_encoded_list.append(block(obs_encoded[i]))
        latent_encoded = torch.cat(latent_encoded_list, dim=-1)
        latent = torch.cat([latent_passthrough, latent_encoded], dim=-1)
        latent = self.head_norm(latent)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for encoder exports)."""
        pass


class _OnnxMLPEncoderModel(nn.Module):
    """Exportable MLP encoder model for ONNX.

    Encoded inputs must be pre-flattened to ``(batch, flat_dim)`` along non-batch dims, since the
    encoder normalizers and MLPs are constructed with flat input shapes.
    """

    is_recurrent: bool = False

    def __init__(self, model: MLPEncoderModel, verbose: bool) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.encoder_blocks = _build_encoder_blocks(model)
        self.head_norm = copy.deepcopy(model.head_norm)
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

        self.obs_groups_encoded: list[str] = list(model.obs_groups_encoded)
        self.obs_dims_encoded: list[int] = list(model.obs_dims_encoded)
        self.obs_dim_passthrough: int = int(model.obs_dim)

    def forward(self, obs_passthrough: torch.Tensor, *obs_encoded: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        latent_passthrough = self.obs_normalizer(obs_passthrough)
        latent_encoded_list = []
        for i, block in enumerate(self.encoder_blocks):
            latent_encoded_list.append(block(obs_encoded[i]))
        latent_encoded = torch.cat(latent_encoded_list, dim=-1)
        latent = torch.cat([latent_passthrough, latent_encoded], dim=-1)
        latent = self.head_norm(latent)
        out = self.mlp(latent)
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor, ...]:
        """Return representative dummy inputs for ONNX tracing."""
        dummy_passthrough = torch.zeros(1, self.obs_dim_passthrough)
        dummy_encoded = tuple(torch.zeros(1, dim) for dim in self.obs_dims_encoded)
        return (dummy_passthrough, *dummy_encoded)

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names (``obs`` for passthrough, group name for each encoded input)."""
        return ["obs", *self.obs_groups_encoded]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
