# Copyright (c) 2024-2026, The UW Lab Project Developers. (https://github.com/uw-lab/UWLab/blob/main/CONTRIBUTORS.md).
# All Rights Reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Vision student + JIT teacher for rsl_rl ``DistillationRunner`` DAgger.

Differences vs vanilla ``rsl_rl.modules.StudentTeacher``:

* Student accepts a dict of proprio + per-camera depth images; uses a shared CNN
  encoder across cameras (feature-concat) then an MLP action head.
* Teacher is a loaded ``torch.jit.load`` module whose forward returns the action
  mean directly (the exporter bakes in the obs normalizer). No state-dict
  renaming; no MLP teacher reconstruction; gSDE std is dropped (we only use
  teacher mean for MSE-on-mean DAgger).
* ``__init__`` skips parent's 2D-shape assertion (images are ``B×C×H×W``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.transforms.v2 as T
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.modules import StudentTeacher
from rsl_rl.modules import MLP, EmpiricalNormalization


def _conv_out(hw: tuple[int, int], k: int, s: int) -> tuple[int, int]:
    return ((hw[0] - (k - 1) - 1) // s + 1, (hw[1] - (k - 1) - 1) // s + 1)


def _replace_bn_with_gn(module: nn.Module, group_size: int = 16) -> int:
    """In-place replace every ``BatchNorm2d`` under ``module`` with ``GroupNorm``.

    Groups = max(num_channels // ``group_size``, 1). Drops BN running stats
    (pretrained ImageNet BN stats are lost); GN affine init'd to (γ=1, β=0).
    Returns count of replacements. Standard recipe for IL/RL — eliminates BN
    train/eval mismatch and the small-effective-batch noise problem.
    """
    n = 0
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            num_channels = child.num_features
            num_groups = max(num_channels // group_size, 1)
            setattr(module, name, nn.GroupNorm(num_groups=num_groups, num_channels=num_channels, affine=True))
            n += 1
        else:
            n += _replace_bn_with_gn(child, group_size=group_size)
    return n


class DepthCNN(nn.Module):
    """4-layer conv encoder. Slim variant of DEXTRAH's ``CustomCNN``.

    Channels halved (8→16→32→64 vs DEXTRAH's 16→32→64→128) and LayerNorm
    removed. ``flatten → Linear`` head preserves per-pixel spatial info —
    needed for 6D pose regression where localization matters.

    Input:  ``(B, in_channels, H, W)`` depth image already normalized to [0, 1].
    Output: ``(B, embed_dim)`` feature vector.
    """

    def __init__(self, in_channels: int, height: int, width: int, embed_dim: int = 128):
        super().__init__()
        h, w = height, width

        h1, w1 = _conv_out((h, w), 6, 2)
        h2, w2 = _conv_out((h1, w1), 4, 2)
        h3, w3 = _conv_out((h2, w2), 4, 2)
        h4, w4 = _conv_out((h3, w3), 4, 2)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 8, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
        )
        self.flat_dim = 64 * h4 * w4
        self.head = nn.Sequential(
            nn.Linear(self.flat_dim, embed_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x).flatten(1))


class ResNet18Encoder(nn.Module):
    """Torchvision ResNet18 backbone + linear projection to ``embed_dim``.

    If ``pretrained_path`` is set, loads ImageNet weights from that file before
    replacing conv1 (so all other layers keep the pretrained values). Scratch
    init otherwise. All ``BatchNorm2d`` layers are swapped to ``GroupNorm`` to
    eliminate train/eval mismatch and batch-size sensitivity.

    For 3-channel (RGB) input:
      * ImageNet mean/std normalization is applied in forward (matches the
        pretrained-weights' expected input distribution).
      * Training-only photometric augmentations (ColorJitter, GaussianBlur,
        RandomGrayscale, GaussianNoise) are applied before normalization,
        gated on ``self.training``.

    For non-RGB (e.g. depth) input these RGB-specific transforms are skipped.
    """

    def __init__(self, in_channels: int, embed_dim: int = 128, pretrained_path: str = ""):
        super().__init__()
        import torchvision.models as tvm

        backbone = tvm.resnet18(weights=None)
        if pretrained_path:
            state_dict = torch.load(pretrained_path, map_location="cpu", weights_only=True)
            backbone.load_state_dict(state_dict, strict=True)
            print(f"[ResNet18Encoder] loaded ImageNet weights from {pretrained_path}")
        if in_channels != 3:
            # Replace first conv to accept non-RGB input. If pretrained, initialize
            # the new conv1 by averaging the original RGB weights across channels
            # (standard technique — preserves low-level filter structure for depth).
            new_conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            if pretrained_path:
                with torch.no_grad():
                    avg_w = backbone.conv1.weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
                    if in_channels == 1:
                        new_conv1.weight.copy_(avg_w)
                    else:
                        new_conv1.weight.copy_(avg_w.expand(-1, in_channels, -1, -1))
            backbone.conv1 = new_conv1

        n_replaced = _replace_bn_with_gn(backbone)
        print(f"[ResNet18Encoder] BatchNorm2d -> GroupNorm: {n_replaced} layers replaced")

        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.proj = nn.Linear(512, embed_dim)

        self.is_rgb = in_channels == 3
        if self.is_rgb:
            # ImageNet normalization buffers (broadcast over B, H, W via [1,3,1,1])
            self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
            # Photometric augmentations — applied only when ``self.training``.
            self.train_augs = T.Compose([
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
                T.RandomGrayscale(p=0.05),
                T.GaussianNoise(mean=0.0, sigma=0.01, clip=True),
            ])
        else:
            self.train_augs = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_rgb:
            if self.training and self.train_augs is not None:
                x = self.train_augs(x)
            x = (x - self.imagenet_mean) / self.imagenet_std
        f = self.backbone(x).flatten(1)
        return torch.relu(self.proj(f))


class StudentTeacherVision(StudentTeacher):
    """Vision-enabled StudentTeacher: depth CNN student + JIT teacher."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        teacher_jit_path: str,
        vision_groups: list[str],
        embed_dim: int = 128,
        student_hidden_dims: tuple[int, ...] | list[int] = (512, 256),
        activation: str = "elu",
        init_noise_std: float = 0.1,
        noise_std_type: str = "scalar",
        student_obs_normalization: bool = True,
        encoder_type: str = "depth_cnn",
        encoder_pretrained_path: str = "",
        encoder_freeze_iters: int = 0,
        aux_enabled: bool = False,
        aux_target_group: str = "aux_target",
        aux_hidden_dims: tuple[int, ...] | list[int] = (256, 128),
        predict_std: bool = False,
        teacher_returns_std: bool = False,
        aux_target_keys: list[str] | None = None,
        **kwargs,
    ) -> None:
        # Bypass StudentTeacher.__init__ (it asserts 1D obs and builds an MLP teacher).
        nn.Module.__init__(self)
        Normal.set_default_validate_args(False)

        self.obs_groups = obs_groups
        self.vision_groups = list(vision_groups)
        self.num_actions = num_actions

        # Proprio dim: concatenation of all groups listed under obs_groups["policy"].
        proprio = torch.cat([obs[g] for g in obs_groups["policy"]], dim=-1)
        assert proprio.ndim == 2, f"policy groups must be 1D; got shape {proprio.shape}"
        num_proprio = proprio.shape[-1]

        # Image shapes: all cameras must share the same (C, H, W). Accept both
        # 4D (B, C, H, W) — single-frame — and 5D (B, T, C, H, W) — frame
        # history. With ``flatten_history_dim=False`` on RGB obs terms, the
        # group shape is (B, T, C, H, W) and the encoder processes each frame
        # independently (DP-style).
        first_img = obs[self.vision_groups[0]]
        if first_img.ndim == 4:
            self.n_obs_steps = 1
            _, in_channels, H, W = first_img.shape
        elif first_img.ndim == 5:
            _, self.n_obs_steps, in_channels, H, W = first_img.shape
        else:
            raise AssertionError(
                f"vision group {self.vision_groups[0]} must be 4D (B,C,H,W) "
                f"or 5D (B,T,C,H,W); got shape {first_img.shape}"
            )
        for g in self.vision_groups[1:]:
            assert obs[g].shape == first_img.shape, (
                f"all vision groups must share shape; {g} has {obs[g].shape} vs {first_img.shape}"
            )

        # Per-view image encoders (DP/DEXTRAH style — no shared weights across
        # cameras). Each camera gets its own ResNet18+GroupNorm; pretrained
        # weights are loaded into each independently. ``depth_cnn`` keeps
        # the older shared encoder for non-RGB legacy paths.
        if encoder_type == "depth_cnn":
            shared_encoder = DepthCNN(in_channels, H, W, embed_dim=embed_dim)
            self.encoders = nn.ModuleDict({g: shared_encoder for g in self.vision_groups})
            self._per_view_encoders = False
        elif encoder_type == "resnet18":
            self.encoders = nn.ModuleDict({
                g: ResNet18Encoder(in_channels, embed_dim=embed_dim, pretrained_path=encoder_pretrained_path)
                for g in self.vision_groups
            })
            self._per_view_encoders = True
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")
        # Keep legacy attribute name for backward compat (recurrent variant).
        self.depth_encoder = next(iter(self.encoders.values()))
        # Vision feature is per-view + per-frame concatenated.
        vision_feat_dim = len(self.vision_groups) * self.n_obs_steps * embed_dim
        print(
            f"Encoders: {len(self.encoders)} views x {self.n_obs_steps} frames "
            f"x {embed_dim} embed = {vision_feat_dim} vision_feat_dim"
        )

        # DEXTRAH-style frozen-backbone warmup. If > 0, backbone starts with
        # ``requires_grad=False`` and is unfrozen once the algorithm hits
        # ``encoder_freeze_iters`` updates (see ``maybe_unfreeze_backbone``).
        # proj/std_head/aux_head/student MLP always stay trainable.
        self.encoder_freeze_iters = int(encoder_freeze_iters)
        self._backbone_frozen = self.encoder_freeze_iters > 0 and encoder_type == "resnet18"
        if self._backbone_frozen:
            self._set_backbone_requires_grad(False)
            print(f"[StudentTeacherVision] froze ResNet18 backbone for first {self.encoder_freeze_iters} iters")

        # Action head: [proprio ; per-cam image features] → num_actions.
        self.student = MLP(num_proprio + vision_feat_dim, num_actions, list(student_hidden_dims), activation)
        print(f"Student (proprio={num_proprio}, vision_feat={vision_feat_dim}): {self.student}")
        print(f"Encoder[{encoder_type}] (C={in_channels}, H={H}, W={W}, embed={embed_dim})")

        # Optional aux head: regresses object-pose targets from vision features only.
        # Forces the CNN to learn pose-aware features (per DEXTRAH).
        self.aux_enabled = bool(aux_enabled)
        self.aux_target_group = aux_target_group
        if self.aux_enabled:
            if aux_target_group not in obs:
                raise ValueError(
                    f"aux_enabled=True but obs has no '{aux_target_group}' group; "
                    f"got keys: {list(obs.keys())}"
                )
            # AuxTargetCfg is concatenate_terms=True (a single flat tensor) so it
            # survives rsl_rl's storage (which doesn't recurse into nested
            # TensorDicts). The per-target structure lives only on the policy /
            # algo side: keys come from ``aux_target_keys``, equal dims inferred.
            aux_obs = obs[aux_target_group]
            if hasattr(aux_obs, "items"):
                raise ValueError(
                    f"aux_target group '{aux_target_group}' must be a flat tensor "
                    f"(set concatenate_terms=True on the ObsGroup); "
                    f"got dict with keys {list(aux_obs.keys())}"
                )
            if not aux_target_keys:
                raise ValueError(
                    "aux_enabled=True but ``aux_target_keys`` is empty; "
                    "set it on StudentTeacherVisionPolicyCfg to match the "
                    "AuxTargetCfg ObsGroup field order (e.g. "
                    "['insertive_in_wrist', 'receptive_in_wrist', 'insertive_in_receptive'])."
                )
            total_aux_dim = int(aux_obs.shape[-1])
            n_keys = len(aux_target_keys)
            if total_aux_dim % n_keys != 0:
                raise ValueError(
                    f"aux_target total dim {total_aux_dim} is not divisible by "
                    f"len(aux_target_keys)={n_keys}; equal per-key dim required."
                )
            self.aux_keys: list[str] = list(aux_target_keys)
            self.aux_dim_per_key: int = total_aux_dim // n_keys
            self.aux_heads = nn.ModuleDict(
                {
                    k: MLP(vision_feat_dim, self.aux_dim_per_key, list(aux_hidden_dims), activation)
                    for k in self.aux_keys
                }
            )
            print(
                f"Aux heads (vision_feat={vision_feat_dim} -> {self.aux_dim_per_key} per key): "
                f"{self.aux_keys}"
            )
        else:
            self.aux_heads = None
            self.aux_keys = []
            self.aux_dim_per_key = 0

        # Normalize proprio (images already in [0,1] from process_image).
        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_proprio)
        else:
            self.student_obs_normalizer = nn.Identity()

        # JIT teacher — normalizer baked in, forward returns action mean.
        self.teacher = torch.jit.load(teacher_jit_path)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher_obs_normalization = False
        self.teacher_obs_normalizer = nn.Identity()
        self.loaded_teacher = True

        # Student exploration noise.
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"noise_std_type must be 'scalar' or 'log'; got {self.noise_std_type}")

        # Optional state-dependent std head. When enabled, overrides the scalar/log
        # noise above for DAgger-time std supervision (DEXTRAH-style weighted loss).
        # Outputs log-std, exp'd on read.
        self.predict_std = bool(predict_std)
        std_head_in_dim = num_proprio + vision_feat_dim
        if self.predict_std:
            self.std_head = nn.Linear(std_head_in_dim, num_actions)
            # Initialize so exp(log_std) ≈ init_noise_std at init (stable start).
            nn.init.zeros_(self.std_head.weight)
            nn.init.constant_(self.std_head.bias, float(torch.log(torch.tensor(init_noise_std)).item()))
            print(f"Std head ({std_head_in_dim} -> {num_actions}) enabled; log-std output")
        else:
            self.std_head = None

        # Teacher contract: if JIT returns (mean, std) tuple, callers can fetch both
        # via ``evaluate_with_std``; ``evaluate`` still returns mean-only for
        # legacy callers (e.g. ``DistillationDAgger.act`` for action routing).
        self.teacher_returns_std = bool(teacher_returns_std)

        self.distribution = None

    def _set_backbone_requires_grad(self, flag: bool) -> None:
        """Toggle ``requires_grad`` on the pretrained conv trunk only.

        ``ResNet18Encoder.backbone`` holds conv1..conv5_x (the ImageNet stack);
        ``proj`` (512→embed_dim linear) stays trainable so vision features can
        adapt to the action space even during the freeze window. No-op for
        ``DepthCNN`` (nothing pretrained to preserve).
        """
        for enc in self.encoders.values():
            if isinstance(enc, ResNet18Encoder):
                for p in enc.backbone.parameters():
                    p.requires_grad = flag

    def maybe_unfreeze_backbone(self, num_updates: int) -> None:
        """Unfreeze the ResNet18 backbone once ``num_updates >= encoder_freeze_iters``.

        Called by the DAgger algorithm at the top of each ``update()`` step.
        No-op if already unfrozen or if freeze was never requested.
        """
        if not self._backbone_frozen:
            return
        if num_updates >= self.encoder_freeze_iters:
            self._set_backbone_requires_grad(True)
            self._backbone_frozen = False
            print(f"[StudentTeacherVision] unfroze ResNet18 backbone at iter {num_updates}")

    def _encode_vision(self, obs: TensorDict) -> torch.Tensor:
        """Encode all camera views, frame-stacked. Returns flat (B, vision_feat_dim).

        For each view g: obs[g] shape is (B, T, C, H, W) — reshape to
        (B*T, C, H, W), encode through the per-view encoder, reshape back to
        (B, T, embed_dim), flatten to (B, T*embed_dim). Concat across views.
        Single-frame (4D) inputs are auto-promoted to T=1.
        """
        feats = []
        for g in self.vision_groups:
            img = obs[g]
            if img.ndim == 4:
                # (B, C, H, W) — single frame
                feat = self.encoders[g](img)  # (B, embed_dim)
            elif img.ndim == 5:
                # (B, T, C, H, W) — per-frame encode
                B, T, C, H, W = img.shape
                feat = self.encoders[g](img.reshape(B * T, C, H, W))  # (B*T, embed_dim)
                feat = feat.reshape(B, T * feat.shape[-1])  # (B, T*embed_dim)
            else:
                raise AssertionError(f"vision group {g} bad shape: {img.shape}")
            feats.append(feat)
        return torch.cat(feats, dim=-1)

    def _encode_student(self, obs: TensorDict) -> torch.Tensor:
        proprio = torch.cat([obs[g] for g in self.obs_groups["policy"]], dim=-1)
        proprio = self.student_obs_normalizer(proprio)
        img_feats = self._encode_vision(obs)
        return torch.cat([proprio, img_feats], dim=-1)

    def forward_with_aux(self, obs: TensorDict) -> tuple[torch.Tensor, dict[str, torch.Tensor] | None]:
        """Single forward pass that shares CNN features between action head and aux heads.

        Calling ``act_inference`` + ``evaluate_aux`` separately runs the encoder
        twice, doubling peak VRAM. Use this when the aux loss is active.

        ``aux_pred`` is a dict of per-target tensors when ``aux_enabled``;
        ``None`` otherwise.
        """
        proprio = torch.cat([obs[g] for g in self.obs_groups["policy"]], dim=-1)
        proprio = self.student_obs_normalizer(proprio)
        img_feats = self._encode_vision(obs)
        action = self.student(torch.cat([proprio, img_feats], dim=-1))
        aux_pred: dict[str, torch.Tensor] | None = None
        if self.aux_enabled:
            aux_pred = {k: head(img_feats) for k, head in self.aux_heads.items()}
        return action, aux_pred

    def get_aux_target(self, obs: TensorDict) -> torch.Tensor:
        if not self.aux_enabled:
            raise RuntimeError("get_aux_target called but aux head is not enabled")
        return obs[self.aux_target_group]

    def _encode_teacher(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[g] for g in self.obs_groups["teacher"]], dim=-1)

    def _update_distribution(self, feat: torch.Tensor) -> None:
        mean = self.student(feat)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict) -> torch.Tensor:
        feat = self._encode_student(obs)
        self._update_distribution(feat)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        feat = self._encode_student(obs)
        return self.student(feat)

    def evaluate(self, obs: TensorDict) -> torch.Tensor:
        teacher_obs = self._encode_teacher(obs)
        with torch.no_grad():
            out = self.teacher(teacher_obs)
        if isinstance(out, tuple):  # teacher_returns_std=True: (mean, std)
            return out[0]
        return out

    def evaluate_with_std(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, std) from the teacher JIT. Requires ``teacher_returns_std=True``."""
        if not self.teacher_returns_std:
            raise RuntimeError(
                "evaluate_with_std called but teacher_returns_std=False; "
                "re-export the teacher JIT with --std and set teacher_returns_std=True."
            )
        teacher_obs = self._encode_teacher(obs)
        with torch.no_grad():
            out = self.teacher(teacher_obs)
        assert isinstance(out, tuple) and len(out) == 2, (
            f"teacher_returns_std=True but JIT returned {type(out)}"
        )
        return out

    def forward_all_heads(
        self, obs: TensorDict
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None]:
        """Single encoder pass → (μ, σ, aux_pred). Unifies what was previously
        two separate methods (``act_inference_with_std`` + ``forward_with_aux``)
        that each ran the encoder. Use this in the weighted-loss update when
        ``aux_enabled`` and ``predict_std`` are both True.

        ``aux_pred`` is a per-target dict when ``aux_enabled``; None otherwise.
        It reads the vision portion of ``feat`` only — proprio is excluded so
        the encoder bears the full pressure of pose-reconstruction.
        """
        if not self.predict_std:
            raise RuntimeError("forward_all_heads called but predict_std=False")
        # Inline the encoder pass so we don't run proprio twice.
        proprio = torch.cat([obs[g] for g in self.obs_groups["policy"]], dim=-1)
        proprio = self.student_obs_normalizer(proprio)
        img_feats = self._encode_vision(obs)
        feat = torch.cat([proprio, img_feats], dim=-1)
        mean = self.student(feat)
        log_std = self.std_head(feat).clamp(-5.0, 2.0)
        std = torch.exp(log_std)
        aux_pred: dict[str, torch.Tensor] | None = None
        if self.aux_enabled:
            aux_pred = {k: head(img_feats) for k, head in self.aux_heads.items()}
        return mean, std, aux_pred

    def act_inference_with_std(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (mean, std) from the student. Requires ``predict_std=True``."""
        if not self.predict_std:
            raise RuntimeError("act_inference_with_std called but predict_std=False")
        feat = self._encode_student(obs)
        mean = self.student(feat)
        # Clamp to prevent pathological drift during long DAgger training.
        # Init bias = log(init_noise_std) ≈ -2.3 sits well inside this range.
        log_std = self.std_head(feat).clamp(-5.0, 2.0)
        std = torch.exp(log_std)
        return mean, std

    def get_student_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._encode_student(obs)

    def get_teacher_obs(self, obs: TensorDict) -> torch.Tensor:
        return self._encode_teacher(obs)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.student_obs_normalization:
            proprio = torch.cat([obs[g] for g in self.obs_groups["policy"]], dim=-1)
            self.student_obs_normalizer.update(proprio)

    def train(self, mode: bool = True):
        nn.Module.train(self, mode)
        self.teacher.eval()
        return self

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        """Resume from a distillation checkpoint. Skips teacher keys (JIT already loaded)."""
        filtered = {k: v for k, v in state_dict.items() if not k.startswith("teacher.")}
        nn.Module.load_state_dict(self, filtered, strict=False)
        self.loaded_teacher = True
        self.teacher.eval()
        return True
