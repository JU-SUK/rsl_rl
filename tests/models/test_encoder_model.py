# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the MLPEncoderModel."""

from __future__ import annotations

import io
import pytest
import torch
from tensordict import TensorDict

from rsl_rl.models import MLPEncoderModel
from rsl_rl.modules import EmpiricalNormalization

NUM_ENVS = 4
POLICY_DIM = 6
TASK_DIM = 3
HEIGHT_DIM = 32
NUM_ACTIONS = 4

OBS_GROUPS = {
    "actor": ["policy", "task", "height_scan"],
    "critic": ["policy", "task", "height_scan"],
}


def _make_obs() -> TensorDict:
    return TensorDict(
        {
            "policy": torch.randn(NUM_ENVS, POLICY_DIM),
            "task": torch.randn(NUM_ENVS, TASK_DIM),
            "height_scan": torch.randn(NUM_ENVS, HEIGHT_DIM),
        },
        batch_size=[NUM_ENVS],
    )


def _encoder_cfg(output_dim: int = 8) -> dict[str, dict]:
    return {"height_scan": {"output_dim": output_dim, "hidden_dims": [16], "activation": "elu"}}


def _make_model(**kwargs: object) -> tuple[MLPEncoderModel, TensorDict]:
    obs = _make_obs()
    defaults: dict[str, object] = {
        "hidden_dims": [16, 16],
        "activation": "elu",
        "encoder_cfg": _encoder_cfg(),
    }
    defaults.update(kwargs)
    model = MLPEncoderModel(obs, OBS_GROUPS, "actor", output_dim=NUM_ACTIONS, **defaults)
    return model, obs


def test_partition_into_passthrough_and_encoded():
    model, _ = _make_model()
    # Parent stores passthrough subset in obs_groups; encoded subset in obs_groups_encoded.
    assert model.obs_groups == ["policy", "task"]
    assert model.obs_groups_encoded == ["height_scan"]
    assert model.obs_dim == POLICY_DIM + TASK_DIM
    assert model.obs_dims_encoded == [HEIGHT_DIM]
    # latent = passthrough(9) + encoder_out(8) = 17
    assert model._get_latent_dim() == POLICY_DIM + TASK_DIM + 8
    # main MLP first linear consumes the full latent
    first_linear = next(layer for layer in model.mlp if isinstance(layer, torch.nn.Linear))
    assert first_linear.in_features == model._get_latent_dim()


def test_forward_shape():
    model, obs = _make_model()
    out = model(obs)
    assert out.shape == (NUM_ENVS, NUM_ACTIONS)


def test_get_latent_uses_encoder():
    """Mutating the height_scan column must propagate into the latent (encoder is wired in)."""
    model, obs = _make_model()
    model.eval()
    base = model.get_latent(obs).clone()
    obs2 = obs.clone()
    obs2["height_scan"] = obs["height_scan"] + 100.0
    perturbed = model.get_latent(obs2)
    assert not torch.allclose(base, perturbed)


def test_passthrough_normalization_only_covers_passthrough_dim():
    model, _ = _make_model(obs_normalization=True)
    assert isinstance(model.obs_normalizer, EmpiricalNormalization)
    assert tuple(model.obs_normalizer._mean.shape) == (1, POLICY_DIM + TASK_DIM)
    # encoder_normalizers default to Identity
    assert isinstance(model.encoder_normalizers["height_scan"], torch.nn.Identity)


def test_encoder_normalization_creates_per_group_normalizer():
    model, _ = _make_model(encoder_normalization=True)
    norm = model.encoder_normalizers["height_scan"]
    assert isinstance(norm, EmpiricalNormalization)
    assert tuple(norm._mean.shape) == (1, HEIGHT_DIM)


def test_update_normalization_advances_passthrough_and_encoder_counts():
    model, obs = _make_model(obs_normalization=True, encoder_normalization=True)
    assert int(model.obs_normalizer.count) == 0
    assert int(model.encoder_normalizers["height_scan"].count) == 0
    model.update_normalization(obs)
    assert int(model.obs_normalizer.count) == NUM_ENVS
    assert int(model.encoder_normalizers["height_scan"].count) == NUM_ENVS


def test_head_norm_present_by_default_and_disabled_when_off():
    model, _ = _make_model()
    assert isinstance(model.head_norm, torch.nn.LayerNorm)
    model_off, _ = _make_model(head_layer_norm=False)
    assert isinstance(model_off.head_norm, torch.nn.Identity)


def test_state_dict_round_trip_preserves_forward():
    model, obs = _make_model(obs_normalization=True)
    model.eval()
    model.update_normalization(obs)
    expected = model(obs)

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)

    fresh, _ = _make_model(obs_normalization=True)
    fresh.eval()
    fresh.load_state_dict(torch.load(buffer, weights_only=True))
    actual = fresh(obs)

    assert torch.allclose(actual, expected, atol=1e-6)


def test_get_encoder_latents_returns_per_group_dict():
    model, obs = _make_model()
    model.eval()
    latents = model.get_encoder_latents(obs)
    assert set(latents.keys()) == {"height_scan"}
    assert latents["height_scan"].shape == (NUM_ENVS, 8)


def test_get_encoder_latents_uses_normalizer_state():
    """Updating the encoder normalizer should change the encoder latent output."""
    model, obs = _make_model(encoder_normalization=True)
    # EmpiricalNormalization.update is gated on training mode; do updates first, then eval.
    drift = torch.full((NUM_ENVS, HEIGHT_DIM), 100.0)
    for _ in range(5):
        model.encoder_normalizers["height_scan"].update(drift)
    model.eval()
    after = model.get_encoder_latents(obs)["height_scan"]

    # Build a fresh model (untrained normalizer) for the baseline.
    fresh, _ = _make_model(encoder_normalization=True)
    fresh.load_state_dict({k: v for k, v in model.state_dict().items() if "encoder_normalizers" not in k}, strict=False)
    fresh.eval()
    before = fresh.get_encoder_latents(obs)["height_scan"]

    assert not torch.allclose(before, after)


def _split_obs_for_export(model: MLPEncoderModel, obs: TensorDict) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Pre-concatenate passthrough groups and pre-flatten encoded groups in the model's expected order."""
    passthrough = torch.cat([obs[g] for g in model.obs_groups], dim=-1)
    encoded = [obs[g].flatten(start_dim=1) for g in model.obs_groups_encoded]
    return passthrough, encoded


def test_jit_export_matches_eager():
    model, obs = _make_model(obs_normalization=True)
    model.eval()
    expected = model(obs)

    jit_module = model.as_jit()
    passthrough, encoded = _split_obs_for_export(model, obs)
    actual = jit_module(passthrough, encoded)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_jit_module_is_scriptable():
    model, obs = _make_model(obs_normalization=True)
    model.eval()
    jit_module = model.as_jit()
    scripted = torch.jit.script(jit_module)
    passthrough, encoded = _split_obs_for_export(model, obs)
    expected = jit_module(passthrough, encoded)
    actual = scripted(passthrough, encoded)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_onnx_wrapper_dummy_inputs_and_names():
    model, _ = _make_model()
    onnx_module = model.as_onnx(verbose=False)
    dummies = onnx_module.get_dummy_inputs()
    assert len(dummies) == 2  # passthrough + 1 encoded
    assert dummies[0].shape == (1, POLICY_DIM + TASK_DIM)
    assert dummies[1].shape == (1, HEIGHT_DIM)
    assert onnx_module.input_names == ["obs", "height_scan"]
    assert onnx_module.output_names == ["actions"]


def test_onnx_forward_matches_eager():
    model, obs = _make_model(obs_normalization=True)
    model.eval()
    expected = model(obs)
    onnx_module = model.as_onnx(verbose=False)
    passthrough, encoded = _split_obs_for_export(model, obs)
    actual = onnx_module(passthrough, *encoded)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_missing_encoder_cfg_raises():
    obs = _make_obs()
    with pytest.raises(ValueError, match="encoder_cfg"):
        MLPEncoderModel(obs, OBS_GROUPS, "actor", output_dim=NUM_ACTIONS, encoder_cfg=None)


def test_no_encoded_groups_raises():
    obs = _make_obs()
    with pytest.raises(ValueError, match="No observation groups"):
        MLPEncoderModel(obs, OBS_GROUPS, "actor", output_dim=NUM_ACTIONS, hidden_dims=[16], encoder_cfg={})


def test_encoder_cfg_for_absent_group_raises():
    obs = _make_obs()
    with pytest.raises(ValueError, match="not present in the active"):
        MLPEncoderModel(
            obs,
            {"actor": ["policy"]},
            "actor",
            output_dim=NUM_ACTIONS,
            hidden_dims=[16],
            encoder_cfg={"height_scan": {"output_dim": 8, "hidden_dims": [16], "activation": "elu"}},
        )


def test_multidim_obs_flattened_into_encoder():
    """A (B, 1, 4, 4) heightmap should be flattened to (B, 16) before the encoder MLP."""
    obs = TensorDict(
        {
            "policy": torch.randn(NUM_ENVS, POLICY_DIM),
            "height_scan": torch.randn(NUM_ENVS, 1, 4, 4),
        },
        batch_size=[NUM_ENVS],
    )
    model = MLPEncoderModel(
        obs,
        {"actor": ["policy", "height_scan"]},
        "actor",
        output_dim=NUM_ACTIONS,
        hidden_dims=[16, 16],
        activation="elu",
        encoder_cfg={"height_scan": {"output_dim": 8, "hidden_dims": [16], "activation": "elu"}},
    )
    assert model.obs_dims_encoded == [16]
    out = model(obs)
    assert out.shape == (NUM_ENVS, NUM_ACTIONS)
