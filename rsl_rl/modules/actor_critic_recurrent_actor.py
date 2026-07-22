# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict
from typing import cast

from rsl_rl.networks import MLP, HiddenState
from rsl_rl.utils import unpad_trajectories

from .actor_critic_recurrent import ActorCriticRecurrent


class ActorCriticRecurrentActor(ActorCriticRecurrent):
    """Recurrent ACTOR + feed-forward CRITIC (asymmetric memory).

    Same actor path as :class:`ActorCriticRecurrent` (obs -> RNN -> MLP), but the critic is a
    plain MLP on the critic observations — no critic RNN. Rationale: the critic is train-time
    only and can be fed privileged observations that restore the Markov property, so it does
    not need memory; dropping its RNN removes the value-side BPTT and matches the
    "LSTM actor + MLP critic" configuration used by e.g. play2perfect.

    Integration notes (why the overrides below exist):
    * ``get_hidden_states`` must still return a *pair* — ``RolloutStorage._save_hidden_states``
      only skips ``(None, None)`` and otherwise allocates buffers from each tensor's shape, and
      ``PPO.update`` indexes ``hidden_states_batch[1]``. We return a tiny zeros placeholder of
      shape (num_layers, num_envs, 1) for the critic slot: the storage/generator only ever
      manipulate the time/layer/env dims, so the feature size of 1 flows through, costing
      ~num_envs floats per step instead of a full duplicated hidden state.
    * ``evaluate`` receives PADDED trajectory batches during the PPO update (the recurrent
      mini-batch generator pads trajectories and passes ``masks``). The parent's critic RNN
      unpadded internally via ``Memory``; a plain MLP does not — so we call
      ``unpad_trajectories`` ourselves. Skipping this would misalign values with returns
      (padding rows entering the value loss) without necessarily raising an error.
    """

    is_recurrent: bool = True

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        critic_hidden_dims: tuple[int] | list[int] = [256, 256, 256],
        activation: str = "elu",
        rnn_layer_norm: bool = False,
        **kwargs,
    ) -> None:
        # Build the full recurrent actor-critic first (actor RNN+MLP, normalizers, noise),
        # then replace the critic path. The temporarily-built critic RNN is discarded below.
        super().__init__(
            obs,
            obs_groups,
            num_actions,
            critic_hidden_dims=critic_hidden_dims,
            activation=activation,
            **kwargs,
        )

        # Optional LayerNorm on the actor LSTM output before the MLP (play2perfect uses this):
        # normalizes the recurrent features so the MLP sees a stable, well-scaled input from
        # step 0 (the LSTM output is near-zero early in training, collapsing state-conditioned
        # action diversity). Identity when disabled -> no behavior change.
        rnn_hidden_dim = self.memory_a.rnn.hidden_size
        self.actor_ln = torch.nn.LayerNorm(rnn_hidden_dim) if rnn_layer_norm else torch.nn.Identity()
        if rnn_layer_norm:
            print(f"Actor LSTM output LayerNorm: {self.actor_ln}")

        # Replace the critic: plain MLP on the raw critic observations, no memory.
        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            num_critic_obs += obs[obs_group].shape[-1]
        del self.memory_c  # unregister the parent's critic RNN
        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"Critic RNN: None (feed-forward critic on {num_critic_obs} obs)")
        print(f"Critic MLP: {self.critic}")

        # Lazily-created placeholder for the critic slot of the hidden-state pair (see class doc).
        self._critic_dummy_hidden: HiddenState = None

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.memory_a.reset(dones)

    def act(self, obs: TensorDict, masks=None, hidden_state: HiddenState = None) -> torch.Tensor:
        # Same as the parent, with LayerNorm applied to the LSTM output before the actor MLP.
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        out_mem = self.actor_ln(self.memory_a(actor_obs, masks, hidden_state).squeeze(0))
        self._update_distribution(out_mem)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        out_mem = self.actor_ln(self.memory_a(actor_obs).squeeze(0))
        if self.state_dependent_std:
            return self.actor(out_mem)[..., 0, :]
        return self.actor(out_mem)

    def evaluate(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        # hidden_state is accepted for interface parity with the recurrent critic and ignored.
        obs = self.get_critic_obs(obs)
        obs = self.critic_obs_normalizer(obs)
        if masks is not None:
            # PPO-update mode: obs is a padded (time, num_trajs, dim) trajectory batch; drop the
            # padding so the values align row-for-row with the (time, envs, 1) returns batch.
            obs = unpad_trajectories(obs, masks)
        return self.critic(obs)

    def get_hidden_states(self) -> tuple[HiddenState, HiddenState]:
        # cast: Memory.hidden_state is annotated loosely upstream (variadic tuple), but at
        # runtime it is exactly an (h, c) pair for LSTM / a bare tensor for GRU / None.
        hidden_a = cast(HiddenState, self.memory_a.hidden_state)
        if hidden_a is None:
            # Before the first forward (parent behaves identically): storage skips (None, None)
            # and later fills the step-0 slot with zeros == the true initial hidden state.
            return None, None
        # The storage copy loop iterates ``range(len(hidden_a))`` over BOTH slots, so the
        # critic placeholder must mirror the actor hidden's structure — an (h, c) pair of
        # tiny (layers, envs, 1) zeros for LSTM, a single one for GRU.
        ref = hidden_a[0] if isinstance(hidden_a, tuple) else hidden_a  # (layers, envs, hidden)
        dummy = self._critic_dummy_hidden
        dummy_ref = dummy[0] if isinstance(dummy, tuple) else dummy
        if (
            dummy_ref is None
            or isinstance(dummy, tuple) != isinstance(hidden_a, tuple)
            or dummy_ref.shape[:2] != ref.shape[:2]
            or dummy_ref.device != ref.device
        ):
            z = torch.zeros(ref.shape[0], ref.shape[1], 1, device=ref.device, dtype=ref.dtype)
            self._critic_dummy_hidden = (z, z.clone()) if isinstance(hidden_a, tuple) else z
        return hidden_a, self._critic_dummy_hidden
