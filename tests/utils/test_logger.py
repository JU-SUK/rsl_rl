# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for ``Logger.state_dict`` / ``Logger.load_state_dict``."""

from __future__ import annotations

import torch
from collections import deque

import pytest

from rsl_rl.utils.logger import Logger


def _make_logger(num_envs: int = 4, rnd: bool = False, device: str = "cpu") -> Logger:
    """Construct a Logger that skips writer setup (log_dir=None)."""
    cfg = {"algorithm": {"rnd_cfg": {"foo": 1} if rnd else None}}
    return Logger(
        log_dir=None,
        cfg=cfg,
        env_cfg={},
        num_envs=num_envs,
        is_distributed=False,
        gpu_world_size=1,
        gpu_global_rank=0,
        device=device,
    )


class TestSaveLoad:
    """Tests for ``Logger.state_dict`` / ``Logger.load_state_dict``."""

    def test_round_trip_preserves_counters_and_buffers(self) -> None:
        """Persistent fields (counters + completed-episode deques) round-trip."""
        src = _make_logger(num_envs=4)
        src.tot_timesteps = 12345
        src.tot_time = 67.89
        for r, ln in [(1.0, 50), (2.5, 60), (0.3, 40)]:
            src.rewbuffer.append(r)
            src.lenbuffer.append(ln)

        snapshot = src.state_dict()

        dst = _make_logger(num_envs=4)
        dst.load_state_dict(snapshot)

        assert dst.tot_timesteps == 12345
        assert dst.tot_time == pytest.approx(67.89)
        assert list(dst.rewbuffer) == [1.0, 2.5, 0.3]
        assert list(dst.lenbuffer) == [50, 60, 40]

    def test_current_episode_counters_not_persisted(self) -> None:
        """``cur_reward_sum`` / ``cur_episode_length`` are excluded by design (refill on resume)."""
        src = _make_logger(num_envs=4)
        src.cur_reward_sum[:] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        src.cur_episode_length[:] = torch.tensor([10.0, 20.0, 30.0, 40.0])

        snapshot = src.state_dict()
        assert "cur_reward_sum" not in snapshot
        assert "cur_episode_length" not in snapshot

        dst = _make_logger(num_envs=4)
        dst.load_state_dict(snapshot)
        # dst's accumulators stay at their freshly-constructed zeros.
        assert torch.equal(dst.cur_reward_sum, torch.zeros(4))
        assert torch.equal(dst.cur_episode_length, torch.zeros(4))

    def test_rnd_completed_episode_buffers_round_trip(self) -> None:
        """When ``rnd_cfg`` is set, ``erewbuffer`` and ``irewbuffer`` round-trip."""
        src = _make_logger(num_envs=2, rnd=True)
        src.erewbuffer.append(0.5)
        src.irewbuffer.append(0.7)
        src.cur_ereward_sum[:] = torch.tensor([0.1, 0.2])
        src.cur_ireward_sum[:] = torch.tensor([0.3, 0.4])

        snapshot = src.state_dict()
        assert "erewbuffer" in snapshot
        assert "irewbuffer" in snapshot
        # Per-step RND reward sums are excluded for the same reason as cur_reward_sum.
        assert "cur_ereward_sum" not in snapshot
        assert "cur_ireward_sum" not in snapshot

        dst = _make_logger(num_envs=2, rnd=True)
        dst.load_state_dict(snapshot)
        assert list(dst.erewbuffer) == [0.5]
        assert list(dst.irewbuffer) == [0.7]
        assert torch.equal(dst.cur_ereward_sum, torch.zeros(2))
        assert torch.equal(dst.cur_ireward_sum, torch.zeros(2))

    def test_no_rnd_fields_when_rnd_disabled(self) -> None:
        """A non-RND logger's snapshot must not include the RND-only keys."""
        src = _make_logger(rnd=False)
        snap = src.state_dict()
        assert "erewbuffer" not in snap

    def test_deque_maxlen_preserved_on_load(self) -> None:
        """``load_state_dict`` honors the destination's deque maxlen."""
        src = _make_logger()
        for i in range(150):
            src.rewbuffer.append(float(i))
        # Source deque is full at maxlen=100.
        assert len(src.rewbuffer) == 100
        snap = src.state_dict()
        assert len(snap["rewbuffer"]) == 100

        dst = _make_logger()
        dst.load_state_dict(snap)
        assert isinstance(dst.rewbuffer, deque)
        assert dst.rewbuffer.maxlen == 100
        assert list(dst.rewbuffer) == [float(i) for i in range(50, 150)]
