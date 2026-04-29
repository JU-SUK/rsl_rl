# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wall-time benchmark for the gradient-noise-scale modes.

Measures the median per-``update()`` cost of each mode on the runner-test fixture
(num_envs=4, num_steps_per_env=8, num_mini_batches=2, num_learning_epochs=2). The
goal is to give a rough sense of the metric's cost relative to baseline PPO so we
can choose the right default for production runs.

The numbers here are CPU-only and on a tiny model, so absolute timings are not
meaningful — only the *relative* overhead across modes is. Real Isaac Sim
position-task batches are ~2000x larger per minibatch, so backward-bound modes
(``within_minibatch``) cost more in absolute terms but a similar fraction of
total iter time.

Run with ``-s`` to see the timings:

.. code-block:: bash

    ./isaaclab.sh -p -m pytest rsl_rl/tests/runners/test_grad_noise_benchmark.py -v -s
"""

from __future__ import annotations

import statistics
import time
import torch

from rsl_rl.runners import OnPolicyRunner
from tests.runners.test_on_policy_runner import DummyEnv, _make_train_cfg


_NUM_STEPS_PER_ENV = 8


def _build_runner_with_cfg(grad_noise_cfg: dict | None) -> OnPolicyRunner:
    """Build a runner with the given gradient_noise_scale_cfg (or ``None`` to disable)."""
    env = DummyEnv()
    cfg = _make_train_cfg("mlp")
    if grad_noise_cfg is not None:
        cfg["algorithm"]["gradient_noise_scale_cfg"] = grad_noise_cfg
    return OnPolicyRunner(env, cfg, log_dir=None, device="cpu")


def _do_rollout(runner: OnPolicyRunner) -> torch.Tensor:
    """Run one rollout: ``num_steps_per_env`` environment steps, returning the final obs."""
    runner.alg.train_mode()
    obs = runner.env.get_observations()
    for _ in range(_NUM_STEPS_PER_ENV):
        actions = runner.alg.act(obs)
        obs, rewards, dones, extras = runner.env.step(actions)
        runner.alg.process_env_step(obs, rewards, dones, extras)
    return obs


def _time_update(grad_noise_cfg: dict | None, n_warmup: int = 3, n_measure: int = 15) -> list[float]:
    """Measure wall time of ``PPO.update()`` over ``n_measure`` calls (after ``n_warmup`` warmups)."""
    torch.manual_seed(0)
    runner = _build_runner_with_cfg(grad_noise_cfg)
    for _ in range(n_warmup):
        obs = _do_rollout(runner)
        runner.alg.compute_returns(obs)
        runner.alg.update()
    times: list[float] = []
    for _ in range(n_measure):
        obs = _do_rollout(runner)
        runner.alg.compute_returns(obs)
        start = time.perf_counter()
        runner.alg.update()
        times.append(time.perf_counter() - start)
    return times


class TestGradientNoiseScaleOverhead:
    """Per-mode wall-time overhead of the gradient-noise-scale tracker."""

    def test_benchmark_all_single_gpu_modes(self) -> None:
        """Print median + p95 wall time per ``update()`` across the single-GPU modes.

        Always passes; informational only. Use ``-s`` to see the printed table.
        ``ddp_native`` requires a real multi-GPU setup and is exercised by
        ``TestGradientNoiseScaleIntegration.test_ddp_native_smoke_two_ranks`` instead.
        """
        configurations = {
            "disabled": None,
            "across_minibatches": {"enabled": True, "mode": "across_minibatches"},
            "within_minibatch (M=2)": {"enabled": True, "mode": "within_minibatch", "num_micro_shards": 2},
            "within_minibatch (M=4)": {"enabled": True, "mode": "within_minibatch", "num_micro_shards": 4},
        }

        results: dict[str, list[float]] = {name: _time_update(cfg) for name, cfg in configurations.items()}

        baseline_median_ms = statistics.median(results["disabled"]) * 1000.0
        print()
        print(
            f"\n  Per-update wall-time (CPU, tiny model, "
            f"{_NUM_STEPS_PER_ENV} env_steps x 4 mini-batch updates per iter):"
        )
        header = f"  {'mode':>26s} | {'median (ms)':>12s} | {'p95 (ms)':>10s} | {'overhead vs disabled':>22s}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for name, times in results.items():
            median_ms = statistics.median(times) * 1000.0
            p95_ms = max(times) * 1000.0 if len(times) < 20 else statistics.quantiles(times, n=20)[18] * 1000.0
            overhead_pct = (median_ms - baseline_median_ms) / baseline_median_ms * 100.0
            print(f"  {name:>26s} | {median_ms:>12.3f} | {p95_ms:>10.3f} | {overhead_pct:>+19.1f} %")

        # Sanity: every mode must finish in well under 10x disabled (gross perf regression catch).
        for name, times in results.items():
            median = statistics.median(times)
            assert median < 10 * statistics.median(results["disabled"]), (
                f"mode {name!r} is {median / statistics.median(results['disabled']):.1f}x slower than "
                "disabled — likely a perf regression"
            )
