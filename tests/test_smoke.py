"""Smoke tests for the shared plumbing. Fast, no torch, no gym."""

from __future__ import annotations

import numpy as np

from rl_ladder import Run, load_metrics, plot_curves, seed_everything
from rl_ladder.plotting import aggregate, smooth


def test_seed_everything_is_reproducible():
    seed_everything(0)
    first = np.random.rand(5)
    seed_everything(0)
    assert np.allclose(first, np.random.rand(5))


def test_run_writes_config_env_and_metrics(tmp_path):
    run = Run(phase="test", name="smoke", config={"seed": 0, "lr": 0.1}, root=tmp_path)
    for step in range(3):
        run.log(step=step, reward=float(step))
    out = run.finish({"final_reward": 2.0})

    assert (out / "config.json").exists()
    assert (out / "env.json").exists()
    assert (out / "summary.json").exists()

    records = load_metrics(out)
    assert [r["step"] for r in records] == [0, 1, 2]
    assert records[-1]["reward"] == 2.0


def test_load_metrics_on_missing_run_is_empty(tmp_path):
    assert load_metrics(tmp_path / "nope") == []


def test_aggregate_truncates_to_shortest_curve():
    mean, std = aggregate([[1.0, 2.0, 3.0], [3.0, 4.0]])
    assert mean.tolist() == [2.0, 3.0]
    assert std.tolist() == [1.0, 1.0]


def test_smooth_is_a_noop_below_window():
    assert smooth([1.0, 2.0], window=5).tolist() == [1.0, 2.0]
    assert smooth([0.0, 0.0, 3.0], window=3).tolist() == [1.0]


def test_plot_curves_writes_png(tmp_path):
    out = tmp_path / "plots" / "curve.png"
    plot_curves({"a": [[0.0, 1.0, 2.0], [0.0, 2.0, 4.0]]}, title="t", out_path=out)
    assert out.exists() and out.stat().st_size > 0
