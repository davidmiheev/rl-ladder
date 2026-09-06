"""Run directories and metric logging.

One run == one directory under `runs/<phase>/<name>__<timestamp>/` containing:

    config.json    the hyperparameters and seed the run was launched with
    env.json       git sha, python version, host, package versions worth pinning
    metrics.jsonl  one JSON object per logged step
    *.png          plots written by the experiment

`runs/` is gitignored. What gets committed is the phase README's results table
plus any plot you deliberately copy into `docs/assets/`.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() if out.returncode == 0 else "unknown"


def _git_dirty() -> bool:
    try:
        out = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=_repo_root(),
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return bool(out.stdout.strip())


@dataclass
class Run:
    """A single experiment run.

    Args:
        phase: ladder phase id, e.g. `"a1"`. Determines the `runs/` subdirectory.
        name: short run name, e.g. `"ucb1-sweep-c"`.
        config: hyperparameters to snapshot. Must be JSON-serializable.
        root: base directory for runs. Defaults to `<repo>/runs`.

    Examples:

    ```python
    >>> run = Run(phase="a1", name="eps-greedy", config={"seed": 0, "eps": 0.1})
    >>> run.log(step=0, regret=0.0)
    >>> run.finish({"final_regret": 12.3})
    ```
    """

    phase: str
    name: str
    config: dict[str, Any] = field(default_factory=dict)
    root: Path | None = None
    dir: Path = field(init=False)

    def __post_init__(self) -> None:
        root = Path(self.root) if self.root is not None else _repo_root() / "runs"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        self.dir = root / self.phase / f"{self.name}__{stamp}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self._metrics_path = self.dir / "metrics.jsonl"
        self._write_json("config.json", self.config)
        self._write_json(
            "env.json",
            {
                "git_sha": _git_sha(),
                "git_dirty": _git_dirty(),
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "argv": sys.argv,
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
        )

    def _write_json(self, filename: str, payload: dict[str, Any]) -> None:
        (self.dir / filename).write_text(json.dumps(payload, indent=2, default=str) + "\n")

    def log(self, step: int, **metrics: Any) -> None:
        """Append one record of metrics at `step` to `metrics.jsonl`."""
        record = {"step": step, "wall_time": time.time(), **metrics}
        with self._metrics_path.open("a") as handle:
            handle.write(json.dumps(record, default=str) + "\n")

    def finish(self, summary: dict[str, Any] | None = None) -> Path:
        """Write a final `summary.json` and return the run directory."""
        self._write_json("summary.json", summary or {})
        return self.dir

    def path(self, filename: str) -> Path:
        """Path inside the run directory, for plots or checkpoints."""
        return self.dir / filename


def load_metrics(run_dir: str | Path) -> list[dict[str, Any]]:
    """Read a run's `metrics.jsonl` back into a list of records."""
    path = Path(run_dir) / "metrics.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
