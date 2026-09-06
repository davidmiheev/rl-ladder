"""Seeding helpers.

Every experiment in this repo is run over multiple seeds; a single-seed RL curve
says almost nothing. See `docs/experiment-protocol.md`.
"""

from __future__ import annotations

import os
import random

import numpy as np


def seed_everything(seed: int, deterministic_torch: bool = False) -> int:
    """Seed `random`, `numpy`, and (if installed) `torch`.

    Args:
        seed: the seed to apply.
        deterministic_torch: also force deterministic cuDNN kernels. Slower, and
            only worth it when chasing a reproducibility bug.

    Returns:
        The seed, so it can be recorded in a run config.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
    except ImportError:
        return seed

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_torch:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    return seed
