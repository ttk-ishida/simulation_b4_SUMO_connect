from __future__ import annotations

import os
import random
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SeedState:
    seed: int


def set_global_seed(seed: int) -> SeedState:
    """Set seeds for random, numpy and hash.
    This makes simulation runs reproducible.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return SeedState(seed=seed)
