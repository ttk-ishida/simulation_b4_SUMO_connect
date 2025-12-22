from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import yaml


def deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge src into dst (in-place) and return dst."""
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update(dst[k], v)  # type: ignore[index]
        else:
            dst[k] = v
    return dst


def deep_set(d: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set nested key given 'a.b.c' path."""
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]  # type: ignore[assignment]
    cur[keys[-1]] = value


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_merged(base_path: str, scenario_path: Optional[str] = None, scheduler_path: Optional[str] = None) -> Dict[str, Any]:
    cfg = load_yaml(base_path)
    if scenario_path:
        deep_update(cfg, load_yaml(scenario_path))
    if scheduler_path:
        deep_update(cfg, load_yaml(scheduler_path))
    return cfg


def apply_overrides(cfg: Dict[str, Any], overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply overrides like {'task_generation.mu_tasks_per_s_per_vehicle': 0.1}."""
    if not overrides:
        return cfg
    for k, v in overrides.items():
        deep_set(cfg, k, v)
    return cfg
