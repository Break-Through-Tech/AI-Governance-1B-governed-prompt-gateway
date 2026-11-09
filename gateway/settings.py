"""Loads gateway/config.yaml once. Tests change values with `override`."""
import copy
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent / "config.yaml"
_overrides: list[dict] = []


@lru_cache(maxsize=1)
def _load() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def get(section: str) -> dict:
    """Return one section of the config, e.g. get("optimizer")."""
    cfg = _load()
    for extra in _overrides:
        cfg = _merge(cfg, extra)
    return cfg[section]


@contextmanager
def override(values: dict):
    """Temporarily change config values: `with override({"risk": {"block_threshold": 0.5}}): ...`"""
    _overrides.append(values)
    try:
        yield
    finally:
        _overrides.remove(values)
