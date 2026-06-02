# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional, Tuple

_ENV_PREFIX = "TRITON_ASCEND_HYPER_AUTOTUNE"
_TRUE_VALUES = {"1", "true", "yes", "on"}
_FALSE_VALUES = {"0", "false", "no", "off", ""}


def _parse_bool(value: Optional[str], *, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}")


def _parse_int(value: Optional[str], name: str, default: int) -> int:
    if value is None or value.strip() == "":
        return default
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {value!r}") from exc


def _parse_optional_float(value: Optional[str], name: str) -> Optional[float]:
    if value is None or value.strip() == "":
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got {value!r}") from exc


def _parse_bounds(value: Optional[str], name: str, dim: int, default: int) -> Tuple[int, ...]:
    if value is None or value.strip() == "":
        return (default,) * dim
    parts = [part.strip() for part in value.split(",") if part.strip() != ""]
    if len(parts) == 1:
        parsed = _parse_int(parts[0], name, default)
        return (parsed,) * dim
    if len(parts) != dim:
        raise ValueError(f"{name} must contain either one value or {dim} comma-separated values")
    return tuple(_parse_int(part, name, default) for part in parts)


@dataclass(frozen=True)
class HyperAutotuneConfig:
    enabled: bool = False
    dim: int = 0
    max_trials: int = 0
    timeout_sec: Optional[float] = None
    low: Tuple[int, ...] = ()
    high: Tuple[int, ...] = ()
    seed: Optional[int] = None
    log: bool = False
    force: bool = False

    @classmethod
    def disabled(cls) -> "HyperAutotuneConfig":
        return cls()

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "HyperAutotuneConfig":
        env_map = os.environ if env is None else env
        enabled = _parse_bool(env_map.get(_ENV_PREFIX), default=False)
        log = _parse_bool(env_map.get(f"{_ENV_PREFIX}_LOG"), default=False)
        force = _parse_bool(env_map.get(f"{_ENV_PREFIX}_FORCE"), default=False)
        if not enabled:
            return cls.disabled()

        dim = _parse_int(env_map.get(f"{_ENV_PREFIX}_DIM"), f"{_ENV_PREFIX}_DIM", 1)
        if dim <= 0:
            raise ValueError(f"{_ENV_PREFIX}_DIM must be greater than zero")

        max_trials = _parse_int(env_map.get(f"{_ENV_PREFIX}_TRIALS"), f"{_ENV_PREFIX}_TRIALS", 16)
        if max_trials <= 0:
            raise ValueError(f"{_ENV_PREFIX}_TRIALS must be greater than zero")

        timeout_sec = _parse_optional_float(env_map.get(f"{_ENV_PREFIX}_TIMEOUT_SEC"), f"{_ENV_PREFIX}_TIMEOUT_SEC")
        if timeout_sec is not None and timeout_sec <= 0:
            raise ValueError(f"{_ENV_PREFIX}_TIMEOUT_SEC must be greater than zero")

        low = _parse_bounds(env_map.get(f"{_ENV_PREFIX}_LOW"), f"{_ENV_PREFIX}_LOW", dim, 1)
        high = _parse_bounds(env_map.get(f"{_ENV_PREFIX}_HIGH"), f"{_ENV_PREFIX}_HIGH", dim, 8)
        for index, (lo, hi) in enumerate(zip(low, high)):
            if lo > hi:
                raise ValueError(f"hyper autotune lower bound at index {index} exceeds upper bound")

        seed_value = env_map.get(f"{_ENV_PREFIX}_SEED")
        seed = None if seed_value is None or seed_value.strip() == "" else _parse_int(seed_value, f"{_ENV_PREFIX}_SEED", 0)

        return cls(
            enabled=enabled,
            dim=dim,
            max_trials=max_trials,
            timeout_sec=timeout_sec,
            low=low,
            high=high,
            seed=seed,
            log=log,
            force=force,
        )

    def cache_identity(self) -> Tuple[object, ...]:
        return (self.dim, self.max_trials, self.timeout_sec, self.low, self.high, self.seed)
