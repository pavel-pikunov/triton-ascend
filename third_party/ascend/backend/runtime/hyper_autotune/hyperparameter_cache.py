# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Optional, Tuple

from .hyperparameter_config import HyperAutotuneConfig
from .hyperparameter_tuner import HyperparameterTuningResult


@dataclass(frozen=True)
class HyperparameterCacheEntry:
    vector: Tuple[int, ...]
    objective: float


class HyperparameterAutotuneCache:
    def __init__(self):
        self._entries: Dict[Tuple[Hashable, Tuple[object, ...], Hashable], HyperparameterCacheEntry] = {}

    def _make_key(self, autotune_key: Hashable, config: HyperAutotuneConfig, selected_config_identity: Hashable):
        return (autotune_key, config.cache_identity(), selected_config_identity)

    def get(
        self,
        autotune_key: Hashable,
        config: HyperAutotuneConfig,
        selected_config_identity: Hashable,
    ) -> Optional[HyperparameterCacheEntry]:
        return self._entries.get(self._make_key(autotune_key, config, selected_config_identity))

    def put(
        self,
        autotune_key: Hashable,
        config: HyperAutotuneConfig,
        selected_config_identity: Hashable,
        result: HyperparameterTuningResult,
    ) -> None:
        self._entries[self._make_key(autotune_key, config, selected_config_identity)] = HyperparameterCacheEntry(
            vector=result.vector,
            objective=result.objective,
        )


_GLOBAL_CACHE = HyperparameterAutotuneCache()


def get_hyperparameter_autotune_cache() -> HyperparameterAutotuneCache:
    return _GLOBAL_CACHE
