# Hyperparameter autotune: подробное описание для ревью

Этот файл — высокоуровневая документация по новой фиче hyperparameter autotune. Он нужен как справочный материал для ревью: что именно добавлено, как это работает, какие API появились, какие существующие файлы изменены и как запустить проверку без Ascend hardware.

## 1. Что делает фича на высоком уровне

Новая фича добавляет **второй, узкий уровень автотюнинга** поверх уже выбранного `Config` в Ascend autotuner.

Обычный autotuner уже выбирает Triton/Ascend `Config`: tiling/meta-параметры, `num_warps`, `num_stages`, UB tuning и т.п. Hyperparameter autotune не меняет этот `Config` и не добавляет параметры в kernel signature. Вместо этого он подбирает **вектор из ровно 32 целых чисел**, который конвертируется в один compiler flag:

```text
--hyper-parameters <v0> <v1> ... <v31>
```

Дальше этот flag передаётся в Ascend compiler через новый option `extra_compile_flags`.

### Цель

Цель — проверить, может ли дополнительный compiler flag улучшить performance уже выбранной kernel-конфигурации, не меняя пользовательский kernel API и не вмешиваясь в `Config.kwargs`.

### Основной поток

1. `AutoTilingTuner.run()` сначала делает обычный autotune и выбирает `self.best_config`.
2. После выбора `Config` вызывается `HyperAutotuneConfig.from_env()`.
3. Если `TRITON_ASCEND_HYPER_AUTOTUNE` не включён, всё идёт старым путём: `final_kwargs = dict(config.all_kwargs(), **kwargs)` и никаких extra flags не добавляется.
4. Если фича включена:
   - строится cache key из существующего autotune key, identity выбранного config и identity hyper-config;
   - если cache hit и не включён force mode, используется cached vector;
   - иначе `HyperparameterAutotuner` запускает Optuna search;
   - objective для каждой пробы вызывает тот же kernel/config, но с trial-specific `extra_compile_flags`;
   - лучший vector сохраняется в cache;
   - только лучший vector превращается в compiler flag и добавляется в финальный `fn.run()` через `final_kwargs["extra_compile_flags"]`.
5. Compiler path принимает `extra_compile_flags` как option, включает его в option hash/dataclass identity и добавляет flags в command line перед запуском Ascend compiler.

## 2. Почему это безопасно по умолчанию

Фича полностью выключена по умолчанию.

Если env var `TRITON_ASCEND_HYPER_AUTOTUNE` не задана или имеет false-like значение, `HyperAutotuneConfig.from_env()` возвращает disabled config:

```python
HyperAutotuneConfig(enabled=False, dim=0, max_trials=0, low=(), high=(), ...)
```

Такой config не запускает Optuna, не создаёт callback, не пишет в cache и не добавляет `extra_compile_flags`.

## 3. Environment variables

Все настройки читаются через `HyperAutotuneConfig.from_env()`.

| Env var | Назначение | Default при enabled mode |
|---|---|---|
| `TRITON_ASCEND_HYPER_AUTOTUNE` | Главный switch. `1/true/yes/on` включает фичу. | disabled |
| `TRITON_ASCEND_HYPER_AUTOTUNE_DIM` | Размерность integer-vector, который ищет Optuna. | `32` |
| `TRITON_ASCEND_HYPER_AUTOTUNE_TRIALS` | Максимальное число Optuna trials. | `16` |
| `TRITON_ASCEND_HYPER_AUTOTUNE_TIMEOUT_SEC` | Timeout для `study.optimize`. | `None` |
| `TRITON_ASCEND_HYPER_AUTOTUNE_LOW` | Нижняя граница search space. Можно одно число или comma-separated vector. | `1` для каждой из 32 позиций |
| `TRITON_ASCEND_HYPER_AUTOTUNE_HIGH` | Верхняя граница search space. Можно одно число или comma-separated vector. | `8` для каждой из 32 позиций |
| `TRITON_ASCEND_HYPER_AUTOTUNE_SEED` | Seed для `TPESampler`. Если не задан, sampler создаётся без seed. | `None` |
| `TRITON_ASCEND_HYPER_AUTOTUNE_LOG` | Включает диагностические print-сообщения. | `False` |
| `TRITON_ASCEND_HYPER_AUTOTUNE_FORCE` | Игнорирует cache и заставляет tuning запускаться заново. | `False` |

## 4. Новые файлы и их назначение

### 4.1 `third_party/ascend/backend/runtime/hyper_autotune/__init__.py`

Публичный export-файл маленького пакета.

Экспортирует:

- `HYPER_PARAMETER_COUNT`
- `HyperAutotuneConfig`
- `HyperparameterAutotuner`
- `HyperparameterTuningResult`
- `make_compiler_flags`

Назначение: дать короткий импортный путь для основных API пакета и явно зафиксировать минимальный public surface.

### 4.2 `third_party/ascend/backend/runtime/hyper_autotune/hyperparameter_config.py`

Файл отвечает только за parsing и immutable config.

Основные сущности:

#### `HyperAutotuneConfig`

Frozen dataclass с полями:

- `enabled: bool` — включена ли фича;
- `dim: int` — размерность vector;
- `max_trials: int` — bound для Optuna trials;
- `timeout_sec: Optional[float]` — bound по времени;
- `low: Tuple[int, ...]` — нижние границы;
- `high: Tuple[int, ...]` — верхние границы;
- `seed: Optional[int]` — seed для sampler;
- `log: bool` — verbose logging;
- `force: bool` — bypass cache.

#### `HyperAutotuneConfig.disabled()`

Возвращает inert disabled config. Используется для disabled-by-default behavior.

#### `HyperAutotuneConfig.from_env(env=None)`

Парсер env-переменных. Если `env` передан явно, используется он; иначе используется `os.environ`. Это удобно для unit tests и standalone scripts.

Валидация:

- `dim > 0`;
- `max_trials > 0`;
- `timeout_sec is None or timeout_sec > 0`;
- `low`/`high` должны быть либо одним числом, либо vector длины `dim`;
- каждый `low[i] <= high[i]`.

#### `HyperAutotuneConfig.cache_identity()`

Возвращает tuple, который участвует в cache key. Включает параметры, меняющие search/result identity: `dim`, `max_trials`, `timeout_sec`, `low`, `high`, `seed`.

### 4.3 `third_party/ascend/backend/runtime/hyper_autotune/hyperparameter_tuner.py`

Файл содержит Optuna-backed optimizer wrapper и compiler flag generation.

#### `import_optuna()`

Ленивый import helper. Optuna не импортируется при обычном импорте пакета. Она требуется только когда реально запускается tuning.

Если Optuna не установлена, helper поднимает понятный `RuntimeError`:

```text
TRITON_ASCEND_HYPER_AUTOTUNE requires Optuna. Install optuna or unset TRITON_ASCEND_HYPER_AUTOTUNE.
```

#### `make_compiler_flags(vector)`

Преобразует integer vector длины 32 в argv-style tuple: имя compiler flag и 32 отдельных значения:

```python
make_compiler_flags(tuple(range(1, 33)))
# ("--hyper-parameters", "1", "2", ..., "32")
```

Важно: это не kernel argument и не `Config.kwargs`; это compiler-only option.

#### `HyperparameterTuningResult`

Frozen dataclass результата tuning:

- `vector: Tuple[int, ...]` — лучший vector;
- `objective: float` — лучшее objective value.

#### `HyperparameterAutotuner`

Главный wrapper над Optuna.

Constructor:

```python
HyperparameterAutotuner(config, objective, optuna_module=None)
```

- `config` должен быть enabled;
- `objective(vector)` должен вернуть числовую стоимость, которую надо минимизировать;
- `optuna_module` существует для tests/fakes, production path оставляет его `None` и использует настоящий lazy import.

Метод `tune()`:

1. лениво импортирует Optuna;
2. создаёт `TPESampler`;
3. передаёт `seed` только если `config.seed is not None`;
4. создаёт `study` с `direction="minimize"`;
5. запускает:

```python
study.optimize(trial_objective, n_trials=config.max_trials, timeout=config.timeout_sec)
```

6. каждый trial предлагает integer vector через `trial.suggest_int(...)`;
7. если search space включает значение `1`, перед `study.optimize(...)` добавляется initial trial из 32 единиц через `study.enqueue_trial(...)`;
8. если objective падает exception или возвращает `NaN`, trial получает `float("inf")`;
9. если все trials failed/inf, поднимается `RuntimeError("Hyperparameter autotuning failed: all trials failed")`;
10. иначе возвращается `HyperparameterTuningResult`.

### 4.4 `third_party/ascend/backend/runtime/hyper_autotune/hyperparameter_cache.py`

Маленький in-memory cache.

#### `HyperparameterCacheEntry`

Frozen dataclass cache entry:

- `vector: Tuple[int, ...]`;
- `objective: float`.

#### `HyperparameterAutotuneCache`

Хранит entries в dict.

Cache key состоит из:

```python
(existing_autotune_key, hyper_config.cache_identity(), selected_config_identity)
```

Это важно, потому что один и тот же shape/autotune key может иметь разные результаты при разных bounds/trials/seed или при другом выбранном `Config`.

Public methods:

- `get(autotune_key, config, selected_config_identity)`;
- `put(autotune_key, config, selected_config_identity, result)`.

#### `get_hyperparameter_autotune_cache()`

Возвращает process-local global cache. Это intentionally in-memory cache, без disk persistence.

### 4.5 `third_party/ascend/unittest/hyper_autotune/test_hyperparameter_autotune.py`

Unit tests для нового пакета.

Покрывает:

- disabled config parsing;
- enabled env parsing;
- `make_compiler_flags`;
- fake objective tuning;
- seeded и unseeded sampler behavior;
- failed trial -> `inf`;
- all-failed trials -> clear runtime error;
- lazy Optuna error handling;
- cache key behavior;
- disabled integration-style no-op behavior.

Особенность: test file динамически загружает только `hyper_autotune` package и не импортирует весь Triton/Ascend stack. Это позволяет запускать тесты в окружении без `torch`, built Triton и Ascend runtime.

### 4.6 `tools/manual_hyper_autotune_smoke.py`

Минимальный ручной smoke script.

Что делает:

- задаёт env defaults для включения hyper autotune;
- создаёт synthetic objective;
- запускает `HyperparameterAutotuner`;
- печатает best vector, objective и compiler flags.

Требует Optuna, потому что использует настоящий production path.

### 4.7 `tools/hyper_autotune_simulation_benchmark.py`

Более подробный standalone benchmark для проверки без Ascend hardware.

Важно: это **не симуляция оптимизатора**. Скрипт использует настоящий `HyperparameterAutotuner`, а значит требует `optuna`.

Что симулируется:

- окружение/нагрузка, в которой разные compiler-vector значения дают разную synthetic performance;
- nonlinear reference workload (`sin`, `cos`, `sqrt`, `log`);
- одинаковая математическая точность для всех vectors;
- vector-dependent synthetic overhead для performance testing.

Что проверяется:

- настоящий optimizer path;
- flag generation;
- objective evaluation;
- cache round-trip;
- baseline vs tuned latency;
- точность результата относительно reference;
- минимальный speedup threshold.

### 4.8 `tools/hyper_autotune_simulation_benchmark.md`

Краткая инструкция запуска benchmark script.

Команды:

```bash
python3 -m pip install optuna
python3 tools/hyper_autotune_simulation_benchmark.py --trials 80 --repeats 5
```

## 5. Изменения в существующих файлах

### 5.1 `third_party/ascend/backend/runtime/autotuner.py`

Добавлены imports:

- `get_hyperparameter_autotune_cache`;
- `HyperAutotuneConfig`;
- `HyperparameterAutotuner`;
- `make_compiler_flags`.

Изменение в `AutoTilingTuner.run()` добавлено после:

```python
self.best_config = config
```

Поведение:

1. Парсит env через `HyperAutotuneConfig.from_env()`.
2. Если disabled — не делает ничего и сохраняет старый final path.
3. Если enabled:
   - строит `selected_config_identity = repr(config)`;
   - проверяет process-local cache;
   - при cache miss строит `compile_and_benchmark(extra_flags)`;
   - `compile_and_benchmark` запускает тот же selected config через `_make_kernel_call`, но добавляет `extra_compile_flags` только в trial kwargs;
   - запускает `HyperparameterAutotuner`;
   - сохраняет result в cache, если не `force`;
   - добавляет best flags только в final kwargs.

Что специально не делается:

- vector не добавляется в `Config.kwargs`;
- vector не становится kernel argument;
- logging не включается без `TRITON_ASCEND_HYPER_AUTOTUNE_LOG=1`;
- cache не используется при `TRITON_ASCEND_HYPER_AUTOTUNE_FORCE=1`.

### 5.2 `third_party/ascend/backend/compiler.py`

Добавлено поле в `NPUOptions`:

```python
extra_compile_flags: Tuple[str, ...] = ()
```

В `__post_init__` оно нормализуется:

- `None` -> `()`;
- list/other sequence -> `tuple(...)`.

Зачем:

- `NPUOptions` — frozen dataclass и участвует в option hash;
- tuple стабилен и hash/identity-friendly;
- caller может передать list или tuple, compiler path всегда получает tuple.

Также `list(opt.extra_compile_flags)` append-ится в `_compile_option_list` перед построением `cmd_list` в Ascend compiler entrypoints:

- `linalg_to_bin_enable_npu_compile_910_95`;
- `linalg_to_bin_enable_npu_compile_A2_A3`;
- `ttir_to_npubin`.

Зачем append именно в конце:

- это narrow hook для runtime-generated flags;
- существующие compiler options остаются в прежнем порядке;
- trial/final flags попадают в реальный command line Ascend compiler.

## 6. Новое API: краткий справочник

### `HyperAutotuneConfig`

```python
config = HyperAutotuneConfig.from_env()
if config.enabled:
    ...
```

Использовать для чтения env и передачи настроек в tuner/cache.

### `HyperAutotuneConfig.disabled()`

```python
config = HyperAutotuneConfig.disabled()
```

Создаёт inert config.

### `HyperAutotuneConfig.cache_identity()`

```python
identity = config.cache_identity()
```

Используется cache implementation. Обычно внешнему коду напрямую не нужен.

### `make_compiler_flags(vector)`

```python
flags = make_compiler_flags((1,) * 32)
# ("--hyper-parameters", "1", "1", ..., "1")
```

Единственный supported compiler flag builder на данный момент; он возвращает `--hyper-parameters` и ровно 32 значения отдельными argv элементами.

### `HyperparameterAutotuner`

```python
result = HyperparameterAutotuner(config, objective).tune()
```

`objective` принимает `Tuple[int, ...]` и возвращает float cost. Меньше — лучше.

### `HyperparameterTuningResult`

```python
result.vector
result.objective
```

Immutable result object.

### `HyperparameterAutotuneCache`

```python
cache = HyperparameterAutotuneCache()
cache.put(autotune_key, config, selected_config_identity, result)
entry = cache.get(autotune_key, config, selected_config_identity)
```

Локальный cache object.

### `get_hyperparameter_autotune_cache()`

```python
cache = get_hyperparameter_autotune_cache()
```

Process-global cache, используемый integration path.

### `NPUOptions.extra_compile_flags`

```python
fn.run(..., extra_compile_flags=("--hyper-parameters", "1", "1", ..., "1"))
```

Узкий compiler option hook. Не является kernel argument.

## 7. Как запустить проверки

### Unit tests без Optuna/Ascend

```bash
python3 -m pytest --confcutdir=third_party/ascend/unittest/hyper_autotune \
  third_party/ascend/unittest/hyper_autotune/test_hyperparameter_autotune.py -q
```

### Real optimizer synthetic benchmark с Optuna

```bash
python3 -m pip install optuna
python3 tools/hyper_autotune_simulation_benchmark.py --trials 80 --repeats 5
```

Этот benchmark требует Optuna и репозиторий, но не требует Ascend hardware.

### Minimal smoke script с Optuna

```bash
python3 -m pip install optuna
python3 tools/manual_hyper_autotune_smoke.py
```

## 8. Ограничения и важные замечания

- Cache in-memory only: при рестарте процесса tuning results теряются.
- Optuna optional: production import происходит только при enabled hyper autotune path.
- Сейчас есть только один compiler flag builder: `--hyper-parameters <32 space-separated values>`.
- Objective в real integration компилирует/бенчмаркает тот же selected config с trial-specific flags, поэтому tuning может быть дорогим.
- Если bounds включают `1`, tuner сначала ставит в очередь default vector `(1,) * 32`, чтобы начальная точка соответствовала compiler default.
- Failed trials intentionally become `inf`, чтобы единичные compile/runtime failures не останавливали весь search.
- Если все trials failed, это считается настоящей ошибкой и поднимается clear `RuntimeError`.
- Vector нигде не прокидывается как kernel argument; он существует только как compiler flag.
