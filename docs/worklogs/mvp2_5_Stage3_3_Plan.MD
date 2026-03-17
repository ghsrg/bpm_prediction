# MVP2.5 Stage 3.3 Detailed Plan

Date: 2026-03-13  
Status: Planned  
Scope: Periodic Statistics & Version Refresh Service (Camunda-only, GNN-first)

## 0. Джерела істини

1. `mvp2_5_Stage3_Plan.MD`
2. `mvp2_5_Stage3_1_Plan.MD`
3. `mvp2_5_Stage3_2_Plan.MD`
4. `docs/TARGET_ARCHITECTURE.MD`
5. `docs/DATA_MODEL_MVP2_5.MD`
6. `AGENT_GUIDE.MD`

---

## 1. Мета Stage 3.3

Побудувати **Camunda-only, GNN-орієнтований feature store** поверх `ProcessStructureDTO`, який:
1. періодично оновлює статистику з runtime даних,
2. витримує history cleanup та часткову доступність історії,
3. формує корисні precomputed ознаки для GNN,
4. створює immutable snapshots для аналізу еволюції процесу у дисертації.

Ключовий принцип: **жодного прямого зв’язку з BPMS або cross-process графами** у Stage 3.3.

---

## 2. Архітектурні обмеження (Clean Architecture)

## 2.1 Дозволено

1. `ProcessStatisticsRefreshService` у application layer.
2. Читання з `IInstanceGraphPort` (результати Stage 3.1).
3. Читання/запис у `IKnowledgeGraphPort` (структура + snapshots Stage 3.2/3.3).

## 2.2 Заборонено

1. Будь-який прямий доступ до BPMS adapter/портів.
2. Будь-які cross-process агрегації на рівні Stage 3.3.
3. Runtime recompute під час train/eval/infer.

## 2.3 Ізоляція від enrichment

1. `enrichment.enabled=false` не блокує Stage 3.3.
2. Stage 3.3 працює повністю в Camunda-only контурі.
3. Майбутній enrichment підключається окремо через порти, без змін ядра Stage 3.3.

---

## 3. Об’єкти та контракти

## 3.1 Порти (план змін)

1. `IInstanceGraphPort`
   - `get_instance_events(process_name, version_key, since, until)`
   - `get_last_watermark(process_name, version_key)`
   - `set_last_watermark(process_name, version_key, watermark)`

2. `IKnowledgeGraphPort` (розширення під snapshots)
   - `get_latest_process_structure(process_name, version_key)`
   - `append_process_structure_snapshot(process_name, version_key, dto, knowledge_version, snapshot_ts)`
   - `get_previous_snapshot(process_name, version_key)`
   - `get_previous_process_version_snapshot(process_name, prev_version_key)`

## 3.2 DTO розширення

### ProcessStructureDTO (нові секції)

1. `node_stats.windows`
2. `edge_stats.windows`
3. `gnn_features`
4. `version_drift`
5. `confidence`
6. `freshness`
7. `metadata.version_stability`
8. `metadata.knowledge_version`
9. `metadata.snapshot_ts`

### Приблизний фрагмент

```python
ProcessStructureDTO:
  node_stats:
    windows:
      last_7d: ...
      last_30d: ...
      last_90d: ...
      all_time_decay: ...
  edge_stats:
    windows:
      last_7d: ...
      last_30d: ...
      last_90d: ...
      all_time_decay: ...
  gnn_features:
    transition_probability: Dict[str, Dict[str, float]]
    node_features: Dict[str, Dict[str, float]]
    loop_density: Dict[str, float]
    condition_complexity_index: Dict[str, float]
  version_drift:
    duration_delta_pct: float
    loop_change_pct: float
    parallel_burst_delta: float
  metadata:
    version_stability: Literal["stable", "moderate_drift", "high_drift"]
    knowledge_version: str
    snapshot_ts: str
```

---

## 4. Конфігурація Stage 3.3

```yaml
stats_refresh:
  enabled: true
  cron: "*/30 * * * *"
  lookback_hours: 24
  process_filters: ["procurement"]

  # Core flags
  snapshot_enabled: true
  immutable_snapshots: true
  history_cleanup_aware: true

  # Multi-window + decay
  windows_days: [7, 30, 90]
  enable_decay: true
  decay_factor_per_day: 0.995
  decay_window_name: "all_time_decay"

  # GNN-oriented features
  enable_gnn_features: true
  enable_transition_probabilities: true
  enable_condition_complexity: true

  # Robust stats
  enable_outlier_robust_stats: true
  winsorize_limits: [0.05, 0.95]

  # Drift and stability
  enable_version_drift: true
  stability_thresholds:
    moderate: 0.15
    high: 0.35

  # Confidence/Freshness
  enable_confidence_scoring: true
  freshness_half_life_days: 14
  confidence_weights:
    sample_size: 0.4
    freshness: 0.3
    coverage: 0.3

  # Resilience
  min_coverage_for_update: 0.05
  low_coverage_policy: "snapshot_with_warning"  # snapshot_with_warning | skip
```

Примітка:
1. Якщо `snapshot_enabled=false`, сервіс може оновлювати latest-view без append (тільки для локальних debug-сценаріїв).
2. Для прод-режиму: `snapshot_enabled=true` + `immutable_snapshots=true`.

---

## 5. Алгоритм Stage 3.3 (Refresh Pipeline)

## 5.1 Step A: Вибір scope

1. Ітерація по `(process_name, version_key)`.
2. Жодних cross-process merge.

## 5.2 Step B: Інкрементальне читання

1. Взяти `last_watermark`.
2. Прочитати події `since = watermark - lookback`.
3. Використати cleanup-aware diagnostics із Stage 3.1.

## 5.3 Step C: Формування multi-window датасетів

1. `last_7d`, `last_30d`, `last_90d`.
2. `all_time_decay` = попередній decayed snapshot + нові події.

### Формула decay (по днях)

Нехай:
1. `d = days_since_prev_snapshot`
2. `alpha = decay_factor_per_day ** d`

Тоді для будь-якого накопичуваного показника `S`:
1. `S_new = alpha * S_prev + S_increment`

Для середніх:
1. зберігати `(weighted_sum, weighted_count)`
2. `weighted_sum_new = alpha * weighted_sum_prev + increment_sum`
3. `weighted_count_new = alpha * weighted_count_prev + increment_count`
4. `mean_new = weighted_sum_new / max(weighted_count_new, eps)`

## 5.4 Step D: Outlier-robust статистика

На кожному вікні для node/edge:
1. `median_duration`
2. `iqr_duration`
3. `p95_duration`
4. `winsorized_mean`
5. `skewness`
6. `kurtosis`

Правило:
1. Якщо sample size малий (наприклад < 5), частину метрик маркувати `None` + знижувати confidence.

## 5.5 Step E: GNN-first precomputed features

### Transition probability

1. `P(target|source) = count(source->target) / sum_target count(source->*)`
2. Зберігати sparse dict:
   - `transition_probability[source][target] = prob`

### Node features

1. `in_degree_norm = in_degree / max(1, N-1)`
2. `out_degree_norm = out_degree / max(1, N-1)`
3. `parallel_token_variance_30d`

### Loop density

1. Для activity `a`: середня кількість повторів `a` в межах одного instance.

### Condition complexity index (0..1)

1. Базується на `condition_expr` (зі Stage 3.2):
   - кількість змінних,
   - логічних операторів,
   - функцій/викликів,
   - довжина виразу.
2. Нормалізується в [0,1].

## 5.6 Step F: Version drift indicators

Порівнювати поточну статистику з:
1. попереднім snapshot тієї ж версії (короткострокова динаміка),
2. попередньою версією процесу (якщо є) для version-shift.

Зберігати:
1. `duration_delta_pct`
2. `loop_change_pct`
3. `parallel_burst_delta`

Класифікація stability:
1. `stable`
2. `moderate_drift`
3. `high_drift`

## 5.7 Step G: Confidence & freshness scoring

Для кожного node/edge у кожному вікні:
1. `sample_size`
2. `freshness_score` в [0,1]
3. `confidence_score = f(sample_size, freshness, coverage)`

### Рекомендована формула

1. `sample_component = log1p(sample_size) / log1p(sample_ref)`
2. `freshness_score = exp(-ln(2) * age_days / half_life_days)`
3. `confidence_score = clamp(w1*sample_component + w2*freshness_score + w3*coverage, 0, 1)`

## 5.8 Step H: Immutable snapshot write

1. Згенерувати `knowledge_version` (наприклад: `{process_name}:{version_key}:{snapshot_ts}`)
2. Додати `snapshot_ts`.
3. Викликати `append_process_structure_snapshot(...)`.
4. Оновити watermark тільки після успішного snapshot write.

---

## 6. Failure/Resilience Policy

## 6.1 Якщо даних мало (cleanup вплинув)

1. Не падати.
2. При `coverage < min_coverage_for_update`:
   - `snapshot_with_warning`: створити snapshot з прапором low_coverage + нижчим confidence,
   - або `skip` (за конфігом).

## 6.2 Якщо нових подій немає

1. Створити lightweight snapshot (опціонально) з оновленим freshness тільки,
2. або пропустити snapshot, але зафіксувати heartbeat лог/метрику.

## 6.3 Якщо частина метрик не обчислюється

1. Заповнювати частково (`None` для неможливих полів).
2. Обов’язково проставляти diagnostics в metadata.

---

## 7. Структура metadata для дисертації

`metadata` рекомендується зберігати:
1. `knowledge_version`
2. `snapshot_ts`
3. `window_config`
4. `decay_factor_per_day`
5. `refresh_coverage_percent`
6. `version_stability`
7. `warnings_count`
8. `fallback_triggered` (якщо прийшли degraded дані з Stage 3.1)

Це забезпечує відтворюваність експериментів і графіків еволюції.

---

## 8. File-Level Plan

## 8.1 Нові файли (план)

1. `src/application/services/process_statistics_refresh_service.py`
2. `src/domain/entities/process_statistics_snapshot.py`
3. `src/domain/services/statistics_aggregator.py`
4. `src/domain/services/gnn_feature_precompute_service.py`
5. `src/domain/services/version_drift_analyzer.py`
6. `src/domain/services/confidence_scorer.py`

## 8.2 Оновлення існуючих

1. `src/domain/entities/process_structure.py`
   - додати `node_stats`, `edge_stats`, `gnn_features`, `version_drift`, `metadata` поля.
2. `src/domain/ports/knowledge_graph_port.py`
   - додати append/get snapshot methods.
3. `src/cli.py`
   - додати конфіг-прапори Stage 3.3.
4. `configs/experiments/mvp2_*.yaml`
   - додати `stats_refresh` секцію.
5. `docs/DATA_MODEL_MVP2_5.MD`
   - зафіксувати новий контракт DTO.
6. `docs/DATA_FLOWS_MVP2_5.MD`
   - додати flow Stage 3.3.

---

## 9. Тестовий план Stage 3.3

## 9.1 Unit: decay & windows

1. `test_decay_all_time_formula`
2. `test_multi_window_partition_7_30_90`
3. `test_decay_with_time_gap_days`

## 9.2 Unit: robust stats

1. `test_robust_stats_median_iqr_p95`
2. `test_winsorized_mean_outlier_resistance`
3. `test_skewness_kurtosis_small_sample_guard`

## 9.3 Unit: gnn features

1. `test_transition_probability_normalization`
2. `test_degree_norm_bounds`
3. `test_loop_density_computation`
4. `test_condition_complexity_index_range`

## 9.4 Unit: drift & stability

1. `test_version_drift_delta_computation`
2. `test_version_stability_classification`

## 9.5 Unit: confidence/freshness

1. `test_freshness_decay_half_life`
2. `test_confidence_score_bounds`
3. `test_confidence_drops_on_low_coverage`

## 9.6 Integration

1. `test_snapshot_append_immutable`
2. `test_watermark_updates_only_after_successful_snapshot`
3. `test_low_coverage_snapshot_with_warning`
4. `test_camunda_only_no_bpms_dependency`

## 9.7 Regression

1. `pytest -m mvp1_regression -v`
2. `pytest tests/domain/test_eopkg_models_forward.py tests/domain/test_eopkg_gatv2_dual_encoder.py -v`
3. `pytest tests/ -v`

---

## 10. Definition of Done (Stage 3.3)

1. Multi-window statistics (`7d`, `30d`, `90d`, `all_time_decay`) доступні для node/edge.
2. `gnn_features` заповнюється Camunda-only даними.
3. `version_drift` + `version_stability` коректно обчислюються.
4. Outlier-robust метрики працюють стабільно.
5. `freshness_score` і `confidence_score` присутні для node/edge.
6. Snapshot model immutable, з `knowledge_version` і `snapshot_ts`.
7. Сервіс не падає при cleanup/low coverage, має warning/degraded поведінку.
8. Відсутні залежності від BPMS/cross-process logic у Stage 3.3.
9. Наявна повна відтворюваність для дослідницького аналізу у дисертації.

---

## 11. Рекомендована стратегія запуску (перший rollout)

1. Увімкнути:
   - `enable_decay=true`
   - `enable_gnn_features=true`
   - `enable_outlier_robust_stats=true`
   - `enable_version_drift=true`
   - `enable_confidence_scoring=true`
   - `snapshot_enabled=true`
2. Почати з `cron` кожні 30 хвилин.
3. Спочатку запускати в dry-run режимі (без commit snapshot), порівняти метрики.
4. Потім перейти на immutable snapshots у production режимі.

