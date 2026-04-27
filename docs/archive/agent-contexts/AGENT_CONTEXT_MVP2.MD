# AGENT_CONTEXT_MVP2.MD

Останнє оновлення: 2026-03-21  
Статус: deprecated-for-agent-first-read; MVP2 historical compatibility context.

> **Deprecated / Redirect (2026-04-27):**
> This file is historical context and is no longer part of the default
> first-read path for agents.
>
> Current agent entrypoint:
> 1. `AGENTS.MD`
> 2. `docs/current/project-state.md`
> 3. `docs/current/architecture-debt.md`
>
> Use this file only for MVP2 backward-compatibility reasoning when explicitly
> routed by `AGENTS.MD` or requested by the user.

## 1) Що зафіксовано в MVP2

1. Додано knowledge-injection шлях через `IKnowledgeGraphPort`.
2. `DynamicGraphBuilder` формує:
   - `allowed_target_mask`,
   - `structural_edge_index`,
   - `structural_edge_weight`,
   - optional `struct_x`.
3. Модельний шлях MVP2: `EOPKGGATv2` (dual-encoder + cross-attention).
4. Оцінка drift/OOS і slicing по version/prefix довжинах.

## 2) Архітектурні інваріанти MVP2

1. MVP2 має лишатися additive відносно MVP1:
   - усі розширення `GraphTensorContract` тільки optional.
2. Anti-leakage:
   - топологія/структура будується тільки по train-split.
3. Fallback:
   - відсутність структурних тензорів не ламає forward, активується baseline-path.
4. Drift-window policy:
   - крок = `drift_window_sliding` або `drift_window_size`,
   - короткий tail (`len(window) < size`) відкидається.

## 3) Канонічний MVP2 runtime-шлях

`XES/Camunda Trace -> RawTrace -> PrefixPolicy -> PrefixSlice -> DynamicGraphBuilder -> GraphTensorContract -> ModelTrainer -> EOPKGGATv2/Baseline`

## 4) Сумісність із MVP2.5

1. У MVP2.5 knowledge джерело стало repository-driven (file/neo4j/in_memory), але контрактно сумісне з MVP2.
2. `strict_asof` та stats snapshots не змінюють MVP2 baseline інваріант:
   - за відсутності валідних stats/structure шлях не падає.
3. Нові статистичні поля в контрактах і логуванні є розширенням, а не заміною MVP2 механік.

## 5) Мінімальні перевірки для безпечних змін у MVP2-зоні

1. `.\.venv\Scripts\python.exe tools\architecture_guard.py`
2. `.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v`
3. Цільові тести MVP2/MVP2.5:
   - `tests/domain/test_dynamic_graph_builder_*.py`
   - `tests/application/test_window_generator.py`
   - `tests/application/test_evaluator_metrics.py`
