# Analyse_before_mvp2.md

Оновлено: 2026-03-11  
Призначення: аудит перед імплементацією MVP2 + фактичний результат Спринту 1.

## 1. Початкові невідповідності (до Спринту 1)

1. MVP2 docs вимагали `EOPKGFusionGNN`, але в runtime registry були лише `BaselineGCN` і `BaselineGATv2`.
2. MVP2 docs описували розширений `GraphTensorContract` (structural поля), а код використовував MVP1-only контракт.
3. Drift-вікна були на legacy-логіці (`stride/overlap`), при цьому `drift_window_sliding` фактично не був головним драйвером.
4. Unit-тести генератора вікон закріплювали legacy-поведінку (ігнор sliding і short tail).
5. Залишався застарілий текст помилки `Unsupported data.split_strategy ...` після перенесення split-параметрів у `experiment`.
6. Було розходження в імені файла правил архітектури: `ARCHITECTURE_RULES.md` vs `ARCHITECTURE_RULES.MD`.
7. README мав проблему з кодуванням (mojibake).

## 2. Архітектурні рішення (отримані від користувача)

1. MVP2 Phase-1: лише `EOPKGFusionGNN + TopologyExtractorService` (structural injection).
2. Critic, Semaphore, dynamic regularization (`beta`) — виключно MVP3.
3. Джерело знань для MVP2: тільки `TopologyExtractorService` (in-memory, train-derived topology), без Neo4j.
4. Drift windows: повне видалення `stride/overlap`, залишаємо `size/sliding`.
5. Короткий хвіст у drift-вікнах (`len(window) < size`) обов’язково відкидається.
6. `eval_drift`: без додаткового micro-split.
7. Метрики MVP2: F1, ECE, OOS + slices (length/version).
8. Конфіги MVP2 — ізольовано в `configs/experiments/mvp2_*.yaml`; MVP1 конфіги не ламаємо.
9. Контракт-застібка для MVP2: `ARCHITECTURE_MVP2.MD`, `DATA_MODEL_MVP2.MD`, `EVF_MVP2.MD`, `LLD_MVP2.MD`.

## 3. Детальний результат Спринту 1 (виконано)

### 3.1 Cleanup та документація

1. README перезаписано в чистому UTF-8, з актуальним індексом документації та статусом MVP1/MVP2/MVP3.
2. `docs/ARCHITECTURE_RULES.md` перейменовано на `docs/ARCHITECTURE_RULES.MD` (канонізовано регістр імені).
3. Посилання на `ARCHITECTURE_RULES.MD` синхронізовано в коді/документації.
4. `docs/DATA_MODEL_MVP1.MD` оновлено: прибрано `drift_window_stride` і `drift_window_overlap`, залишено політику `size + sliding`.

### 3.2 Логіка Drift Windows

1. Видалено використання `drift_window_stride` / `drift_window_overlap` з runtime-конфігурації trainer.
2. `ModelTrainer._resolve_drift_step()` тепер:
   - повертає `drift_window_sliding`, якщо `> 0`;
   - інакше повертає `drift_window_size` (tumbling fallback).
3. `ModelTrainer._generate_drift_windows()` тепер відкидає short tail:
   - вікно додається лише якщо `len(window) >= size`.
4. Логування drift-режиму оновлено на явну ознаку `keep_short_tail=false`.

### 3.3 Виправлення валідатора

1. Текст помилки змінено:
   - було: `Unsupported data.split_strategy ...`
   - стало: `Unsupported experiment.split_strategy ...`

### 3.4 Тести

1. Оновлено unit-тести `tests/application/test_window_generator.py`:
   - перевірка sliding-кроку (`size=5`, `sliding=2`) з очікуваними вікнами та без короткого хвоста;
   - перевірка tumbling fallback при `sliding=0`;
   - окрема перевірка відкидання короткого хвоста у tumbling режимі.
2. Прогін тестів:
   - команда: `.\.venv\Scripts\python.exe -m pytest tests/ -v`
   - результат: `27 passed`.
3. Додатковий gate:
   - команда: `py -3 tools/architecture_guard.py`
   - результат: `OK`.

## 4. Поточний стан після Спринту 1

1. Спринт 1 завершено успішно: cleanup + sliding windows + non-regression.
2. MVP1 тестовий контур зелений.
3. База готова до переходу на Спринт 2 (імплементація MVP2 Phase-1 компонентів).

## 5. Технічні нотатки

1. Під час масового текстового оновлення тимчасово з’явився BOM у частині файлів; це виправлено (BOM прибрано).
2. Автоматичне видалення `__pycache__` у цьому середовищі було заблоковане policy, тому кеш-директорії могли лишитися локально.

