# EVF.MD

## Статус
`DRAFT (для деталізації після DOC-007)`

## Призначення
Experimentation & Validation Framework: правила запуску експериментів і валідації.

## Мінімальний контур
1. Deterministic split/profile.
2. Baseline порівняння: `IG-only` vs `Fusion Graph`.
3. Метрики: Accuracy, F1, OOS Rate, Drift indicators.
4. Репортинг: збереження конфігів, метрик, артефактів.

## Вихідні артефакти
- metric tables
- charts
- run metadata (`dataset_id`, `schema_version`, `κ`, `git_commit`, `experiment_id`)
