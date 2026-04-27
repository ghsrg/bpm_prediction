# Runbook експериментальної матриці MVP2.5

Оновлено: 2026-03-23

## 1. Призначення

Це операційний runbook для серії експериментів під:
1. розділ 4 дисертації;
2. можливу Scopus-статтю;
3. узгоджене збереження артефактів і результатів.

Документ спирається на:
1. [experiment_plan_mvp2_5.MD](C:/Users/korsr/PycharmProjects/bpm_prediction/experiment_plan_mvp2_5.MD)
2. [experiment_plan_scopus.MD](C:/Users/korsr/PycharmProjects/bpm_prediction/experiment_plan_scopus.MD)

## 2. Базові правила серії

Для фінальної evidence-серії не змішувати швидкі exploratory прогони з підсумковими.

Фіксуємо такі правила:
1. `device=cpu`
2. `training.dataloader_num_workers=0`
3. `training.torch_num_threads=1`
4. `training.torch_num_interop_threads=1`
5. `split_strategy=temporal`
6. для main natural-evolution серії `train_ratio=0.5`
7. для main abrupt-drift серії train має бути до drift onset або максимально наближений до pre-drift
8. для stats-вариантів `stats_time_policy=strict_asof`
9. для фінальних stats-вариантів `on_missing_asof_snapshot=raise`
10. для XES datasets обов'язковий preflight на `use_classifier`

## 3. Naming convention

Щоб не втратити контроль над артефактами, дотримуватись єдиного неймінгу.

### 3.1 Конфіги
Формат:

`configs/experiments/final_<dataset>_<variant>_s<seed>.yaml`

Приклади:
1. `final_bpi2012_baseline_s42.yaml`
2. `final_bpi2012_structure_s42.yaml`
3. `final_bpi2012_edge_stats_s42.yaml`
4. `final_bpi2012_full_stats_s42.yaml`

### 3.2 Outputs
Формат:

`outputs/final/<dataset>/<variant>/seed_<seed>/...`

### 3.3 MLflow
У `experiment.name` тримати таку ж схему:

`FINAL_<DATASET>_<VARIANT>_S<SEED>`

## 4. Що вважається фінальним evidence run

Run вважається фінальним лише якщо одночасно виконані умови:
1. збережений YAML конфіг;
2. збережений `sync-topology` summary або `ingest-topology` summary;
3. збережений `sync-stats` summary або `sync-stats-backfill` summary;
4. збережені основні MLflow metrics;
5. зафіксовані `knowledge_version` і `effective_as_of_ts`;
6. зафіксовані `alignment_*` і `quality_*` поля;
7. у звіті явно вказано `seed`, `use_classifier`, `strict_asof`, `on_missing_asof_snapshot`.

## 5. Seeds для серії

### 5.1 Мінімум для дисертації
1. `42`
2. `43`
3. `44`

### 5.2 Бажано
1. `42`
2. `43`
3. `44`
4. `45`
5. `46`

## 6. Варіанти моделей

### 6.1 Основний набір
1. `baseline`
2. `structure`
3. `edge_stats`
4. `full_stats`

### 6.2 Тлумачення
1. `baseline` = logs-only baseline GATv2
2. `structure` = EOPKG без stats
3. `edge_stats` = EOPKG + лише edge metric
4. `full_stats` = EOPKG + node + edge stats

## 7. Матриця datasets

| Dataset | Drift type | Статус | Що вже є | Що треба доробити |
|---|---|---|---|---|
| BPI2012 | natural evolution / control | ready | legacy train/eval configs, Stage 3.4/4.2 XES stats path | створити final 50/50 configs |
| BPI2020 Permit 1 | natural evolution | partial | data config є | створити experiment configs, перевірити alignment |
| BPI2017 | natural evolution | blocked | data config є | виправити `dataset_label`, зробити окремі train/eval configs, окремий checkpoint |
| DataDrift cb | abrupt drift | partial | train/eval configs є | уточнити train boundary відносно drift onset |
| DataDrift cd | abrupt drift | partial | data config є, eval config pattern є | зробити окремий train/eval pair |
| Corporate logs | natural evolution, multi-version | partial | camunda files/sql path є | фіналізувати namespace, sync-backfill, анонімізований export |

## 8. Runbook по серіях

## 8.1 Серія A. BPI2012

### A1. Preflight alignment
Мета: зафіксувати коректний XES mapping для topology + stats.

Застосовувати:
1. `use_classifier=false`
2. `pairing_strategy=lifo`
3. `strict_load=true`

База:
1. `configs/data/mvp2_5_stage4_2_xes_neo4j_stat.yaml`
2. `configs/experiments/mvp2_5_stage4_2_eopkg_xes_stat.yaml`

Команди:
```powershell
python main.py ingest-topology --config configs/experiments/mvp2_5_stage4_2_eopkg_xes_stat.yaml --out outputs/final/bpi2012/topology_ingest.json
python main.py sync-stats --config configs/experiments/mvp2_5_stage4_2_eopkg_xes_stat.yaml --out outputs/final/bpi2012/sync_stats_probe.json
```

Acceptance:
1. `alignment_is_ok=true`
2. `alignment_event_match_ratio >= 0.9`
3. `alignment_unique_activity_coverage >= 0.9`
4. `non_zero_ratio_overall` не близький до нуля

### A2. Backfill stats timeline
Мета: побудувати history snapshots для `strict_asof`.

Команда:
```powershell
python main.py sync-stats-backfill --config configs/experiments/mvp2_5_stage4_2_eopkg_xes_stat.yaml --step weekly --out-dir outputs/final/bpi2012/backfill --summary-out outputs/final/bpi2012/backfill_summary.json
```

Примітка:
1. для підсумкової серії краще weekly;
2. якщо timeline надто грубий, перейти на daily лише для короткого періоду;
3. monthly для BPI2012 навряд чи дасть достатню granularність.

### A3. Final configs, які треба зробити
Створити 12 або 20 конфігів, залежно від кількості seeds.

Базові шаблони:
1. `01_train_bpi2012.yaml` для `baseline`
2. `02_train_bpi2012.yaml` для `structure`
3. `mvp2_5_stage4_2_eopkg_xes_stat.yaml` для `full_stats`
4. edge-only конфіг взяти з локального exploratory run і винести як окремий `final_bpi2012_edge_stats_sXX.yaml`

Обов'язково змінити у final configs:
1. `fraction=1`
2. `train_ratio=0.5`
3. `split_ratio=[0.7,0.2,0.1]`
4. `device=cpu`
5. `dataloader_num_workers=0`
6. `stats_time_policy=strict_asof`
7. `on_missing_asof_snapshot=raise`
8. `seed=<42|43|44|45|46>`

### A4. Train matrix

| Variant | Base config | Action |
|---|---|---|
| baseline | `configs/experiments/01_train_bpi2012.yaml` | клонувати в final 50/50 full-fraction |
| structure | `configs/experiments/02_train_bpi2012.yaml` | клонувати в final 50/50 full-fraction |
| edge_stats | exploratory edge YAML з UI artifact | винести в окремий final config |
| full_stats | `configs/experiments/mvp2_5_stage4_2_eopkg_xes_stat.yaml` | клонувати в final 50/50 full-fraction |

Команда train:
```powershell
python main.py --config configs/experiments/final_bpi2012_<variant>_s42.yaml
```

Повторити для кожного `seed`.

### A5. Drift evaluation matrix
Після train для кожного variant створити окремий `eval_drift` config:
1. `mode=eval_drift`
2. `load_checkpoint=<best checkpoint from matching train run>`
3. `train_ratio=0.5`
4. `fraction=1`
5. `drift_window_size=1000`
6. `drift_window_sliding=200`

Чому так:
1. це вже близько до наявного `02_eval_drift_bpi2012.yaml`;
2. дає прийнятну кількість вікон на великому логу;
3. легко порівнювати baseline і structure-варианти.

Команда:
```powershell
python main.py --config configs/experiments/final_bpi2012_<variant>_eval_drift_s42.yaml
```

### A6. Що зберігати
1. MLflow run id
2. best checkpoint path
3. drift metrics export
4. `sync_stats_summary.json`
5. `backfill_summary.json`

## 8.2 Серія B. BPI2020 Permit 1

### B1. Поточний статус
Є лише data config:
1. `configs/data/bpi2020_Travel_Permit_1.yaml`

Немає готових experiment configs під final 50/50.

### B2. Що зробити перед запуском
1. створити baseline train config як копію `01_train_bpi2012.yaml` з include на `bpi2020_Travel_Permit_1.yaml`
2. створити structure train config як копію `02_train_bpi2012.yaml`
3. створити stats-safe data config за зразком `mvp2_5_stage4_2_xes_neo4j_stat.yaml`, але для BPI2020
4. перевірити `use_classifier=true` проти `false`
5. виконати preflight topology alignment

### B3. Рекомендована послідовність
1. ingest-topology
2. sync-stats probe
3. sync-stats-backfill weekly
4. baseline train
5. structure train
6. edge_stats train
7. full_stats train
8. eval_drift для всіх варіантів

### B4. Важливе застереження
Не починати multi-seed серію, поки не підтверджено:
1. alignment coverage;
2. ненульовість stats;
3. відсутність pathological NaN/zero-dominant behavior.

## 8.3 Серія C. BPI2017

### C1. Поточний статус
Є data config:
1. `configs/data/bpi2017_Challenge.yaml`

Але є блокер:
1. `dataset_label` зараз виглядає як `bpi2012`

### C2. До запуску серії обов'язково
1. виправити `dataset_label`
2. зробити окремий baseline train config
3. зробити окремий structure train config
4. не використовувати checkpoint від BPI2012 у фінальній серії
5. окремо перевірити `use_classifier`

### C3. Порядок
Такий самий, як для BPI2020.

## 8.4 Серія D. DataDrift cb / cd

### D1. Поточні конфіги
Вже є:
1. `configs/experiments/01_train_2015_DataDrift.yaml`
2. `configs/experiments/01_eval_drift_cdDataDrift.yaml`
3. `configs/data/2015_cb_DataDrift.yaml`
4. `configs/data/2015_cd_DataDrift.yaml`

### D2. Що треба змінити для фінальної серії
1. зробити окремий train config для `cb`
2. зробити окремий train config для `cd`
3. зробити окремі matching eval configs
4. зафіксувати, де саме починається drift phase

### D3. Рекомендований протокол
1. train на pre-drift або head segment
2. eval_drift на onset/post-onset segment
3. `drift_window_size=500`
4. `drift_window_sliding=50`

### D4. Варіанти моделі
Для Scopus-серії достатньо:
1. baseline
2. structure
3. edge_stats

`full_stats` можна лишити як optional ablation, якщо статистичний path для цього датасету буде коректно узгоджено.

## 8.5 Серія E. Corporate logs

### E1. Поточна база
Для Camunda path є готові Stage 4.2 configs:
1. `configs/experiments/mvp2_5_stage4_2_baseline_files_stat.yaml`
2. `configs/experiments/mvp2_5_stage4_2_eopkg_files.yaml`
3. `configs/experiments/mvp2_5_stage4_2_eopkg_files_stat.yaml`

### E2. Рекомендована послідовність
1. `sync-topology`
2. `sync-stats-backfill --step weekly`
3. baseline train
4. structure train
5. edge_stats train
6. full_stats train
7. eval_drift по tail windows

### E3. Для тексту дисертації
Зберігати лише:
1. anonymized plots
2. агреговані таблиці
3. без raw event payload

## 9. Порядок виконання серій

Щоб не розпорошуватись, виконувати саме в такому порядку.

### Phase 1. BPI2012
1. final baseline
2. final structure
3. final edge_stats
4. final full_stats
5. multi-seed repeat
6. plotting

### Phase 2. DataDrift
1. `cb`
2. `cd`
3. plotting abrupt-drift series

### Phase 3. BPI2020
1. preflight
2. train/eval
3. multi-seed repeat

### Phase 4. Corporate logs
1. weekly backfill
2. final train/eval
3. anonymized plots

### Phase 5. BPI2017
1. лише після виправлення конфіга і якщо ще потрібен для розширення вибірки

## 10. Чекліст перед кожним train run

1. Чи правильний dataset?
2. Чи правильний variant?
3. Чи правильний `seed`?
4. Чи `fraction=1` для фінального run?
5. Чи `train_ratio=0.5`, якщо це natural-evolution run?
6. Чи `stats_time_policy=strict_asof`, якщо це stats run?
7. Чи `on_missing_asof_snapshot=raise`, якщо це фінальний stats run?
8. Чи збігається `use_classifier` з preflight висновком?
9. Чи існує topology artifact?
10. Чи існує stats timeline/backfill?

## 11. Чекліст після кожного run

1. Зберегти MLflow run id.
2. Зберегти checkpoint path.
3. Зберегти команду запуску.
4. Зберегти YAML конфіг як artifact.
5. Занести result row у зведену таблицю.
6. Якщо це drift run, експортувати window metrics.

## 12. Мінімальна зведена таблиця результатів

Завести одну master table, де кожен рядок = один run.

Колонки:
1. `dataset`
2. `variant`
3. `seed`
4. `mode`
5. `config_path`
6. `run_id`
7. `best_epoch`
8. `best_val_macro_f1`
9. `test_macro_f1`
10. `test_ece`
11. `test_oos`
12. `knowledge_version`
13. `effective_as_of_ts`
14. `alignment_event_match_ratio`
15. `alignment_unique_activity_coverage`
16. `alignment_node_coverage`
17. `notes`

## 13. Що не робити

1. Не змішувати `fraction=0.1` exploratory runs з фінальною evidence table.
2. Не порівнювати BPI2012 stats-safe runs із legacy runs без примітки про різний data path.
3. Не запускати full-stats multi-seed серію, якщо preflight alignment не підтверджено.
4. Не використовувати чужий checkpoint для іншого dataset у фінальній серії.
5. Не описувати `latest` як anti-leakage policy.

## 14. Перший практичний спринт, який варто зробити зараз

### Sprint A. Закрити BPI2012 повністю
1. створити final 50/50 configs для `baseline`, `structure`, `edge_stats`, `full_stats`
2. прогнати `seed=42`
3. перевірити метрики
4. якщо все стабільно, прогнати `43, 44`
5. зібрати перший plotting pack

### Sprint B. Після цього
1. перенести той самий протокол на `BPI2020`
2. окремо зробити abrupt-drift серію `cb/cd`

## 15. Рекомендація

Якщо мета зараз швидко зібрати сильну доказову базу, найкращий порядок такий:
1. BPI2012 `baseline vs structure vs edge_stats vs full_stats`
2. DataDrift `cb/cd` для stress-test
3. BPI2020 для external validity
4. corporate logs для практичного підтвердження

Саме в такому порядку найменше ризику втратити час на слабко підготовлені датасети.
