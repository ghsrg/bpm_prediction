# План експериментів MVP2.5 для розділу 4 дисертації

Оновлено: 2026-03-23

## 1. Призначення документа

Цей документ фіксує експериментальний план для MVP2.5 у формі, придатній для подальшого опису в дисертації без "дописування з пам'яті". План спирається на поточну реалізацію коду, чинні конфіги, наявні worklog-и та вже зафіксовані результати в `mlruns`.

Ціль MVP2.5 у межах дисертації:
1. довести вплив структурної обізнаності на якість next-activity prediction;
2. показати коректний anti-leakage режим для статистичного збагачення через `strict_asof`;
3. показати поведінку моделі на природній еволюції процесу та на контрольованому structural drift;
4. підготувати доказову базу для переходу до MVP3, але не підміняти MVP2.5 claims незавершеним `Reliability Semaphore`.

## 2. Відповідність цілям дисертації

На підставі `docs/Дисертація Коротенко.docx` для розділу 4 зараз доцільно закривати такі задачі:
1. оцінка впливу `EOPKG` на якість інференсу;
2. аналіз стійкості моделей до структурного дрейфу;
3. перевірка зниження деградації під час еволюції процесу;
4. оцінка метрик не лише точності, а й калібрування та виходу поза допустимий нормативний простір (`OOS`).

Що ще не варто декларувати як повністю доведене в цьому циклі:
1. повноцінний `Reliability Semaphore`;
2. calibrated OOD-threshold policy;
3. завершений human-in-the-loop контур;
4. повністю research-grade snapshot-homogeneous batching.

## 3. Експериментальна гіпотеза MVP2.5

### 3.1 Основна гіпотеза
Додавання нормативної структури процесу до моделі прогнозування підвищує або стабілізує якість прогнозу на еволюціонуючих процесах у порівнянні з logs-only baseline.

### 3.2 Похідні гіпотези
1. `Structure-only` дає стабільний виграш або меншу деградацію відносно baseline на drift-сценаріях.
2. `Structure + statistics` корисне лише за умови коректного activity-to-node alignment і anti-leakage політики `strict_asof`.
3. `Edge-only statistics` можуть бути практично стабільнішими за повний набір node+edge stats.
4. Метрики `ECE` та `OOS` мають змінюватися узгоджено зі складністю drift-сценарію.

## 4. Методологічні інваріанти, які треба явно описати у тексті дисертації

### 4.1 Відтворюваність і seed
У коді `seed` встановлюється через `random`, `numpy`, `torch`, `torch.cuda`, а також `torch.backends.cudnn.deterministic=True` і `benchmark=False`.

Практичний висновок:
1. `seed=42` у поточних конфігах є зафіксованим і має бути зазначений у методиці;
2. це покращує відтворюваність, але не гарантує 100% бітову ідентичність між запусками при `num_workers > 0`, GPU або різному runtime scheduling;
3. для фінальних доказових прогонів бажано:
   - `device=cpu`,
   - `training.dataloader_num_workers=0`,
   - `training.torch_num_threads=1`,
   - `training.torch_num_interop_threads=1`,
   - однаковий snapshot timeline для всіх порівнюваних моделей.

### 4.2 Каскад підготовки даних
Поточний pipeline застосовує дані в такому порядку:
1. хронологічне сортування трас;
2. macro split за `train_ratio`;
3. відсікання по `fraction`;
4. для `train` додатковий micro split `train/val/test` за `split_ratio`.

Важливий наслідок:
1. для `mode=train` модель бачить лише head-частину журналу;
2. для `mode=eval_drift` модель оцінюється лише на хвості після `train_ratio`;
3. drift-вікна не будуються по всьому логу, а лише по tail-сегменту після macro split.

### 4.3 Політика drift-вікон
У `eval_drift`:
1. вікна будуються хронологічно;
2. `size = drift_window_size`;
3. `step = drift_window_sliding`, а якщо `0`, тоді `step = drift_window_size`;
4. короткий хвіст, менший за `drift_window_size`, відкидається.

Це треба прямо писати в методиці, щоб пояснити кількість вимірів на графіку drift.

### 4.4 Як саме формується структура
У поточній архітектурі структура не витягується "на льоту" під час train.

Правильний опис:
1. topology artifacts готуються offline через `ingest-topology` або `sync-topology`;
2. train/eval читають уже збережену структуру через `IKnowledgeGraphPort`;
3. для XES-сценаріїв XES-адаптер лишається trace-ingestion адаптером, але XES може бути джерелом для offline topology build;
4. у Stage 4.2 для XES+stats використовується `mapping.knowledge_graph.backend: neo4j` і `strict_load: true`.

### 4.5 Як саме формується статистика
`sync-stats` зараз:
1. пише contract version `1.0`;
2. обчислює node/edge/global агрегати;
3. додає producer quality telemetry;
4. застосовує alignment gate;
5. застосовує quality gate;
6. зберігає snapshot з `knowledge_version` і `effective as_of`.

Для поточних експериментів треба фіксувати:
1. `stats_time_policy`;
2. `process_scope_policy`;
3. `as_of`;
4. thresholds alignment/quality gates;
5. `knowledge_version` і `effective_as_of_ts`.

### 4.6 Anti-leakage політика
У runtime builder:
1. `latest` бере останній доступний snapshot;
2. `strict_asof` бере snapshot з `snapshot.as_of_ts <= timestamp останньої події префікса`;
3. якщо strict snapshot відсутній, діє `experiment.on_missing_asof_snapshot`.

Для дисертації це критично:
1. основні evidence-runs треба проводити з `strict_asof`;
2. для фінальної серії бажано `on_missing_asof_snapshot=raise`, а не `disable_stats`;
3. якщо використовується `disable_stats`, це треба чесно виносити в обмеження.

### 4.7 Activity-to-node alignment
Для XES/BPMN-зв'язаних експериментів це найбільш чутливе місце.

На BPI2012 вже зафіксовано:
1. при `mapping.xes_adapter.use_classifier=true` активності ставали схожими на `concept:name + lifecycle`, що ламало перетин з `ProcessNode.node_id`;
2. після переходу на `use_classifier=false` stats стали ненульовими і alignment відновився;
3. отже для stats-safe XES експериментів це налаштування має бути явно зафіксоване в методології.

### 4.8 Batch policy для structural payload
Поточне обмеження Stage 4.2:
1. якщо в одному batch змішані різні snapshot versions, runtime лише попереджає;
2. для structural payload береться перший graph у batch;
3. це тимчасова Option-A policy, а не фінальна snapshot-homogeneous batching.

Це треба винести в обмеження дослідження.

## 5. Набори даних для розділу 4

## 5.1 Серія A. Природна еволюція процесу

### A1. BPI Challenge 2012
Статус: готовий як основний відкритий control dataset.

Поточні конфіги:
1. `configs/data/bpi2012_Challenge.yaml` для legacy baseline/structure серії;
2. `configs/data/mvp2_5_stage4_2_xes_neo4j_stat.yaml` для stats-safe Stage 4.2 серії.

Що важливо описати:
1. лог: `BPI_Challenge_2012.xes`;
2. у Stage 4.2 структура читається з `neo4j`, `strict_load=true`;
3. для stats-safe серії `use_classifier=false`;
4. node stats: version + process scope;
5. edge metric: `transition_probability`.

Роль у дослідженні:
1. control dataset для порівняння baseline, structure, edge-stats, full-stats;
2. зручний для абляцій;
3. зручний для перевірки, що статистика не шкодить при коректному alignment.

### A2. BPI Challenge 2017
Статус: потенційно корисний, але не готовий без додаткової ревізії.

Зауваження:
1. `configs/data/bpi2017_Challenge.yaml` містить підозрілий `dataset_label: "bpi2012"`;
2. у legacy конфігу стоїть `use_classifier=true`;
3. наявний `eval_drift` конфіг використовує checkpoint від BPI2012, це не можна виносити у фінальну доказову серію як коректний in-dataset drift protocol.

Рішення:
1. до фінальної серії виправити label;
2. підготувати окремий train checkpoint саме для BPI2017;
3. перевірити, чи потрібен `use_classifier=false` для stats-safe topology alignment.

### A3. BPI Challenge 2020 Travel Permit 1
Статус: рекомендований як третій відкритий датасет для серії природної еволюції.

Переваги:
1. інший домен;
2. інший набір trace/event attributes;
3. краще показує переносимість методики, ніж повторне використання лише BPI2012.

Ризики:
1. у наявному конфігу також `use_classifier=true`;
2. потрібно окремо перевірити topology alignment для offline structure/stats path.

### A4. Корпоративні багатоверсійні логи
Статус: найсильніший матеріал для дисертації, але не для відкритої статті без анонімізації.

Роль:
1. показати довготривалу еволюцію процесів;
2. показати drift по 50+ версіях;
3. побудувати анонімізовані графіки `F1`, `ECE`, `OOS`, drift intensity.

Вимога:
1. у тексті дисертації показувати лише агреговані та анонімізовані дані;
2. фіксувати `knowledge_version` і `effective_as_of` у кожній серії.

## 5.2 Серія B. Контрольований різкий drift

### B1. Business Process Drift / Loan Assessment
Статус: підходить не як "довга історія", а як stress-test на abrupt drift.

Наявні конфіги:
1. `configs/data/2015_cb_DataDrift.yaml`
2. `configs/data/2015_cd_DataDrift.yaml`

Поточні властивості:
1. `use_classifier=false`;
2. `pairing_strategy=lifo`;
3. `complete_transitions=["complete"]`;
4. стандартний logs-only feature set без structural stats.

Правильне трактування в розділі 4:
1. кожен drift-type є окремим сценарієм;
2. вісь X краще робити як `window index` або `pre-drift / onset / post-drift`, а не календарний час;
3. якщо onset відомий, train краще обмежити pre-drift сегментом, а не просто першими 50%.

## 6. Варіанти моделей для порівняння

Для розділу 4 рекомендовано не змішувати всі можливі варіації в одну таблицю.

### 6.1 Обов'язкова серія
1. `Baseline`
2. `Structure`
3. `Structure + edge statistics`
4. `Structure + full statistics`

### 6.2 Що брати як головний результат
1. baseline є нульовою лінією;
2. `Structure` показує чистий вклад topology;
3. `Structure + edge statistics` зараз виглядає найбільш перспективним production-like варіантом;
4. `full statistics` треба залишати в основній серії, але інтерпретувати обережно: це одночасно кандидат на покращення і перевірка меж застосовності.

## 7. Рекомендована матриця експериментів

## 7.1 Серія A. Природна еволюція

### A/B matrix
Для кожного dataset:
1. train на перших `50%` трас;
2. drift evaluation на других `50%`;
3. однаковий `seed`, однакові split settings, однаковий checkpoint protocol;
4. однаковий topology/stats timeline для всіх model variants.

### Рекомендований список
1. `BPI2012`: Baseline / Structure / EdgeStats / FullStats
2. `BPI2020 Permit 1`: Baseline / Structure / EdgeStats / FullStats
3. `BPI2017`: лише після виправлення конфіга і окремого train checkpoint
4. корпоративний датасет: Baseline / Structure / EdgeStats / FullStats

## 7.2 Серія B. Abrupt drift

Для кожного drift scenario:
1. train на pre-drift сегменті або максимально наближеному до нього head-сегменті;
2. drift windows по onset/post-onset зоні;
3. порівнювати не календарну динаміку, а профіль падіння якості відносно onset.

Рекомендовано:
1. `cb`
2. `cd`
3. за можливості ще 1-2 типи drift із того ж набору

## 7.3 Multi-seed policy

Поточні локальні конфіги зазвичай мають `seed=42`.

Для фінального розділу 4 рекомендовано:
1. мінімум 3 seeds: `42, 43, 44`;
2. бажано 5 seeds: `42, 43, 44, 45, 46`;
3. виносити в таблиці середнє та стандартне відхилення.

## 8. Базові конфіги та налаштування, які вже можна фіксувати в тексті

## 8.1 Legacy BPI2012 structure experiment
Файл: `configs/experiments/02_train_bpi2012.yaml`

Ключові параметри:
1. `mode=train`
2. `fraction=0.1`
3. `split_strategy=temporal`
4. `train_ratio=0.5`
5. `split_ratio=[0.7,0.2,0.1]`
6. `training.retrain=true`
7. `seed=42`

Модель:
1. `EOPKGGATv2`
2. `hidden_dim=64`
3. `struct_encoder_type=GATv2Conv`
4. `cross_attention_heads=4`
5. `dropout=0.2`
6. `batch_size=128`
7. `epochs=30`
8. `learning_rate=0.001`
9. `patience=10`
10. `delta=1e-4`
11. `device=cpu`

## 8.2 Stage 4.2 BPI2012 baseline with stats-ready data path
Файл: `configs/experiments/mvp2_5_stage4_2_baseline_xes_stat.yaml`

Ключові параметри:
1. `mode=train`
2. `fraction=0.1`
3. `split_strategy=temporal`
4. `train_ratio=0.7`
5. `split_ratio=[0.7,0.2,0.1]`
6. `stats_time_policy=strict_asof`
7. `training.retrain=true`
8. `seed=42`

## 8.3 Stage 4.2 BPI2012 EOPKG with stats-ready data path
Файл: `configs/experiments/mvp2_5_stage4_2_eopkg_xes_stat.yaml`

Ключові параметри:
1. ті самі split settings, що і в baseline Stage 4.2;
2. `stats_time_policy=strict_asof`;
3. структура + offline stats через `neo4j`.

## 8.4 Data config для Stage 4.2 stats-safe BPI2012
Файл: `configs/data/mvp2_5_stage4_2_xes_neo4j_stat.yaml`

Критичні параметри:
1. `mapping.adapter=xes`
2. `mapping.knowledge_graph.backend=neo4j`
3. `mapping.knowledge_graph.strict_load=true`
4. `mapping.knowledge_graph.ingest_split=train`
5. `mapping.xes_adapter.use_classifier=false`
6. `pairing_strategy=lifo`
7. `stats_quality_gate.enabled=true`
8. `stats_quality_gate.on_fail=allow_with_warning`

Stats mapping:
1. node stats по `version` і `process` scope;
2. edge weight = `transition_probability`;
3. encoding для stats зараз `identity`.

## 8.5 DataDrift baseline train
Файл: `configs/experiments/01_train_2015_DataDrift.yaml`

Ключові параметри:
1. `mode=train`
2. `train_ratio=0.5`
3. `split_strategy=temporal`
4. `fraction=1`
5. `training.retrain=false`
6. `seed=42`

## 8.6 DataDrift eval_drift
Файл: `configs/experiments/01_eval_drift_cdDataDrift.yaml`

Ключові параметри:
1. `mode=eval_drift`
2. `train_ratio=0.01`
3. `drift_window_size=500`
4. `drift_window_sliding=50`
5. `load_checkpoint=...best.pth`
6. `seed=42`

Примітка:
цей конкретний eval конфіг не є ідеальним шаблоном для фінального доказового abrupt-drift протоколу, бо його потрібно звіряти з реальною drift boundary.

## 9. Метрики, які треба обов'язково виносити в розділ 4

### 9.1 Основні
1. `val_macro_f1`
2. `test_macro_f1`
3. `test_accuracy`
4. `test_top3_accuracy`

### 9.2 Надійність та допустимість
1. `test_ece`
2. `test_oos`

### 9.3 Зрізи
1. метрики за довжиною префікса;
2. метрики за версією процесу;
3. drift-window метрики: `window_macro_f1`, `window_test_ece`, `window_start_ts`, `window_end_ts`.

### 9.4 Аудит stats path
1. `alignment_event_match_ratio`
2. `alignment_unique_activity_coverage`
3. `alignment_node_coverage`
4. `history_coverage_percent`
5. `non_zero_ratio_overall`
6. `knowledge_version`
7. `effective_as_of_ts`

## 10. Які діаграми показувати в розділі 4

## 10.1 Для природної еволюції
1. лінійний графік `macro-F1` по drift windows;
2. лінійний графік `ECE` по drift windows;
3. лінійний графік `OOS` по drift windows;
4. heatmap `dataset × model_variant` по `macro-F1`;
5. bar chart середнього `macro-F1` з error bars по seeds;
6. окремий audit-plot для `history_coverage_percent` і `non_zero_ratio_overall`.

## 10.2 Для abrupt drift
1. `macro-F1` по normalized windows;
2. вертикальна позначка drift onset;
3. `ECE` та `OOS` на тій самій фазовій осі;
4. boxplot або forest-plot приросту відносно baseline;
5. таблиця `pre-drift mean`, `post-drift mean`, `delta_drop`.

## 10.3 Для корпоративних даних
1. anonymized version timeline;
2. F1/ECE/OOS по версіях;
3. drift heatmap по версіях/періодах;
4. за потреби окремо trajectory `quality of stats snapshots`.

## 11. Попередні верифіковані результати, які вже можна використати як чернетку

Нижче наведені лише ті результати, які вдалося підтвердити локально через `mlruns`.

### 11.1 Legacy baseline на BPI2012
Run: `MVP1_Baseline_train_05_bpi2012_GATv2`

Зафіксовано:
1. `best val_macro_f1 = 0.55798`
2. `test_macro_f1 = 0.53510`
3. `test_oos = 1.0`

Інтерпретація:
1. це сильний контрольний baseline для ранньої серії;
2. але його не варто напряму змішувати з Stage 4.2 stats-safe runs без пояснення відмінностей у data path.

### 11.2 Stage 4.2 baseline на BPI2012 у stats-ready контурі
Run: `Baseline_XES_bpi2012`

Зафіксовано:
1. `best val_macro_f1 = 0.37306`
2. `test_macro_f1 = 0.36361`
3. `test_oos = 0.13514`

### 11.3 Stage 4.2 EOPKG + full stats на BPI2012 у stats-ready контурі
Run: `MVP2_5_Stage4_2_EOPKG_XES_Stat_bpi2012_stage4_2_xes_stat_GATv2`

Зафіксовано:
1. `best val_macro_f1 = 0.35261`
2. `test_macro_f1 = 0.33687`
3. `test_oos = 0.13436`

Інтерпретація:
1. у поточному локально збереженому run full-stats не зруйнував модель повністю;
2. однак покращення над baseline не показав;
3. отже повний stats bundle поки що не можна вважати безумовно корисним.

### 11.4 Exploratory edge-only run
Run: `EOPKG_XES_Stat_Edge_bpi2012`

Зафіксовано:
1. `best val_macro_f1 = 0.54484`
2. `test_macro_f1 = 0.56098`
3. `test_oos = 0.02839`

Інтерпретація:
1. edge-only stats зараз є найперспективнішим експериментальним напрямком;
2. його обов'язково треба включити в фінальну матрицю абляцій;
3. але фінальний висновок треба робити лише після multi-seed повторення.

## 12. Обмеження дослідження, які треба чесно винести в текст

1. `snapshot-homogeneous batching` ще не реалізований; можливі змішані snapshot batches з Option-A forward policy.
2. `on_missing_asof_snapshot` у частині конфігів може відключати stats замість fail-fast; це треба явно звітувати.
3. `use_classifier` для XES є dataset-sensitive; універсальне значення не можна декларувати без preflight alignment.
4. BPI2017 ще не готовий до фінальної серії без виправлення конфіга і окремого checkpoint.
5. BPI2018 у поточному репозиторії не має готового data config.
6. Повний пакет node+edge stats може містити надлишковий шум і вимагає подальшого відбору ознак.
7. Корпоративні дані закриті, тому зовнішня відтворюваність цієї частини експерименту обмежена.

## 13. Що ще доробити перед масовим прогоном

### 13.1 Обов'язково
1. зафіксувати final research configs для кожного dataset і variant;
2. перевести strict evidence-runs на `on_missing_asof_snapshot=raise`;
3. зробити окремий canonical export результатів у `csv/json`;
4. прогнати multi-seed серію;
5. для кожного dataset зберігати `sync-topology` summary і `sync-stats` summary.

### 13.2 Дуже бажано
1. підготувати dataset registry для drift scenario;
2. окремо зафіксувати onset для abrupt-drift датасетів;
3. виправити `bpi2017_Challenge.yaml`;
4. сформувати один узгоджений plotting notebook/script.

## 14. Тезовий шаблон для опису кожного експерименту в дисертації

## 14.1 Опис набору даних
1. Назва датасету.
2. Джерело.
3. Тип drift: відсутній / природна еволюція / різкий structural drift.
4. Кількість трас, подій, активностей.
5. Особливості атрибутів.
6. Чи є topology artifact, яким способом він побудований.

## 14.2 Опис експерименту
1. Мета порівняння.
2. Варіанти моделі.
3. Конфігурація split.
4. `seed`, `device`, `workers`.
5. `stats_time_policy`, `on_missing_asof_snapshot`.
6. Налаштування XES mapping (`use_classifier`, `pairing_strategy`).
7. Налаштування quality/alignment gates.

## 14.3 Обмеження
1. alignment assumption;
2. snapshot coverage;
3. batch policy;
4. dataset-specific configuration sensitivity;
5. закритість або неповна публічність даних.

## 14.4 Результати
1. Основна метрика `macro-F1`.
2. Додатково `ECE` та `OOS`.
3. Поведінка на drift-вікнах.
4. Порівняння з baseline.
5. Висновок: чи підтверджує результат гіпотезу.

## 15. Рекомендований висновок для поточного етапу

На поточному етапі MVP2.5 уже придатний для сильної експериментальної серії розділу 4, якщо сфокусуватися на:
1. `Baseline vs Structure vs EdgeStats vs FullStats`;
2. `strict_asof` як обов'язковій anti-leakage політиці;
3. explicit reporting `alignment + quality + snapshot provenance`.

Для фінальної дисертаційної аргументації найсильніша схема:
1. відкриті датасети для відтворюваності;
2. корпоративні датасети для практичної цінності;
3. окремо natural evolution і abrupt drift;
4. без передчасних claims про завершений MVP3 semaphore.
