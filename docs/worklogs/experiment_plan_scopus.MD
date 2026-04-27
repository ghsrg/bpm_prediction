# План робіт для Scopus-статті

Оновлено: 2026-03-23

## 1. Призначення статті

Ця стаття має бути зосереджена виключно на порівнянні:
1. `logs-only baseline`;
2. `structure-aware model`;
3. `structure + drift-safe statistics`.

Предмет статті:
1. вплив нормативної структури на next-activity prediction;
2. вплив historical statistics за умови anti-leakage retrieval;
3. поведінка моделей під process drift;
4. межі корисності статистичного збагачення.

Що не входить у цю статтю:
1. `Reliability Semaphore`;
2. OOD policy як окремий внесок;
3. human-in-the-loop;
4. повна наскрізна архітектура дисертації.

Це треба вважати жорстким scope lock.

## 2. Позиціонування відносно вже опублікованих статей

У вас уже є дві статті:

1. DOI: `https://doi.org/10.32782/IT/2025-2-5`  
   Назва: `ПОРІВНЯННЯ GNN-АРХІТЕКТУР ДЛЯ ПРОГНОЗУ АКТИВНОСТІ У БІЗНЕС-ПРОЦЕСАХ: LOGS-ONLY ТА BPMN-ПІДХОДИ`

2. DOI: `https://doi.org/10.32782/tnv-tech.2025.5.1.40`  
   Назва: `PREDICTING ACTIVITIES IN BUSINESS PROCESSES WITH VARYING PREFIX LENGTHS: AN EVALUATION OF GNN IN LOGS-ONLY AND BPMN MODES`

## 2.1 Що вже було закрито цими статтями

Перша стаття вже покриває:
1. порівняння GNN-архітектур;
2. базове зіставлення `logs-only` і `BPMN` режимів;
3. загальну корисність структурної інформації як такої.

Друга стаття вже покриває:
1. вплив довжини префікса;
2. аналіз поведінки `logs-only` та `BPMN` режимів на різних prefix length;
3. оцінку моделей у статичному протоколі без центрального акценту на drift-safe temporal stats.

## 2.2 Що має бути новим у цій статті

Щоб не було самоплагіату, нова стаття має змістовно відрізнятися за research question, protocol і claims.

Новий внесок цієї статті:
1. не просто `logs-only vs BPMN`, а `structure vs structure + statistics` під drift;
2. не просто prefix-length analysis, а `drift-aware evaluation`;
3. не просто наявність структури, а `strict_asof`-safe використання historical statistics;
4. не просто якість прогнозу, а поведінка в умовах природної еволюції та різкого drift;
5. окрема роль `activity-to-topology alignment` як методичного фактору;
6. абляція `Structure` проти `Structure + edge statistics` проти `Structure + full statistics`.

## 2.3 Що не можна переносити в нову статтю як центральний матеріал

Не можна повторно подавати як новий результат:
1. загальний факт, що BPMN/structure може допомагати відносно logs-only;
2. таблиці й графіки, де головний сюжет лише `logs-only vs BPMN`;
3. основний narrative про varying prefix lengths;
4. формулювання висновків попередніх статей майже дослівно.

Допустимо використовувати лише як передумову:
1. короткий огляд уже здобутих результатів;
2. 1-2 посилання на власні попередні статті в related work;
3. короткий місток: "попередні роботи показали користь structure-aware підходу в статичному режимі, тоді як поточна стаття досліджує drift-safe enrichment".

## 3. Наукова ідея нової статті

### 3.1 Робоча назва

`Structure-Aware Next-Activity Prediction under Process Drift: Ablation of Topology and Historical Statistics with Strict As-Of Retrieval`

### 3.2 Українська робоча формула

`Вплив нормативної структури та historical statistics на прогнозування активностей бізнес-процесів в умовах drift`

### 3.3 Головна теза

Нормативна структура бізнес-процесу є корисним джерелом інформації для next-activity prediction, але статистичне збагачення стає науково коректним і практично корисним лише тоді, коли:
1. немає leakage з майбутнього;
2. є коректний activity-to-topology alignment;
3. statistics подаються у правильній формі, де edge-level сигнали можуть бути кориснішими за важкий full-stats bundle.

## 4. Research questions

Стаття має відповідати на 3 чіткі питання.

### RQ1
Чи покращує нормативна структура якість next-activity prediction відносно `logs-only baseline` на датасетах з process evolution?

### RQ2
Чи дає `Structure + edge statistics` стабільніший або кращий результат, ніж `Structure-only` і `Structure + full statistics`?

### RQ3
Як змінюється поведінка моделей на drift windows у двох режимах:
1. natural evolution;
2. abrupt drift?

## 5. Межі claims статті

У статті можна стверджувати:
1. structure-aware підхід порівнюється з logs-only baseline в drift-aware protocol;
2. `strict_asof` є необхідною anti-leakage умовою для historical stats;
3. edge-level statistics можуть бути більш практичними, ніж full-stats enrichment;
4. quality of enrichment залежить від alignment.

У статті не можна стверджувати:
1. що задача OOD detection уже вирішена;
2. що `Reliability Semaphore` уже валідований;
3. що модель вже підтримує повноцінну lifelong adaptation без окремого доказу;
4. що поточний batching policy є остаточним industrial-grade рішенням.

## 6. Новизна статті

Новизну треба формулювати не як "ще одне порівняння BPMN і logs-only", а так:

1. Запропоновано drift-aware протокол оцінювання structure-aware next-activity prediction.
2. Запропоновано та валідується anti-leakage режим `strict_asof` для подання historical statistics.
3. Досліджено вплив різних форм структурного збагачення:
   - topology only;
   - edge statistics;
   - full statistics.
4. Виявлено, що ефективність structural enrichment залежить від alignment policy для XES/BPMN mapping.
5. Показано, що не будь-яке статистичне збагачення є корисним, а його форма і temporal protocol є критично важливими.

Саме це і є відмежуванням від двох попередніх статей.

## 7. Набори даних для статті

## 7.1 Основний набір

### D1. BPI Challenge 2012
Роль:
1. control dataset;
2. основний reproduction baseline;
3. перший датасет для повної абляції.

Чому він потрібен:
1. уже є підтверджені локальні результати;
2. уже досліджувався у попередніх роботах, тому дозволяє показати еволюцію дослідження;
3. підходить для чесного порівняння старого статичного режиму з новим drift-aware protocol.

### D2. BPI Challenge 2020 Travel Permit 1
Роль:
1. external validity;
2. перевірка, що висновок не прив'язаний лише до BPI2012;
3. інший домен, інші атрибути процесу.

### D3. Business Process Drift
Рекомендовано використати:
1. `cb`
2. `cd`

Роль:
1. stress-test на abrupt drift;
2. окрема серія для перевірки падіння якості навколо drift onset.

## 7.2 Умовно додатковий набір

### D4. BPI2017
Включати лише після:
1. виправлення `dataset_label`;
2. окремого train checkpoint;
3. перевірки alignment.

### D5. Корпоративні дані
Не робити центральною частиною статті.

Можна використати лише як:
1. короткий external validity note;
2. додаток або окремий subsection без сирих даних.

## 8. Варіанти моделей у статті

## 8.1 Основні варіанти

1. `Baseline`
2. `Structure`
3. `Structure + edge statistics`

## 8.2 Додатковий ablation варіант

4. `Structure + full statistics`

### Практична порада
Основний текст статті будувати на перших трьох варіантах.

`full statistics`:
1. залишити в ablation section;
2. інтерпретувати як перевірку меж застосовності;
3. не робити центральним позитивним result, якщо він нестабільний.

## 9. Експериментальний протокол статті

## 9.1 Reproducibility settings

Для фінальної серії:
1. `seed`: мінімум `42, 43, 44`
2. бажано `42, 43, 44, 45, 46`
3. `device=cpu`
4. `dataloader_num_workers=0`
5. `torch_num_threads=1`
6. `torch_num_interop_threads=1`

## 9.2 Natural evolution protocol

Для `BPI2012` і `BPI2020`:
1. train на перших `50%` трас;
2. drift-tail evaluation на других `50%`;
3. однакові split settings для всіх варіантів;
4. однаковий seed list;
5. однаковий snapshot timeline.

## 9.3 Abrupt drift protocol

Для `cb` і `cd`:
1. train на pre-drift частині або максимально близькому head segment;
2. eval по windows навколо onset/post-onset;
3. X-вісь: `window index` або `relative phase`;
4. не трактувати ці датасети як довгу природну часову еволюцію.

## 9.4 Drift windows

Базовий робочий протокол:
1. `drift_window_size=500` для Business Process Drift;
2. `drift_window_sliding=50`;
3. для BPI2012 можна починати з `1000/200`;
4. фінальні значення треба зафіксувати до старту multi-seed серії і не змінювати між моделями.

## 10. Методичні умови, які обов'язково треба описати у статті

## 10.1 Побудова структури

Опис має бути таким:
1. topology artifacts будуються offline через `ingest-topology` або `sync-topology`;
2. train/eval не витягують структуру з логу під час інференсу;
3. структура читається з knowledge backend.

## 10.2 Подання historical statistics

Опис має бути таким:
1. statistics синхронізуються окремо через `sync-stats`;
2. snapshots мають `knowledge_version` і `as_of_ts`;
3. у runtime для scientific runs використовується `strict_asof`;
4. lookup виконується за timestamp останньої події префікса.

## 10.3 Alignment

Окремо треба написати:
1. XES classifier може змінювати `activity_id`;
2. для BPI2012 stats-safe path критичним було `use_classifier=false`;
3. перед статистичними експериментами проводився alignment preflight.

## 10.4 Обмеження batching

У статті треба чесно вказати:
1. поточний runtime лише попереджає про mixed snapshot batches;
2. structural payload у такому випадку береться з першого graph у batch;
3. це обмеження реалізації, але воно однакове для всіх порівнюваних structure/stat runs.

## 11. Метрики статті

## 11.1 Основні

1. `macro-F1`
2. `ECE`
3. `OOS`

## 11.2 Додаткові

1. `top-3 accuracy`
2. prefix-length slices
3. by-version slices

## 11.3 Audit metrics

1. `alignment_event_match_ratio`
2. `alignment_unique_activity_coverage`
3. `alignment_node_coverage`
4. `history_coverage_percent`
5. `non_zero_ratio_overall`
6. `knowledge_version`
7. `effective_as_of_ts`

## 12. Які результати вже можна використати як робочу основу

Підтверджені локальні run-результати:
1. `MVP1_Baseline_train_05_bpi2012_GATv2`: `test_macro_f1 = 0.53510`
2. `Baseline_XES_bpi2012`: `test_macro_f1 = 0.36361`
3. `MVP2_5_Stage4_2_EOPKG_XES_Stat_bpi2012_stage4_2_xes_stat_GATv2`: `test_macro_f1 = 0.33687`
4. `EOPKG_XES_Stat_Edge_bpi2012`: `test_macro_f1 = 0.56098`

Робоча інтерпретація:
1. edge-only enrichment зараз є найсильнішим кандидатом на центральний позитивний результат;
2. full-stats треба перевіряти обережно;
3. старі static-protocol runs не можна механічно змішувати з новим drift-aware protocol у тій самій таблиці без примітки.

## 13. План робіт по статті

## Етап 1. Зафіксувати article scope

Результат етапу:
1. фінальна research question;
2. фінальний список datasets;
3. фінальний список model variants;
4. явне відмежування від двох попередніх статей.

Done, якщо:
1. затверджено заголовок і abstract line;
2. затверджено, що семафор не входить у scope;
3. затверджено 3 основні внески статті.

## Етап 2. Підготувати final configs

Потрібно:
1. створити final `50/50` configs для `BPI2012`;
2. створити final configs для `BPI2020`;
3. підготувати final configs для `cb` і `cd`;
4. закріпити `seed list`;
5. закріпити `strict_asof`;
6. закріпити `on_missing_asof_snapshot=raise` для evidence runs.

Done, якщо:
1. усі configs створені;
2. вони не залежать від UI state;
3. усі збережені в `configs/experiments/final_*`.

## Етап 3. Підготувати topology і stats artifacts

Потрібно:
1. `ingest-topology` або `sync-topology` для кожного dataset;
2. `sync-stats-backfill --step weekly` для stats variants;
3. зберегти summary files;
4. пройти alignment preflight.

Done, якщо:
1. для кожного dataset є topology artifact;
2. для stats runs є backfill timeline;
3. alignment metrics в межах acceptance thresholds.

## Етап 4. Провести train серію

Для кожного dataset і кожного seed:
1. baseline
2. structure
3. edge_stats
4. optional full_stats

Done, якщо:
1. усі train runs завершені;
2. checkpoints збережені;
3. MLflow params і metrics присутні;
4. немає конфлікту між seeds і configs.

## Етап 5. Провести drift evaluation серію

Для кожного train checkpoint:
1. eval_drift run;
2. збір per-window metrics;
3. експорт для plotting.

Done, якщо:
1. для кожного run є drift-window metrics;
2. усі timestamps збережені;
3. всі метрики зіставні між моделями.

## Етап 6. Побудувати figures і tables

Потрібно:
1. main comparison table;
2. drift curves;
3. ablation plot;
4. audit table по alignment/stats quality.

Done, якщо:
1. усі figures будуються з одного reproducible pipeline;
2. figures не дублюють фігури з двох попередніх статей;
3. у figures є чіткий новий сюжет: drift-safe structural enrichment.

## Етап 7. Написати статтю

Структура:
1. Introduction
2. Related Work
3. Method
4. Experimental Protocol
5. Results
6. Discussion
7. Limitations
8. Conclusion

Done, якщо:
1. у тексті немає повторення старих формулювань;
2. є чітка відмінність від попередніх двох статей;
3. contribution section узгоджений із реальними експериментами.

## 14. Anti-self-plagiarism checklist

Перед подачею статті перевірити:

1. Чи не повторюється назва або центральна research question попередніх статей.
2. Чи не повторюються таблиці `logs-only vs BPMN` як головний результат.
3. Чи не повторюються figures про varying prefix lengths як центральний сюжет.
4. Чи не копіюється abstract або conclusion майже дослівно.
5. Чи попередні статті процитовані як related work.
6. Чи явно описано, що нова стаття додає drift-aware protocol і strict-as-of stats.
7. Чи новий experimental section побудований навколо `structure vs edge_stats vs full_stats`, а не навколо вже опублікованих порівнянь.

## 15. Що можна повторно використати без ризику

Допустимо використати:
1. короткі визначення задачі next-activity prediction;
2. базовий опис dataset sources;
3. посилання на власні попередні результати;
4. окремі базові формули або notation, якщо вони потрібні для безперервності, але переписані заново і не є дослівним копіюванням.

## 16. Що треба написати заново

Обов'язково заново:
1. problem statement;
2. novelty statement;
3. experimental protocol;
4. methodology section;
5. results discussion;
6. conclusion.

Також треба зробити новими:
1. figures;
2. tables;
3. captions;
4. discussion of threats to validity.

## 17. Рекомендована структура розділу Results у статті

### 17.1 Natural evolution datasets
1. BPI2012
2. BPI2020

### 17.2 Abrupt drift datasets
1. `cb`
2. `cd`

### 17.3 Ablation
1. baseline vs structure
2. structure vs edge_stats
3. edge_stats vs full_stats

### 17.4 Discussion
1. де структура допомагає;
2. де statistics допомагають;
3. де full-stats починає шкодити;
4. чому alignment і strict-asof є критичними.

## 18. Планований головний висновок статті

Нова стаття має завершуватися не тезою "structure корисна взагалі", бо це вже частково було показано раніше, а тезою:

1. структура корисна в drift-aware setting;
2. historical statistics треба використовувати лише в anti-leakage режимі;
3. edge-level enrichment є практично сильним компромісом між користю і стабільністю;
4. quality enrichment залежить від alignment, тому protocol matters не менше, ніж сама модель.

## 19. Рекомендація по фактичному порядку виконання

Щоб стаття з'явилась швидко і без розмивання scope, порядок такий:
1. повністю закрити `BPI2012`
2. одразу зробити `cb/cd`
3. після цього `BPI2020`
4. лише потім вирішувати, чи потрібен `BPI2017`

Це дасть:
1. один сильний control dataset;
2. один сильний abrupt-drift block;
3. один external-validity dataset;
4. достатню доказову базу без перенасичення статті.
