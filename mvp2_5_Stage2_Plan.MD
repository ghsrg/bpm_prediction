# MVP2.5 Stage 2 Report (Validation & Hardening Plan)

Date: 2026-03-13  
Status: Planned

## 1. Мета Stage 2

Підтвердити, що Stage 1 refactor (Port + InMemoryNetworkXRepository + DI) є поведінково еквівалентним попередній реалізації, без регресій у MVP1/MVP2, і достатньо стабільний для переходу до Stage 3 (Camunda adapters).

## 2. Scope Stage 2

У Stage 2 виконуємо тільки валідацію та hardening:
1. Поглиблені тести репозиторію та портів.
2. Інтеграційна перевірка пайплайну після refactor.
3. Перевірка сумісності візуалізації через порт.
4. Розширення regression shield для критичних сценаріїв.

Не в scope:
1. Camunda ingestion.
2. Neo4j реалізація.
3. Зміни в архітектурі моделей.
4. Зміна формул метрик.

## 3. Детальний план робіт

## 3.1 Repository Hardening

Ціль: гарантувати коректність repository semantics для майбутнього backend swap.

Дії:
1. Додати тести на колізії ключів `process_name + version`.
2. Додати тести на поведінку `list_versions(process_name=None)` vs `list_versions(process_name=...)`.
3. Додати тести на частотний edge-filter (`min_edge_frequency`) для порогів 0 / 1 / N.
4. Додати тести на edge-case порожніх графів (без ребер, лише 1 вузол).
5. Додати тести на неоднозначний `get_process_structure(version)` (ambiguous index).

Результат:
- Повний контракт репозиторію задокументований тестами.

## 3.2 Service/Builder Integration Validation

Ціль: довести, що `TopologyExtractorService -> IKnowledgeGraphPort -> DynamicGraphBuilder` працює стабільно в реальних режимах `train`, `eval_cross_dataset`, `eval_drift`.

Дії:
1. Додати інтеграційний тест на end-to-end flow:
   - extract topology з train traces,
   - build graph contracts,
   - перевірка `allowed_target_mask` і structural tensors.
2. Додати тест на multi-process isolation:
   - два process_name в одному repo,
   - builder читає тільки свій process namespace.
3. Додати тест на unknown version behavior:
   - `process_version` відсутня в repo -> `allowed_target_mask=None` без падіння.
4. Додати тест на dataset_name-based version path:
   - XES без explicit version,
   - `dataset_name` проходить повний шлях до builder.

Результат:
- Інтеграційний контракт між компонентами зафіксований.

## 3.3 Visualization Through Port (No Re-read Policy)

Ціль: перевірити, що візуалізація повністю працює через порт знань і не залежить від повторного парсингу для доступу до topology graph.

Дії:
1. Додати тест, який перевіряє використання `get_graph_for_visualization(...)`.
2. Додати тест на правильну фільтрацію ребер та стабільність `plot_topology(...)`.
3. Додати тест на помилки:
   - process/version не знайдено,
   - після фільтрації не залишилось ребер.

Результат:
- Візуалізація готова до прозорої міграції на Neo4j backend у Stage 4.

## 3.4 Regression Shield Expansion

Ціль: зафіксувати критичні інваріанти MVP1/MVP2 в автоматичних перевірках.

Дії:
1. Додати марковані regression tests для:
   - відсутності хардкоду `"1"` у version lookup path.
   - DI wiring (shared repository instance між extractor/builder).
2. Додати smoke test `prepare_data()` для різних `dataset_name`.
3. Додати тест на стабільність Dual-Encoder forward після refactor даних.

Результат:
- Підсилений anti-regression контур перед Stage 3.

## 3.5 Documentation Sync

Ціль: мінімізувати технічну неоднозначність перед Camunda stage.

Дії:
1. Оновити відповідні MVP2.5 нотатки:
   - port contract,
   - key strategy (`process_name`, `version`),
   - visualization contract.
2. Зафіксувати migration notes:
   - які методи репозиторію є обов’язковими для Neo4j-адаптера.

Результат:
- Однозначний контракт для наступних етапів.

## 4. Test Execution Plan (Stage 2)

Обов’язкові прогони:
1. `python -m pytest -m mvp1_regression -v`
2. `python -m pytest tests/domain/test_eopkg_models_forward.py tests/domain/test_eopkg_gatv2_dual_encoder.py -v`
3. `python -m pytest tests/infrastructure/ -v`
4. `python -m pytest tests/application/test_topology_extractor_service.py tests/domain/test_dynamic_graph_builder_*.py -v`
5. `python -m pytest tests/application/test_visualize_topology_fallback.py -v`
6. (Фінальний) `python -m pytest tests/ -v`

## 5. Критерії приймання (Definition of Done)

Stage 2 вважається завершеним, якщо:
1. Усі нові/оновлені тести зелені.
2. `mvp1_regression` зелений.
3. Dual-Encoder model tests зелені.
4. Не виявлено витоків між process namespaces.
5. Візуалізація працює через knowledge port contract.
6. Немає змін у поведінці тренувального/евалуаційного пайплайнів (крім очікуваного logging/validation hardening).

## 6. Ризики Stage 2 та пом’якшення

1. Ризик: приховані залежності на service-local state.
- Пом’якшення: інтеграційні тести лише через repository state.

2. Ризик: колізії ключів між датасетами.
- Пом’якшення: strict tests на `process_name + version` ізоляцію.

3. Ризик: latent regression у drift/evaluator.
- Пом’якшення: прогони model + drift-related test suites після кожного блоку.

4. Ризик: розбіжність поведінки при заміні backend.
- Пом’якшення: чіткий repository contract test set як backend-agnostic compliance suite.

## 7. Покроковий порядок виконання

1. Розширити infrastructure tests для repository.
2. Додати integration tests extractor->builder.
3. Посилити visualization tests.
4. Запустити часткові прогони (targeted).
5. Запустити `mvp1_regression`.
6. Запустити model unit tests.
7. Запустити повний `tests/`.
8. Підготувати Stage 2 execution report з фактичними результатами.

## 8. Output артефакти Stage 2

1. Оновлені/нові тестові файли Stage 2.
2. Лог виконання тестів (targeted + full).
3. Stage 2 execution report (після імплементації плану).
