# ETALON_GAP_ANALYSIS

## ETALON vs CURRENT INTERFACE_CONTRACTS

| Area | CURRENT INTERFACE_CONTRACTS | ETALON фактично | Gap type |
|---|---|---|---|
| Event DTO | `EventRecord{case_id, activity_id, event_time, resource_id, process_version, payload}` | Немає TypedDict `EventRecord`; події читаються як таблиці (`ACT_HI_ACTINST`, `ACT_HI_TASKINST`, `bpm_tasks`) з Camunda/BPMS колонками. | Missing canonical DTO |
| process version κ | `stage_metadata.process_version` + `EventRecord.process_version` | `proc_def_version` переноситься в node attr `PROC_VER_`; stage metadata відсутня. | Semantic mismatch |
| Graph edge types | Фіксований enum (`control_flow`, `resource`, `data`, `normative`, `observed`) | Фактичні ребра: sequenceFlow (`id`,`name`), `boundaryLink`, `attached`, stitching без типу. | Mismatched semantics |
| Graph node DTO | `GraphNodeDTO{node_id,node_type,attrs,kappa,ref_bpmn_id}` | `networkx` nodes з наборами runtime attrs; немає формалізованого `node_type` enum contract. | Missing abstraction |
| Tensor DTO | `TensorGraphDTO{x,edge_index,edge_attr,y,run_metadata,stage_metadata}` | `Data` містить додаткові поля (`doc_features`, `time_target`, `node_ids`, `timestamps`) і не містить run/stage metadata. | Contract incompleteness |
| Label schema | Узагальнено (`y` без жорсткої формули) | AD: `y∈{0,1}` від registry normal/anomalous. PR: `y=node_map[next_node]` з prefix-next-event логіки. | Under-specified contract |
| Prefix | Не формалізовано | Prefix = `current_nodes` за `SEQUENCE_COUNTER_`; у тензорі кодується маскою в останньому каналі `x` (+ timestamps). | Missing invariant |
| Normalization | Є `feature_config_version`, але без конкретики | Min-max + category-index mapping через `global_statistics`; невалідні/непарсабельні значення -> `0.0`, unseen text -> `-1`. | Missing concrete spec |
| Drift/OOD | Очікується `DriftInputDTO/DriftResultDTO` (data/concept/wasserstein, ood_flag) | Немає окремого drift модуля у train loop; тільки `out_of_scope_rate` через валідність `global_node_dict`/`node_ids`. | Missing functionality |
| Train step contract | `train_step(...) -> TrainStepResultDTO` | Немає `train_step`; `train_epoch -> avg_loss(float)`. | API mismatch |
| Batch | Не деталізований | Ручний batch (cat/stack + `batch_tensor` + ad-hoc Batch object). | Implicit implementation detail |
| Metadata propagation | run+stage metadata mandatory | Реально передається тільки `doc_info -> doc_features`, `global_node_dict`; решти metadata немає. | Missing metadata propagation |

## Missing invariants (explicit list)
1. Потрібен `SEQUENCE_COUNTER_ > 0` для побудови префіксів і target node.
2. Потрібна консистентність `edge_index`/`edge_attr` по E.
3. Потрібні `node_count.max` і `edge_count.max` для padding у PR-моделях.
4. `doc_features` повинні мати стабільну розмірність у батчі.
5. `global_node_dict` має бути узгоджений із `node_ids` для коректного `out_of_scope_rate`.
