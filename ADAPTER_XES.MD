# ADAPTER_XES.MD

## Статус
`DRAFT (DOC-007 aligned)`

## Призначення
Специфікація адаптера для імпорту XES-логів у canonical контракт `EventLogBatch`.

## Мінімальні вимоги
1. Підтримка читання `.xes` та базової валідації структури trace/event.
2. Мапінг у поля: `case_id`, `activity_id`, `event_time`, `resource_id`, `process_version`, `payload`.
3. Проставлення `metadata`: `dataset_id`, `schema_version`, `source_type=xes`.
4. У разі відсутнього `process_version` застосувати configurable fallback (`unknown_kappa`) з warning-логом.

## Контракти
- Input: `file_path`.
- Output: `EventLogBatch`.
- Errors: invalid xml/xes schema, empty traces, timestamp parse error.

## Примітка
Детальні сигнатури та ABC визначені в `INTERFACE_CONTRACTS.MD`.
