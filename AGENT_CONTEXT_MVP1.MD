# AGENT_CONTEXT_MVP1.MD

Останнє оновлення: 2026-03-09
Статус: після аудиту архітектурних розбіжностей MVP1

## Що виправлено

1. `GraphTensorContract` синхронізовано на рівні гайду:
- В `AGENT_GUIDE.MD` зафіксовано актуальний контракт:
  - `x_cat: torch.LongTensor`
  - `x_num: torch.FloatTensor`
  - `edge_index: torch.LongTensor`
  - `edge_type: torch.LongTensor`
  - `y: torch.LongTensor`
  - `batch: torch.LongTensor`
  - `num_nodes: int`

2. Прибрано порушення напрямку залежностей (Domain -> Application):
- `src/domain/services/prefix_policy.py` більше не імпортує `src.application.ports.*`.
- `src/domain/services/baseline_graph_builder.py` більше не імпортує `src.application.ports.*`.
- Domain-сервіси працюють через duck typing і залишаються незалежними від Application.

3. `tools/architecture_guard.py` оновлено під поточну структуру:
- Перевіряє `src/domain`, `src/application`, `src/adapters`.
- Блокує:
  - імпорти `src.application|src.adapters|src.infrastructure` із Domain;
  - імпорти `src.adapters` із Application;
  - імпорти `src.application.use_cases` із Adapters.
- Команда перевірки:
  - `py -3 tools/architecture_guard.py`

4. Стек залежностей синхронізовано:
- `requirements.txt` доповнено фактично використаними пакетами:
  - `numpy`
  - `matplotlib`
  - `tqdm`
- `AGENT_GUIDE.MD` оновлено: дозволений стек тепер включає `PyYAML`, `psutil`, `tqdm`, `matplotlib`.

## Поточний інваріант архітектури

- `src/domain` не залежить від `src/application` та `src/adapters`.
- `src/application` оркеструє use-cases і залежить від domain + портів.
- `src/adapters` реалізують інфраструктурні інтеграції, не залежать від concrete use-cases.

## Правила для наступних AI-агентів

- Не повертати єдиний `x` у `GraphTensorContract` без узгодженого RFC для всіх MVP1 доків і коду.
- Не імпортувати `src.application.*` із `src/domain/*`.
- Перед PR/merge завжди запускати:
  - `py -3 tools/architecture_guard.py`
- При додаванні нових прямих імпортів сторонніх бібліотек:
  - оновити `requirements.txt`;
  - синхронізувати список у `AGENT_GUIDE.MD` (розділ стеку).
