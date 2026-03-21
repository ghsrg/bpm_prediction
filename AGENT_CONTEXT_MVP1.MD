# AGENT_CONTEXT_MVP1.MD

Останнє оновлення: 2026-03-21  
Статус: MVP1 зафіксовано як стабільний baseline, підтримується як non-regression контур.

## 1) Поточна роль MVP1 у проєкті

1. MVP1 залишається еталонним logs-only шляхом для перевірки регресій.
2. Усі розширення MVP2/MVP2.5 мають бути additive і не ламати MVP1-config сценарії.
3. Значення за замовчуванням у спільних компонентах мають зберігати MVP1-поведінку.

## 2) Незмінні інваріанти MVP1

1. Контракт пайплайна не змінюється:
   - `RawTrace -> PrefixSlice -> GraphTensorContract`.
2. MVP2/MVP2.5 поля в контрактах допускаються лише як optional.
3. Якщо структурні/статистичні тензори відсутні, runtime не має падати й працює як baseline.
4. Нова логіка має вмикатись лише через явні конфіг-прапори або режими.

## 3) Канонічні MVP1 сценарії запуску

1. `configs/experiments/01_train_bpi2012.yaml`
2. `configs/experiments/01_eval_drift_bpi2012.yaml`
3. `configs/experiments/01_eval_cross_dataset_bpi2012.yaml`

Базова команда:
```bash
python main.py --config <path_to_mvp1_experiment_yaml>
```

## 4) Обов'язковий regression gate для будь-яких змін

1. `py -3 tools/architecture_guard.py`
2. `.\.venv\Scripts\python.exe -m pytest -m mvp1_regression -v`
3. `.\.venv\Scripts\python.exe -m pytest tests/ -v`

## 5) Політика сумісності з MVP2/MVP2.5

1. Дозволено розширювати лише через:
   - нові експериментальні конфіги (`mvp2_*`, `mvp2_5_*`),
   - нові optional-поля DTO/тензорних контрактів.
2. Заборонено:
   - змінювати MVP1 YAML так, щоб вони перестали запускатися,
   - переносити MVP2/MVP2.5 поведінку в MVP1 path за замовчуванням.
3. Якщо змінюється shared-компонент, потрібен окремий тест на збереження MVP1 behavior.
