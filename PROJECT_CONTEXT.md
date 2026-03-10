# PROJECT_CONTEXT.md

Оновлено: 2026-03-10  
Призначення: короткий актуальний контекст для агентів у періоді переходу MVP1 -> MVP2.

---

## 1. Project State

`bpm_prediction` тепер працює у **dual-track** режимі:

- **MVP1**: завершений і стабільний logs-only baseline.
- **MVP2**: активна розробка нових можливостей.

Ключова вимога: MVP1 експерименти мають завжди залишатися runnable.

---

## 2. MVP1 Baseline (Frozen Runtime Contract)

Базовий потік даних MVP1:

`XESAdapter -> RawTrace -> PrefixPolicy -> PrefixSlice -> BaselineGraphBuilder -> GraphTensorContract -> ModelTrainer`

Критичні інваріанти MVP1:

- `SchemaResolver` в MVP1 — identity-resolver (без semantic mapping).
- Categorical unknown -> `<UNK>=0`.
- Numeric missing для z-score -> mu-imputation (нульовий normalized signal).
- Без EPOKG/Critic/Reliability/Continual Learning.

---

## 3. MVP2 Policy

Новий функціонал розробляється так, щоб не ламати MVP1:

- через окремі `mode`/конфіг-параметри;
- через нові конфіг-файли (рекомендовано окремий префікс/каталог для MVP2);
- зі збереженням backward compatibility за замовчуванням.

---

## 4. Config Ownership

У поточній структурі:

- `data` — статичні властивості датасету (шляхи/label тощо).
- експериментальні гіперпараметри розщеплення/режимів — у `experiment`.

Це потрібно, щоб один dataset-config не дублювався під багато експериментів.

---

## 5. MVP1 Experiments Must Stay Handy

Мінімальний список MVP1 конфігів, що мають лишатися робочими:

- `configs/experiments/01_train_bpi2012.yaml`
- `configs/experiments/01_eval_drift_bpi2012.yaml`
- `configs/experiments/01_eval_cross_dataset_bpi2012.yaml`

Запуск:

```bash
python main.py --config configs/experiments/01_train_bpi2012.yaml
```

---

## 6. Operational Gate Before Merge

Для будь-яких змін у перехідний період:

1. `py -3 tools/architecture_guard.py`
2. `pytest tests/ -v`
3. Швидка перевірка хоча б одного MVP1 запуску (train або eval)

---

## 7. Source of Truth Priority

1. `AGENT_GUIDE.MD`
2. `PROJECT_CONTEXT.md`
3. `docs/ARCHITECTURE_GUIDELINES.MD`
4. `docs/ARCHITECTURE_RULES.md`
5. решта docs MVP1/MVP2

