# LLD_MVP1.MD (Low-Level Design)

## 1. Статус
`APPROVED` (Scope: MVP1 - Baseline GNN)

## 2. Призначення
Фіксує low-level алгоритми MVP1 для prefix-generation, feature-encoding, tensor construction та навчання baseline GNN.

---

## 3. Prefix Generation (`PrefixPolicy`)
### 3.1. Математика нарізки
Для траси \(\sigma=(e_1,\dots,e_T)\) формуються префікси \(\sigma_{\le k}\), де цільова подія: \(y=e_{k+1}\).

### 3.2. Anti-leakage
Жоден префікс не містить цільову подію або майбутні події:
\[
\sigma_{\le k} \cap \{e_{k+1},\dots,e_T\}=\varnothing
\]

---

## 4. Feature Engineering (\(\Phi_{enc}\))

## 4.1. Категоріальні ознаки
- `activity_id`, `resource_id` та інші embedding-ознаки кодуються індексами словника.
- `\langle UNK \rangle` має індекс `0`.

## 4.2. Неперервні та часові ознаки
У MVP1 активні:
1. `z-score` стандартизація;
2. `time2vec_dayofweek`, `time2vec_month`, `time2vec_dayofmonth`, `time2vec_weekofyear`, `time2vec_hour`;
3. `time2vec_work_hours`.

Для числової ознаки:
\[
z = \frac{x-\mu}{\sigma}
\]
Далі застосовується policy-driven clipping:
\[
z_{clip}=\min\left(z_{max},\max(z_{min},z)\right)
\]

## 4.3. Policy-driven керування
Поведінка енкодера визначається `policies` у конфігу:
- `zscore_clip.{enabled,min,max}`;
- `time2vec.work_hours.{working_days,start_hour,end_hour,timezone}`;
- `activity_fallback_feature`.

Hardcoded dataset-specific policy у коді заборонено.

## 4.4. Schema resolution
Lookup значень ознак виконується через `SchemaResolver`:
\[
K_{lookup}=[name, source\_key, fallback\_keys]
\]
У MVP1:
\[
\operatorname{resolve\_value}(cfg,v)=v
\]
(без semantic synonym mapping).

---

## 5. Graph Tensor Construction

`BaselineGraphBuilder` формує контракт:
\[
X_{cat} \in \mathbb{N}^{N\times F_{cat}},\;
X_{num} \in \mathbb{R}^{N\times F_{num}},\;
edge\_index \in \mathbb{N}^{2\times E}
\]
\[
y \in \{0,\dots,C-1\},\quad batch \in \mathbb{N}^{N}
\]

Топологія MVP1: лише послідовні ребра \((e_i \rightarrow e_{i+1})\).

---

## 6. Baseline GNN
1. Вхід: `GraphTensorContract`.
2. Message passing layers (GCN/GATv2).
3. Readout (global pooling).
4. Output logits \(\hat{Y}\in\mathbb{R}^{B\times C}\).

Loss:
\[
\mathcal{L}_{total}=\mathcal{L}_{task}\;\text{(CrossEntropy)}
\]

---

## 7. Trainer Orchestration (MVP1)
`ModelTrainer.run()` є dispatcher для:
- `_run_train_pipeline`
- `_run_eval_cross_dataset`
- `_run_eval_drift`

Дрейф-оцінка виконує: chronological sort -> window generation -> evaluate -> log.
