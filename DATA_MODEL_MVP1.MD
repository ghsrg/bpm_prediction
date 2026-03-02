# DATA_MODEL_MVP1

**Project:** bpm_prediction\
**Scope:** MVP1 (G_obs only)\
**Purpose:** Formal data model specification for implementation.

------------------------------------------------------------------------

## 0. Canon & Naming Rules

-   All variables use snake_case.
-   Canonical names must match VARIABLES.MD.
-   MVP1 operates only on observed_graph (G_obs).

------------------------------------------------------------------------

## 1. Attribute Tiering (Класифікація атрибутів)

### 1.1 Fundamental (System Core)

Critical identifiers and ordering fields:

-   case_id - ID траси/кейсу (джерело: XES trace concept:name)
-   activity_id — назва/код активності (XES event concept:name)
-   timestamp — час події (XES time:timestamp)
-   position_in_trace — позиція події у трасі (обчислюється)
-   event_index - стабільний індекс події у межах траси (обчислюється)

Invariant: Without these fields graph construction is impossible.

### 1.2 Base (Standard BPM)

-   duration
-   resource_id

Must be available at time t (no leakage).

### 1.3 Custom (Disabled in MVP1)

Business-specific fields (not part of node_features in MVP1).

### 1.4 Anti-Leakage Rule

Any feature must be available at event time t.\
No future-dependent aggregates allowed.

------------------------------------------------------------------------

## 2. Identification & Hierarchy

### 2.1 IDs in MVP1

-   case_id (Trace ID)
-   event_id (deterministic hash)
-   activity_id (activity template ID)

Invariant: event_id must be reproducible.

------------------------------------------------------------------------

## 3. External Boundary Objects

### 3.1 EventRecord

Normalized event from XES.

Fields: - case_id - event_id - activity_id - timestamp - resource_id
(optional) - duration (optional) - position_in_trace

### 3.2 RawTrace

Ordered list of EventRecord.

Fields: - case_id - events: List\[EventRecord\]

Invariant: events sorted by timestamp.

------------------------------------------------------------------------

## 4. Experiment-Level Objects

### 4.1 ExperimentConfig

Fields: - dataset - seed - split_strategy - prefix_policy -
prefix_length - process_version

### 4.2 ModelConfig

Fields: - model_type - hidden_dim - num_layers - dropout

### 4.3 SplitConfig

Fields: - train_ratio - val_ratio - test_ratio - split_seed

Invariant: trace-level split only.

------------------------------------------------------------------------

## 5. Application DTO

### 5.1 PrefixSlice

Fields: - case_id - prefix_events - prefix_length

### 5.2 PredictionRequest

Fields: - process_version - prefix_slice

### 5.3 TrainingBatch

Fields: - graph_batch (PyG Batch)

------------------------------------------------------------------------

## 6. Domain Core Objects

### 6.1 InstanceGraph (observed_graph)

Graph representation of prefix execution.

Nodes: - activity_id - timestamp - resource_id - duration -
position_in_trace

Edges: - sequential_relation only

### 6.2 GraphTensorContract

Required tensors:

-   x (node_features) \[num_nodes, node_dim\]
-   edge_index \[2, num_edges\]
-   batch \[num_nodes\]
-   y \[batch_size\]

Optional (MVP2+): - edge_attr - doc_features

Invariant: identical construction for train and inference.

------------------------------------------------------------------------

## 7. Modeling & Inference Objects

### 7.1 GNNModel

Input: graph_batch\
Output: logits

### 7.2 FusionOperator (Γ)

Input: prefix_embedding + context_embedding\
Output: fused_representation

### 7.3 ContextEmbedding

Embedding of process_version (κ).

### 7.4 MaskLayer (IdentityMask)

No-op in MVP1.

### 7.5 Critic (NullCritic)

No-op in MVP1.

### 7.6 ReliabilityModule

confidence_score = max(softmax(logits))

------------------------------------------------------------------------

## 8. Training Objects

### 8.1 LossBundle

prediction_loss only in MVP1.

### 8.2 MetricBundle

Required: - accuracy - top_k\_accuracy - precision - recall - f1_score -
confusion_matrix - train_loss - validation_loss - training_time

------------------------------------------------------------------------

## 9. Planned Placeholders (Inactive in MVP1)

-   structural_mask
-   critic_loss
-   drift_signal
-   macro_stability_state
-   reliability_state
-   version_transition_operator
