# MVP2.5 Topology-Native Candidate Semantic Grounding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement Level 3 Semantic Grounding for `EOPKGTopologyConditioned` candidate scoring so that unseen activities (such as `t_monitor_ten` in `v4`) receive a meaningful semantic representation rather than a generic shared zero vector. This enables zero-shot alignment based on semantic name similarities between prefix execution states and new topology candidate nodes.

**Architecture:** Introduce a configuration-driven `SemanticEmbeddingProvider` that maps activity label strings to dense semantic vectors. The `DynamicGraphBuilder` will query this provider during graph construction and attach `struct_semantic_x` to the PyG `GraphTensorContract`. The model `EOPKGTopologyConditioned` will project these semantic representations and fuse them with parametric node identity embeddings *before* structural statistics are added, ensuring that unseen process version candidates are semantically grounded from their task names without distorting graph features.

**Tech Stack:** Python, PyTorch, project `GraphTensorContract`, `DynamicGraphBuilder`, `EOPKGTopologyConditioned`, pytest.

---

## Technical Specifications

### 1. Configuration Additions

We will introduce the following config keys under `model.semantic_grounding`:

```yaml
model:
  semantic_grounding:
    enabled: true
    mode: "lexical_hash_bow"  # "lexical_hash_bow" or "static_dictionary"
    dictionary_path: "configs/semantics/loan_v1_v4_hash_bow_embeddings.json"
    alpha: 0.5  # Seen candidates fusion weight
    unseen_alpha: 0.9  # Unseen candidates fusion weight (strongly recommended to be larger)
    embedding_dim: 32  # Output dimension of the semantic embedding before projection
```

These configuration parameters must be declared in `configs/ui/config_catalog.yaml`.

### 2. Semantic Embedding Providers

To remain 100% offline-compatible and robust in all execution sandboxes, we support two modes:
1. `lexical_hash_bow`: A deterministic character n-gram / word bag representation normalized to unit L2 length. This requires 0 external dependencies, is fast, and computes embeddings dynamically for any novel label name based on keyword overlaps (e.g., `t_approve_loan` and `t_reject_loan` share the word `loan`). MD5 hashing is used to ensure cross-process determinism.
2. `static_dictionary`: Loads a pre-computed JSON dictionary mapping label names to float lists.

### 3. Fused Node Representation

In `EOPKGTopologyConditioned._struct_input`, candidate identity is updated using adaptive interpolation weights before adding `struct_x`:

$$h_{semantic\_identity} = (1 - \alpha_{eff}) \cdot h_{identity} + \alpha_{eff} \cdot \text{proj}(x_{semantic})$$

$$\alpha_{eff} = \begin{cases} \alpha & \text{if candidate is seen} \\ \alpha_{unseen} & \text{if candidate is unseen} \end{cases}$$

$$h_0 = \text{struct\_norm} \left( h_{semantic\_identity} + \text{proj}(struct\_x) \right)$$

where:
- $h_{identity}$ is the learned candidate class embedding `self.struct_node_emb` (or `self.struct_non_target_emb` for unseen candidates).
- $x_{semantic}$ is the semantic vector of the candidate label.
- $\text{proj}$ is a linear projection `nn.Linear(embedding_dim, hidden_dim)`.
- $struct\_x$ is the structural stats projection (if stats are enabled).

---

## File Map

### New Files
- `src/domain/services/semantic_embedding_provider.py`
  - Encapsulates semantic vector generation (`lexical_hash_bow` and `static_dictionary` parser).
- `tests/domain/test_semantic_embedding_provider.py`
  - Unit tests verifying embedding generation determinism, correctness, and L2 normalization.
- `configs/semantics/loan_v1_v4_hash_bow_embeddings.json`
  - JSON dictionary mapping all 38 labels (seen + unseen) to 32-dimensional deterministic BoW embeddings.

### Modified Files
- `src/domain/entities/tensor_contract.py`
  - Add `struct_semantic_x` to `GraphTensorContract` TypedDict.
- `configs/ui/config_catalog.yaml`
  - Add configuration schemas for `model.semantic_grounding.*`.
- `src/domain/services/dynamic_graph_builder.py`
  - Load and query `SemanticEmbeddingProvider` at initialization;
  - Attach `struct_semantic_x` tensor of shape `[K_v, D_sem]` to the `GraphTensorContract`.
- `src/domain/models/topology_conditioned.py`
  - Add `self.semantic_proj` linear layer;
  - Modify `_struct_input` to project and fuse `struct_semantic_x` into candidate node representations.
- `src/cli.py`
  - Expose and parse semantic config, instantiate and pass semantic config to the graph builder, and bump `GRAPH_DATASET_CACHE_SCHEMA` to 5.
- `tests/application/test_cli_flatten_config.py`
  - Add unit tests verifying kwargs routing.

---

## Task Checklist

### Task 1: Extend Tensor Contract and Model Config Routing

**Files:**
- Modify: `src/domain/entities/tensor_contract.py:20-40`
- Modify: `src/cli.py:270-320`
- Modify: `src/domain/models/topology_conditioned.py:24-45`
- Modify: `tests/domain/test_eopkg_topology_conditioned.py:35-55`
- Modify: `tests/application/test_cli_flatten_config.py:250-285`

- [ ] **Step 1: Write model routing tests**

Add a test case in `tests/domain/test_eopkg_topology_conditioned.py` to verify `semantic_*` constructor parameters can be parsed:

```python
def test_topology_conditioned_constructor_routes_semantic_params():
    model = _model(
        semantic_grounding_enabled=True,
        semantic_alpha=0.6,
        semantic_unseen_alpha=0.95,
        semantic_embedding_dim=16
    )
    assert model.semantic_grounding_enabled is True
    assert model.semantic_alpha == 0.6
    assert model.semantic_unseen_alpha == 0.95
    assert model.semantic_embedding_dim == 16
```

Add a kwargs parsing test case to `tests/application/test_cli_flatten_config.py`:

```python
def test_build_model_factory_kwargs_routes_semantic_grounding():
    from src.cli import _build_model_factory_kwargs
    from tests.domain.test_eopkg_topology_conditioned import _layout
    
    cfg = {
        "type": "EOPKGTopologyConditioned",
        "semantic_grounding": {
            "enabled": True,
            "alpha": 0.6,
            "unseen_alpha": 0.95,
            "embedding_dim": 16,
        }
    }
    kwargs = _build_model_factory_kwargs(model_cfg=cfg, feature_layout=_layout(), output_dim=5)
    assert kwargs["semantic_grounding_enabled"] is True
    assert kwargs["semantic_alpha"] == 0.6
    assert kwargs["semantic_unseen_alpha"] == 0.95
    assert kwargs["semantic_embedding_dim"] == 16
```

- [ ] **Step 2: Run tests to verify they fail**

Run:
```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests/domain/test_eopkg_topology_conditioned.py -k "test_topology_conditioned_constructor_routes_semantic_params" -v
.\.venv-modern\Scripts\python.exe -m pytest tests/application/test_cli_flatten_config.py -k "test_build_model_factory_kwargs_routes_semantic_grounding" -v
```
Expected: FAIL.

- [ ] **Step 3: Update `tensor_contract.py`, `topology_conditioned.py`, and `cli.py`**

In `src/domain/entities/tensor_contract.py`, add `struct_semantic_x` to `GraphTensorContract` TypedDict:
```python
    struct_semantic_x: NotRequired[torch.FloatTensor | None]
```

In `src/domain/models/topology_conditioned.py`, update `__init__` signature and fields:
```diff
     def __init__(
         self,
         feature_layout: FeatureLayout,
         hidden_dim: int,
         output_dim: int,
         dropout: float = 0.2,
         pooling_strategy: str = "last_node",
         observed_encoder: str = "GATv2",
         struct_encoder: str = "GATv2",
         candidate_scoring: str = "cosine",
         candidate_pooling: str = "logmeanexp",
         candidate_temperature_init: float = 0.1,
         candidate_temperature_min: float = 0.05,
         candidate_temperature_max: float = 10.0,
         candidate_temperature_trainable: bool = False,
+        semantic_grounding_enabled: bool = False,
+        semantic_alpha: float = 0.5,
+        semantic_unseen_alpha: float = 0.9,
+        semantic_embedding_dim: int = 32,
     ) -> None:
         super().__init__()
         self.feature_layout = feature_layout
         self.hidden_dim = int(hidden_dim)
         self.output_dim = int(output_dim)
+        self.semantic_grounding_enabled = bool(semantic_grounding_enabled)
+        self.semantic_alpha = float(semantic_alpha)
+        self.semantic_unseen_alpha = float(semantic_unseen_alpha)
+        self.semantic_embedding_dim = int(semantic_embedding_dim)
```

In `src/cli.py` inside `_build_model_factory_kwargs`, parse and append semantic config properties:
```python
    semantic_grounding_cfg = model_cfg.get("semantic_grounding")
    if not isinstance(semantic_grounding_cfg, dict):
        semantic_grounding_cfg = {}

    return {
        # ... existing keys ...
        "semantic_grounding_enabled": _as_bool(semantic_grounding_cfg.get("enabled"), default=False),
        "semantic_alpha": float(semantic_grounding_cfg.get("alpha", 0.5)),
        "semantic_unseen_alpha": float(semantic_grounding_cfg.get("unseen_alpha", 0.9)),
        "semantic_embedding_dim": int(semantic_grounding_cfg.get("embedding_dim", 32)),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests/domain/test_eopkg_topology_conditioned.py -k "test_topology_conditioned_constructor_routes_semantic_params" -v
.\.venv-modern\Scripts\python.exe -m pytest tests/application/test_cli_flatten_config.py -k "test_build_model_factory_kwargs_routes_semantic_grounding" -v
```
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/domain/entities/tensor_contract.py src/domain/models/topology_conditioned.py src/cli.py tests/domain/test_eopkg_topology_conditioned.py tests/application/test_cli_flatten_config.py
git commit -m "feat: extend GraphTensorContract and add model config routing for semantic grounding"
```

---

### Task 2: Semantic Embedding Provider

**Files:**
- Create: `src/domain/services/semantic_embedding_provider.py`
- Create: `tests/domain/test_semantic_embedding_provider.py`

- [ ] **Step 1: Write failing unit tests for SemanticEmbeddingProvider**

Create `tests/domain/test_semantic_embedding_provider.py`:

```python
import pytest
import torch
from src.domain.services.semantic_embedding_provider import SemanticEmbeddingProvider

def test_lexical_hash_bow_provider_normalizes_and_matches_similar_labels():
    provider = SemanticEmbeddingProvider(mode="lexical_hash_bow", embedding_dim=16)
    emb_approve = provider.get_embedding("t_approve_loan")
    emb_reject = provider.get_embedding("t_reject_loan")
    emb_monitor = provider.get_embedding("t_monitor_one")

    assert emb_approve.shape == (16,)
    # Check L2 normalization
    assert pytest.approx(float(torch.norm(emb_approve).item()), 1e-5) == 1.0
    
    # "t_approve_loan" and "t_reject_loan" share the word "loan"
    similarity_loan = float(torch.dot(emb_approve, emb_reject).item())
    # "t_approve_loan" and "t_monitor_one" share no words
    similarity_monitor = float(torch.dot(emb_approve, emb_monitor).item())
    
    assert similarity_loan > similarity_monitor
    assert similarity_loan > 0.0

def test_lexical_hash_bow_is_deterministic_across_provider_instances():
    p1 = SemanticEmbeddingProvider(mode="lexical_hash_bow", embedding_dim=16)
    p2 = SemanticEmbeddingProvider(mode="lexical_hash_bow", embedding_dim=16)

    e1 = p1.get_embedding("t_monitor_ten")
    e2 = p2.get_embedding("t_monitor_ten")

    assert torch.allclose(e1, e2)

def test_lexical_hash_bow_handles_empty_and_unknown_labels():
    provider = SemanticEmbeddingProvider(mode="lexical_hash_bow", embedding_dim=16)
    e_empty = provider.get_embedding("")
    e_unknown = provider.get_embedding("t_unk")
    
    assert e_empty.shape == (16,)
    assert e_unknown.shape == (16,)
    assert pytest.approx(float(torch.norm(e_empty).item()), 1e-5) == 1.0
    assert pytest.approx(float(torch.norm(e_unknown).item()), 1e-5) == 1.0
```

Run test:
```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests/domain/test_semantic_embedding_provider.py -v
```
Expected: Fail (ModuleNotFoundError or import error).

- [ ] **Step 2: Implement SemanticEmbeddingProvider with deterministic hashing**

Create `src/domain/services/semantic_embedding_provider.py`:

```python
from __future__ import annotations
import json
import os
import re
import hashlib
import torch

class SemanticEmbeddingProvider:
    def __init__(
        self,
        mode: str = "lexical_hash_bow",
        embedding_dim: int = 32,
        dictionary_path: str | None = None,
    ) -> None:
        self.mode = str(mode).strip().lower()
        # Handle "bag_of_words" alias for backward compatibility
        if self.mode == "bag_of_words":
            self.mode = "lexical_hash_bow"
            
        self.embedding_dim = int(embedding_dim)
        self.dictionary_path = dictionary_path
        self.static_dict: dict[str, list[float]] = {}

        if self.mode == "static_dictionary":
            if not dictionary_path or not os.path.exists(dictionary_path):
                raise ValueError(f"dictionary_path must be valid if mode='static_dictionary'. Path: {dictionary_path}")
            with open(dictionary_path, "r", encoding="utf-8") as f:
                self.static_dict = json.load(f)

    def get_embedding(self, label: str) -> torch.Tensor:
        label_str = str(label).strip()
        if self.mode == "static_dictionary":
            val = self.static_dict.get(label_str)
            if val is not None:
                if len(val) != self.embedding_dim:
                    raise ValueError(f"Label '{label_str}' embedding size {len(val)} != expected {self.embedding_dim}")
                return torch.tensor(val, dtype=torch.float32)
            # Fallback to lexical Bow if label missing in dictionary
            return self._compute_lexical_hash_bow(label_str)
        return self._compute_lexical_hash_bow(label_str)

    def _compute_lexical_hash_bow(self, label: str) -> torch.Tensor:
        # Split string by non-alphanumeric chars and remove t_ prefix
        cleaned = re.sub(r"^t_", "", label.lower())
        words = [w for w in re.split(r"[_\s]+", cleaned) if w]
        
        vector = torch.zeros(self.embedding_dim, dtype=torch.float32)
        if not words:
            # Fallback unit vector if empty
            vector[0] = 1.0
            return vector

        # Compute stable hash-based word embeddings to form a lexical bag-of-words projection
        for word in words:
            for char_idx, char in enumerate(word):
                # Use deterministic MD5 hash instead of builtin hash() to prevent process-seed randomness
                key = f"{word}_{char}_{char_idx}".encode("utf-8")
                h_idx = int(hashlib.md5(key).hexdigest(), 16) % self.embedding_dim
                vector[h_idx] += 1.0

        norm = torch.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector
```

- [ ] **Step 3: Run unit tests to verify they pass**

Run:
```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests/domain/test_semantic_embedding_provider.py -v
```
Expected: PASS.

- [ ] **Step 4: Commit**
```bash
git add src/domain/services/semantic_embedding_provider.py tests/domain/test_semantic_embedding_provider.py
git commit -m "feat: add deterministic SemanticEmbeddingProvider with lexical BoW hashes"
```

---

### Task 3: Expose Config Catalog Keys

**Files:**
- Modify: `configs/ui/config_catalog.yaml`

- [ ] **Step 1: Modify catalog to expose semantic grounding configurations**

Open `configs/ui/config_catalog.yaml` and append to the `model` section:

```yaml
  model.semantic_grounding.enabled:
    affects: Enables semantic grounding features for GNN candidates.
    default: false
    description: Enable Level 3 Semantic Grounding for candidate GNN features.
    enum:
    - 'false'
    - 'true'
    label: 'Model: Semantic Grounding Enabled'
    path: model.semantic_grounding.enabled
    required_in_modes: []
    required_when:
      model.type: EOPKGTopologyConditioned
    section: model
    ui:
      group: core
      order: 34
      priority: 20
      tab: model
  model.semantic_grounding.mode:
    affects: Strategy for building semantic embedding vectors.
    default: lexical_hash_bow
    description: Strategy for semantic vectors. lexical_hash_bow or static_dictionary.
    enum:
    - lexical_hash_bow
    - static_dictionary
    label: 'Model: Semantic Grounding Mode'
    path: model.semantic_grounding.mode
    required_in_modes: []
    required_when:
      model.semantic_grounding.enabled: 'true'
      model.type: EOPKGTopologyConditioned
    section: model
    ui:
      group: core
      order: 35
      priority: 20
      tab: model
  model.semantic_grounding.dictionary_path:
    affects: Source file path for pre-computed semantic embeddings dictionary.
    default: configs/semantics/loan_v1_v4_hash_bow_embeddings.json
    description: JSON path for pre-computed semantic embeddings dictionary.
    enum: []
    label: 'Model: Semantic Embedding Dict Path'
    path: model.semantic_grounding.dictionary_path
    required_in_modes: []
    required_when:
      model.semantic_grounding.enabled: 'true'
      model.semantic_grounding.mode: static_dictionary
      model.type: EOPKGTopologyConditioned
    section: model
    ui:
      group: advanced
      order: 1036
      priority: 100
      tab: model
  model.semantic_grounding.alpha:
    affects: Weight parameter for semantic vs learned candidate embedding interpolation.
    default: 0.5
    description: Interpolation weight between candidate class embedding and projected semantic embedding.
    enum: []
    label: 'Model: Semantic Grounding Alpha'
    path: model.semantic_grounding.alpha
    required_in_modes: []
    required_when:
      model.semantic_grounding.enabled: 'true'
      model.type: EOPKGTopologyConditioned
    section: model
    ui:
      group: core
      order: 36
      priority: 20
      tab: model
  model.semantic_grounding.unseen_alpha:
    affects: Weight parameter for semantic grounding on unseen candidates.
    default: 0.9
    description: Interpolation weight between candidate class embedding and projected semantic embedding for unseen nodes.
    enum: []
    label: 'Model: Semantic Grounding Unseen Alpha'
    path: model.semantic_grounding.unseen_alpha
    required_in_modes: []
    required_when:
      model.semantic_grounding.enabled: 'true'
      model.type: EOPKGTopologyConditioned
    section: model
    ui:
      group: core
      order: 37
      priority: 20
      tab: model
  model.semantic_grounding.embedding_dim:
    affects: Dimensionality of semantic embeddings.
    default: 32
    description: Output dimension of raw semantic embedding before linear projection.
    enum: []
    label: 'Model: Semantic Grounding Dim'
    path: model.semantic_grounding.embedding_dim
    required_in_modes: []
    required_when:
      model.semantic_grounding.enabled: 'true'
      model.type: EOPKGTopologyConditioned
    section: model
    ui:
      group: advanced
      order: 1038
      priority: 100
      tab: model
```

- [ ] **Step 2: Run validation command to ensure no config errors**

Run:
```powershell
.\.venv-modern\Scripts\python.exe tools\architecture_guard.py
```
Expected: `[ARCH_GUARD] OK`.

- [ ] **Step 3: Commit**
```bash
git add configs/ui/config_catalog.yaml
git commit -m "config: add semantic grounding configuration schemas to catalog"
```

---

### Task 4: Static Embedding Dictionary and Tooling

**Files:**
- Create: `configs/semantics/loan_v1_v4_hash_bow_embeddings.json`
- Create: `tools/generate_semantic_embeddings.py`

- [ ] **Step 1: Write generation script with deterministic hash**

Create `tools/generate_semantic_embeddings.py`:

```python
import json
import os
import hashlib
import re

labels = [
    't_add_notes', 't_approve_loan', 't_assess_eligibility', 't_audit_case', 't_book_loan', 
    't_call_client', 't_cancel_loan', 't_check_form', 't_check_funding', 't_check_quote', 
    't_collect_docs', 't_confirm_payout', 't_create_account', 't_credit_check', 't_final_review', 
    't_identity_check', 't_monitor_eight', 't_monitor_five', 't_monitor_four', 't_monitor_nine', 
    't_monitor_one', 't_monitor_seven', 't_monitor_six', 't_monitor_three', 't_monitor_two', 
    't_plan_setup', 't_prepare_pack', 't_property_check', 't_reject_loan', 't_reserve_funds', 
    't_risk_assess', 't_send_notice', 't_send_offer', 't_send_quote', 't_update_risk', 
    't_verify_docs', 't_verify_terms', 't_monitor_ten'
]

def generate_embeddings(dim=32):
    embeddings = {}
    for label in labels:
        cleaned = re.sub(r"^t_", "", label.lower())
        words = [w for w in re.split(r"[_\s]+", cleaned) if w]
        vector = [0.0] * dim
        if not words:
            vector[0] = 1.0
        else:
            for word in words:
                for char_idx, char in enumerate(word):
                    key = f"{word}_{char}_{char_idx}".encode("utf-8")
                    h_idx = int(hashlib.md5(key).hexdigest(), 16) % dim
                    vector[h_idx] += 1.0
            
            # Normalize vector
            norm = sum(x**2 for x in vector)**0.5
            if norm > 0:
                vector = [x / norm for x in vector]
        embeddings[label] = vector
        
    os.makedirs("configs/semantics", exist_ok=True)
    with open("configs/semantics/loan_v1_v4_hash_bow_embeddings.json", "w", encoding="utf-8") as f:
        json.dump(embeddings, f, indent=2)
    print(f"Generated {len(embeddings)} semantic embeddings of dimension {dim}.")

if __name__ == "__main__":
    generate_embeddings(32)
```

- [ ] **Step 2: Execute generation script**

Run:
```powershell
.\.venv-modern\Scripts\python.exe tools/generate_semantic_embeddings.py
```
Expected: Prints `Generated 38 semantic embeddings of dimension 32.` and creates `configs/semantics/loan_v1_v4_hash_bow_embeddings.json`.

- [ ] **Step 3: Commit**
```bash
git add configs/semantics/loan_v1_v4_hash_bow_embeddings.json tools/generate_semantic_embeddings.py
git commit -m "feat: generate deterministic static semantic embedding dictionary"
```

---

### Task 5: Integrate Semantic Features in DynamicGraphBuilder

**Files:**
- Modify: `src/domain/services/dynamic_graph_builder.py:35-85`, `862-930`, `200-215`
- Modify: `src/cli.py:2065-2080`
- Create: `tests/domain/test_dynamic_graph_builder_semantic_features.py`

- [ ] **Step 1: Write unit test for graph builder semantic payloads**

Create `tests/domain/test_dynamic_graph_builder_semantic_features.py`:

```python
from __future__ import annotations
import pytest
import torch
from src.domain.entities.event_record import EventRecord
from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.entities.process_structure import ProcessStructureDTO
from src.domain.entities.raw_trace import RawTrace
from src.domain.services.dynamic_graph_builder import DynamicGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder
from src.infrastructure.repositories.in_memory_networkx_repository import InMemoryNetworkXRepository

def _event(idx: int, activity: str) -> EventRecord:
    return EventRecord(
        activity_id=activity,
        timestamp=float(1700100000 + idx),
        resource_id="R1",
        lifecycle="complete",
        position_in_trace=idx,
        duration=1.0,
        time_since_case_start=float(idx),
        time_since_previous_event=1.0 if idx > 0 else 0.0,
        extra={"concept:name": activity, "org:resource": "R1"},
        activity_instance_id=f"ai_{idx}_{activity}",
    )

def _trace(case_id: str, version: str, activities: list[str]) -> RawTrace:
    return RawTrace(
        case_id=case_id,
        process_version=version,
        events=[_event(i, act) for i, act in enumerate(activities)],
        trace_attributes={},
    )

def test_dynamic_graph_builder_attaches_semantic_embeddings_to_contract(mock_feature_configs):
    encoder = FeatureEncoder(
        feature_configs=mock_feature_configs,
        traces=[_trace("train", "v1", ["A", "B"])],
    )
    repository = InMemoryNetworkXRepository()
    repository.save_process_structure(
        "v2",
        ProcessStructureDTO(
            version="v2",
            allowed_edges=[("A", "C"), ("C", "B")],
            nodes=[
                {"id": "A", "bpmn_tag": "task", "type": "task", "label": "A"},
                {"id": "C", "bpmn_tag": "task", "type": "task", "label": "C"},
                {"id": "B", "bpmn_tag": "task", "type": "task", "label": "B"},
            ],
        ),
    )
    prefix = PrefixSlice(
        case_id="eval",
        process_version="v2",
        prefix_events=[_event(0, "A")],
        target_event=_event(1, "C"),
    )

    settings = {
        "model": {
            "semantic_grounding": {
                "enabled": True,
                "mode": "lexical_hash_bow",
                "embedding_dim": 16,
            }
        }
    }

    contract = DynamicGraphBuilder(
        feature_encoder=encoder,
        knowledge_port=repository,
        candidate_identity_mode="topology_native",
        semantic_grounding=settings["model"]["semantic_grounding"],
    ).build_graph(prefix)

    assert "struct_semantic_x" in contract
    assert isinstance(contract["struct_semantic_x"], torch.Tensor)
    assert contract["struct_semantic_x"].shape == (3, 16)
    
    # Check L2 normalization of candidate rows
    for row in range(3):
        norm = torch.norm(contract["struct_semantic_x"][row]).item()
        assert pytest.approx(norm, 1e-5) == 1.0
```

Run test:
```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests/domain/test_dynamic_graph_builder_semantic_features.py -v
```
Expected: Fail (TypeError on constructor or attribute errors).

- [ ] **Step 2: Modify DynamicGraphBuilder constructor and resolution**

In `src/domain/services/dynamic_graph_builder.py` add imports:
```python
from src.domain.services.semantic_embedding_provider import SemanticEmbeddingProvider
```

Modify `__init__` to accept `semantic_grounding`:
```diff
     def __init__(
         self,
         feature_encoder: FeatureEncoder,
         knowledge_port: IKnowledgeGraphPort,
         process_name: str | None = None,
         graph_feature_mapping: Optional[Dict[str, Any]] = None,
         stats_time_policy: str = "latest",
         stats_quality_gate: Optional[Dict[str, Any]] = None,
         on_missing_asof_snapshot: str = "disable_stats",
         cache_policy: str = "full",
         cache_dir: str | None = None,
         candidate_identity_mode: str = "fixed_vocab_bridge",
+        semantic_grounding: Optional[Dict[str, Any]] = None,
     ) -> None:
```

Initialize provider inside `__init__`:
```python
        self.semantic_grounding = dict(semantic_grounding or {})
        self.semantic_grounding_enabled = bool(self.semantic_grounding.get("enabled", False))
        self.semantic_provider = None
        if self.semantic_grounding_enabled:
            self.semantic_provider = SemanticEmbeddingProvider(
                mode=self.semantic_grounding.get("mode", "lexical_hash_bow"),
                embedding_dim=int(self.semantic_grounding.get("embedding_dim", 32)),
                dictionary_path=self.semantic_grounding.get("dictionary_path", None)
            )
```

In `_resolve_compiled_topology` (around lines 862-870):
```python
        candidate_class_index = torch.tensor(candidate_class_values, dtype=torch.long)
        candidate_is_unseen = candidate_class_index < 0

        struct_semantic_x = None
        if self.semantic_grounding_enabled and self.semantic_provider is not None:
            sem_list = []
            for label in candidate_labels:
                sem_list.append(self.semantic_provider.get_embedding(label))
            if sem_list:
                struct_semantic_x = torch.stack(sem_list, dim=0)
            else:
                struct_semantic_x = torch.zeros((0, self.semantic_provider.embedding_dim), dtype=torch.float32)
```

Add `struct_semantic_x` to the `compiled` dictionary (around line 907):
```python
        if struct_semantic_x is not None:
            compiled["struct_semantic_x"] = struct_semantic_x
```

In `build_graph` (around lines 200-205):
```python
        struct_semantic_x = compiled.get("struct_semantic_x")
        if isinstance(struct_semantic_x, torch.Tensor):
            contract["struct_semantic_x"] = struct_semantic_x
```

- [ ] **Step 3: Modify `src/cli.py` to route config to DynamicGraphBuilder**

Find the instantiation of `DynamicGraphBuilder` in `src/cli.py` (around line 2069) and pass semantic config:
```diff
     graph_builder = DynamicGraphBuilder(
         feature_encoder=feature_encoder,
         knowledge_repo=knowledge_repo,
         process_name=dataset_name,
         graph_feature_mapping=graph_feature_mapping,
         stats_time_policy=stats_time_policy,
         on_missing_asof_snapshot=on_missing_asof_snapshot,
         cache_policy=cache_policy,
         cache_dir=cache_dir,
         candidate_identity_mode=training_cfg.get("candidate_identity_mode", "fixed_vocab_bridge"),
+        semantic_grounding=config.get("model", {}).get("semantic_grounding", None),
     )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests/domain/test_dynamic_graph_builder_semantic_features.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/domain/services/dynamic_graph_builder.py src/cli.py tests/domain/test_dynamic_graph_builder_semantic_features.py
git commit -m "feat: extract and attach semantic embeddings in DynamicGraphBuilder"
```

---

### Task 6: Implement Semantic Projection in EOPKGTopologyConditioned

**Files:**
- Modify: `src/domain/models/topology_conditioned.py`
- Modify: `tests/domain/test_eopkg_topology_conditioned.py`

- [ ] **Step 1: Write unit test with backpropagation check**

Add model-level validation tests in `tests/domain/test_eopkg_topology_conditioned.py`:

```python
def test_topology_conditioned_with_semantic_grounding():
    layout = _layout()
    model = create_model(
        "EOPKGTopologyConditioned",
        feature_layout=layout,
        hidden_dim=8,
        output_dim=5,
        dropout=0.0,
        pooling_strategy="last_node",
        semantic_grounding_enabled=True,
        semantic_alpha=0.4,
        semantic_unseen_alpha=0.9,
        semantic_embedding_dim=16,
    )
    batch = _batch()
    batch["struct_semantic_x"] = torch.randn(5, 16) # 5 nodes, embedding_dim 16
    
    logits = model(batch)
    assert logits.shape == torch.Size([1, 5])
    
    # Verify forward_candidate also works with semantic grounding
    output = model.forward_candidate(batch)
    assert output.candidate_logits.shape == torch.Size([1, 5])

    # Verify gradients flow back to semantic_proj linear layer
    batch["struct_semantic_x"].requires_grad = False
    logits = model(batch)
    loss = logits.sum()
    loss.backward()
    assert model.semantic_proj.weight.grad is not None

def test_topology_conditioned_semantic_grounding_rejects_invalid_alpha():
    layout = _layout()
    with pytest.raises(ValueError, match="alpha must be in range"):
        create_model(
            "EOPKGTopologyConditioned",
            feature_layout=layout,
            hidden_dim=8,
            output_dim=5,
            semantic_grounding_enabled=True,
            semantic_alpha=1.5,
        )
```

Run test:
```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests/domain/test_eopkg_topology_conditioned.py -k "test_topology_conditioned_with_semantic_grounding" -v
```
Expected: FAIL.

- [ ] **Step 2: Update model constructor to validate alpha & initialize projection**

In `src/domain/models/topology_conditioned.py`, inside `__init__`:
```python
        # Validate ranges
        if self.semantic_grounding_enabled:
            if not (0.0 <= self.semantic_alpha <= 1.0):
                raise ValueError(f"model.semantic_grounding.alpha must be in range [0.0, 1.0]. Got {self.semantic_alpha}")
            if not (0.0 <= self.semantic_unseen_alpha <= 1.0):
                raise ValueError(f"model.semantic_grounding.unseen_alpha must be in range [0.0, 1.0]. Got {self.semantic_unseen_alpha}")
                
            self.semantic_proj = nn.Linear(self.semantic_embedding_dim, self.hidden_dim)
```

- [ ] **Step 3: Modify `_struct_input` to correctly isolate identity fusion**

Update `_struct_input` inside `src/domain/models/topology_conditioned.py` to isolate the identity fusion from `struct_x` metrics:

```python
    def _struct_input(self, contract: GraphTensorContract, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        node_to_class = contract.get("struct_node_to_class_index")
        if not isinstance(node_to_class, torch.Tensor):
            raise ValueError("EOPKGTopologyConditioned requires struct_node_to_class_index.")
        node_to_class = node_to_class.to(device=device, dtype=torch.long)
        identity = self._node_identity_features(node_to_class, device)

        # Fuse identity with semantic embedding first, before adding struct_x!
        struct_semantic_x = contract.get("struct_semantic_x")
        if self.semantic_grounding_enabled and isinstance(struct_semantic_x, torch.Tensor):
            struct_semantic_x = struct_semantic_x.to(device=device, dtype=torch.float32)
            if struct_semantic_x.size(0) != node_to_class.numel():
                raise ValueError("struct_semantic_x and struct_node_to_class_index must describe the same number of nodes.")
            if struct_semantic_x.size(1) != self.semantic_embedding_dim:
                raise ValueError(
                    f"struct_semantic_x features {struct_semantic_x.size(1)} != "
                    f"expected semantic_embedding_dim {self.semantic_embedding_dim}"
                )
            
            projected_semantic = self.semantic_proj(struct_semantic_x)
            
            # Apply adaptive alpha values for seen vs unseen candidates
            unseen_mask = (node_to_class < 0) | (node_to_class >= self.output_dim)
            alpha_tensor = torch.full_like(node_to_class, self.semantic_alpha, dtype=torch.float32).to(device)
            alpha_tensor[unseen_mask] = float(self.semantic_unseen_alpha)
            alpha_tensor = alpha_tensor.unsqueeze(1)  # Reshape to [nodes, 1] for broadcasting
            
            identity = (1.0 - alpha_tensor) * identity + alpha_tensor * projected_semantic

        h_base = identity
        struct_x = contract.get("struct_x")
        if isinstance(struct_x, torch.Tensor):
            struct_x = struct_x.to(device=device, dtype=torch.float32)
            if struct_x.size(0) != node_to_class.numel():
                raise ValueError("struct_x and struct_node_to_class_index must describe the same number of nodes.")
            h_base = h_base + self.struct_input_proj(struct_x)

        return self.struct_norm(h_base), node_to_class
```

- [ ] **Step 4: Run tests to verify they pass**

Run:
```powershell
.\.venv-modern\Scripts\python.exe -m pytest tests/domain/test_eopkg_topology_conditioned.py -v
```
Expected: PASS.

- [ ] **Step 5: Commit**
```bash
git add src/domain/models/topology_conditioned.py tests/domain/test_eopkg_topology_conditioned.py
git commit -m "feat: implement linear projection and adaptive fusion of semantic embeddings in EOPKGTopologyConditioned"
```

---

### Task 7: Bump Cache Schema and Run Full Pipeline Verification

**Files:**
- Modify: `src/cli.py:51`

- [ ] **Step 1: Bump `GRAPH_DATASET_CACHE_SCHEMA` in `src/cli.py`**

Open `src/cli.py` and modify `GRAPH_DATASET_CACHE_SCHEMA` to invalidate existing offline graph caches:
```diff
-GRAPH_DATASET_CACHE_SCHEMA = 4
+GRAPH_DATASET_CACHE_SCHEMA = 5
```

- [ ] **Step 2: Run all unit and regression tests**

Run verification suite:
```powershell
.\.venv-modern\Scripts\python.exe tools/architecture_guard.py
.\.venv-modern\Scripts\python.exe -m pytest -m mvp1_regression -v
.\.venv-modern\Scripts\python.exe -m pytest tests/ -v
```
Expected: All validation checks and tests PASS.

- [ ] **Step 3: Run a trial training command**

Execute a fast training run with semantic grounding enabled to ensure no CLI or trainer crashes:
```powershell
.\.venv-modern\Scripts\python.exe main.py --config configs/experiments/ui_run_co8n0zpp.yaml --model.semantic_grounding.enabled true --model.semantic_grounding.mode lexical_hash_bow --training.epochs 2 --experiment.fraction 0.02
```
Expected: Runs successfully, logs metrics, and finishes without errors.

- [ ] **Step 4: Commit**
```bash
git add src/cli.py
git commit -m "chore: bump cache schema to 5 and complete verification"
```

---

## Evaluation / Zero-Shot Ablation Criteria

To evaluate whether the semantic grounding feature has succeeded in directing meaningful zero-shot predictions onto unseen process version candidates:

1. **unseen_candidate_avg_rank**: The average rank of correct unseen targets in model logits. Without semantic grounding, this should decay to random or fixed structural ordering. With semantic grounding, it should improve.
2. **unseen_candidate_avg_logit**: Mean logit of correct unseen candidates. Check if the model starts to project higher logits onto unseen nodes.
3. **target_in_top3_for_unseen**: Rate at which the correct unseen target is positioned within top 3 prediction logits.

These metrics should be recorded and analyzed inside the select trace logs and MLflow experiment runs.

---

## Stage 2 / Future Directions: Process Context Grounding (MVP3)

Lexical tags only provide lexical similarity, which is a useful cold-start heuristic but does not fully capture the semantic role of an activity node inside a BPMN graph.

To implement a complete Process Context Grounding block ($x_{context}(v)$) in MVP3:

$$x_{context}(v) = [ x_{label}(v) \parallel x_{role}(v) \parallel x_{topology}(v) \parallel x_{neighborhood}(v) \parallel x_{statistics}(v) ]$$

1. **$x_{role}(v)$ (BPMN Node Role)**: One-hot representation of the BPMN node type (e.g., `userTask`, `serviceTask`, `gatewayXOR`, `intermediateThrowEvent`).
2. **$x_{topology}(v)$ (Local Structural Features)**: Local graph metrics (e.g., node degree, PageRank, betweenness centrality) generated directly from the BPMN topology.
3. **$x_{neighborhood}(v)$ (Local Neighborhood context)**: Contextual vectors aggregated from immediate predecessors and successors.
4. **$x_{statistics}(v)$ (Historical Execution Statistics)**: Temporal performance values (mean duration, execution counts) of the node activity if historic execution traces are available.

This contextual grounding will fully resolve activity semantic roles, paving the way for complete zero-shot process transition adaptation.
