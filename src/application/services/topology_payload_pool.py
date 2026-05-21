"""Structural payload collection and replacement for topology-conditioned learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping

import torch

from src.domain.entities.tensor_contract import GraphTensorContract


@dataclass(frozen=True)
class TopologyPayload:
    version_label: str
    struct_x: torch.Tensor | None
    structural_edge_index: torch.Tensor
    structural_edge_weight: torch.Tensor | None
    struct_node_to_class_index: torch.Tensor | None


def _clone_optional_tensor(raw: Any) -> torch.Tensor | None:
    return raw.detach().clone() if isinstance(raw, torch.Tensor) else None


def _version_label(graph: Any, idx_to_version: Mapping[int, str]) -> str | None:
    raw = getattr(graph, "process_version_idx", None)
    if isinstance(raw, torch.Tensor) and raw.numel() > 0:
        idx = int(raw.view(-1)[0].item())
        return str(idx_to_version.get(idx, f"v{idx}"))
    raw_label = getattr(graph, "process_version", None)
    if raw_label is not None:
        label = str(raw_label).strip()
        return label or None
    return None


@dataclass(frozen=True)
class TopologyPayloadPool:
    payloads_by_version: Dict[str, TopologyPayload]

    @property
    def version_labels(self) -> list[str]:
        return sorted(self.payloads_by_version)

    @classmethod
    def from_dataset(
        cls,
        dataset: Iterable[Any],
        *,
        idx_to_version: Mapping[int, str] | None = None,
    ) -> "TopologyPayloadPool":
        versions = dict(idx_to_version or {})
        payloads: Dict[str, TopologyPayload] = {}
        for graph in dataset:
            label = _version_label(graph, versions)
            edge_index = getattr(graph, "structural_edge_index", None)
            if label is None or not isinstance(edge_index, torch.Tensor) or edge_index.numel() == 0:
                continue
            if label in payloads:
                continue
            payloads[label] = TopologyPayload(
                version_label=label,
                struct_x=_clone_optional_tensor(getattr(graph, "struct_x", None)),
                structural_edge_index=edge_index.detach().clone(),
                structural_edge_weight=_clone_optional_tensor(getattr(graph, "structural_edge_weight", None)),
                struct_node_to_class_index=_clone_optional_tensor(getattr(graph, "struct_node_to_class_index", None)),
            )
        return cls(payloads_by_version=payloads)

    def wrong_version_payload(self, current_version: str) -> TopologyPayload | None:
        for label in self.version_labels:
            if str(label) != str(current_version):
                return self.payloads_by_version[label]
        return None


def _contract_device(contract: Mapping[str, Any]) -> torch.device:
    for value in contract.values():
        if isinstance(value, torch.Tensor):
            return value.device
    return torch.device("cpu")


def replace_structural_payload(
    contract: Mapping[str, Any],
    payload: TopologyPayload | None,
) -> GraphTensorContract:
    replaced: GraphTensorContract = dict(contract)  # type: ignore[assignment]
    if payload is None:
        return replaced
    device = _contract_device(contract)
    if isinstance(payload.struct_x, torch.Tensor):
        replaced["struct_x"] = payload.struct_x.to(device=device).clone()
    replaced["structural_edge_index"] = payload.structural_edge_index.to(device=device).clone()
    if isinstance(payload.structural_edge_weight, torch.Tensor):
        replaced["structural_edge_weight"] = payload.structural_edge_weight.to(device=device).clone()
    elif "structural_edge_weight" in replaced:
        replaced.pop("structural_edge_weight", None)
    if isinstance(payload.struct_node_to_class_index, torch.Tensor):
        replaced["struct_node_to_class_index"] = payload.struct_node_to_class_index.to(device=device).clone()
    return replaced


def drop_structural_edges(
    contract: Mapping[str, Any],
    *,
    drop_ratio: float,
    seed: int,
) -> GraphTensorContract:
    corrupted: GraphTensorContract = dict(contract)  # type: ignore[assignment]
    edge_index = corrupted.get("structural_edge_index")
    if not isinstance(edge_index, torch.Tensor) or edge_index.dim() != 2 or int(edge_index.size(1)) <= 1:
        return corrupted
    edge_count = int(edge_index.size(1))
    keep_prob = 1.0 - min(1.0, max(0.0, float(drop_ratio)))
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    keep_mask = torch.rand(edge_count, generator=generator, device="cpu") < keep_prob
    if not torch.any(keep_mask):
        keep_mask[0] = True
    keep_mask = keep_mask.to(device=edge_index.device, dtype=torch.bool)
    corrupted["structural_edge_index"] = edge_index[:, keep_mask].clone()
    edge_weight = corrupted.get("structural_edge_weight")
    if isinstance(edge_weight, torch.Tensor) and int(edge_weight.numel()) == edge_count:
        corrupted["structural_edge_weight"] = edge_weight[keep_mask].clone()
    return corrupted
