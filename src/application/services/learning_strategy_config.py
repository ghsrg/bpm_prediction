"""Learning strategy config parsing for trainer-level topology conditioning."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping


class FusionSupportLevel(str, Enum):
    """Support level of a fusion mode for topology-conditioned training."""

    SUPPORTED = "supported"
    ALLOWED_BUT_WEAK = "allowed_but_weak"
    FORBIDDEN = "forbidden"


_SUPPORTED_FUSION_MODES = {
    "struct_xattn",
    "class_aware_structural_scoring",
    "topology_state_encoder",
    "topology_state_graph_encoder",
    "topology_conditioned_candidate_scoring",
}
_ALLOWED_BUT_WEAK_FUSION_MODES = {
    "class_mean_attention",
    "class_mean_concat",
    "structural_prior_encoder",
}
_FUSION_ALIASES = {
    "structxattn": "struct_xattn",
    "classawarestructuralscoring": "class_aware_structural_scoring",
    "topologystateencoder": "topology_state_encoder",
    "topologystategraphencoder": "topology_state_graph_encoder",
    "classmeanattention": "class_mean_attention",
    "attention": "class_mean_attention",
    "classmeanconcat": "class_mean_concat",
    "concat": "class_mean_concat",
    "struct_pool_concat": "class_mean_concat",
    "structuralpriorencoder": "structural_prior_encoder",
    "topologyconditionedcandidatescoring": "topology_conditioned_candidate_scoring",
    "eopkgtopologyconditioned": "topology_conditioned_candidate_scoring",
}


def _as_bool(raw: Any, *, default: bool = False) -> bool:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return bool(default)
    text = str(raw).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def normalize_fusion_mode(raw: Any) -> str:
    text = str(raw or "").strip()
    normalized = text.replace("-", "_").strip().lower()
    return _FUSION_ALIASES.get(normalized, normalized)


def fusion_support_level(raw_fusion_mode: Any, *, structural_mode: bool = True) -> FusionSupportLevel:
    if not structural_mode:
        return FusionSupportLevel.FORBIDDEN
    fusion_mode = normalize_fusion_mode(raw_fusion_mode)
    if fusion_mode in _SUPPORTED_FUSION_MODES:
        return FusionSupportLevel.SUPPORTED
    if fusion_mode in _ALLOWED_BUT_WEAK_FUSION_MODES:
        return FusionSupportLevel.ALLOWED_BUT_WEAK
    return FusionSupportLevel.FORBIDDEN


@dataclass(frozen=True)
class LearningStrategyConfig:
    """Flat trainer config for learning strategy behavior."""

    learning_strategy: str = "standard"
    fusion_support_level: FusionSupportLevel = FusionSupportLevel.FORBIDDEN
    wrong_version_negative_enabled: bool = False
    wrong_version_negative_weight: float = 0.10
    wrong_version_margin: float = 0.20
    drop_edges_negative_enabled: bool = False
    drop_edges_negative_weight: float = 0.05
    drop_edges_margin: float = 0.10
    drop_edges_ratio: float = 0.15
    allowed_set_loss_enabled: bool = False
    allowed_set_loss_weight: float = 0.05
    retention_enabled: bool = False
    retention_policy: str = "version_decay"
    current_version_policy: str = "latest_in_train"
    retention_recent_versions: int = 1
    retention_recent_weight: float = 0.50
    retention_old_weight: float = 0.15
    retention_obsolete_weight: float = 0.00

    @property
    def is_topology_conditioned(self) -> bool:
        return self.learning_strategy == "topology_conditioned"

    @classmethod
    def from_training_config(
        cls,
        config: Mapping[str, Any],
        *,
        fusion_mode: Any = "",
        structural_mode: bool = True,
    ) -> "LearningStrategyConfig":
        strategy = str(config.get("learning_strategy", "standard")).strip().lower()
        if strategy not in {"standard", "topology_conditioned"}:
            raise ValueError(
                "Unsupported training.learning_strategy "
                f"'{strategy}'. Available: ['standard', 'topology_conditioned']."
            )
        support_level = fusion_support_level(fusion_mode, structural_mode=bool(structural_mode))
        if strategy == "standard":
            return cls(learning_strategy="standard", fusion_support_level=support_level)
        if not structural_mode:
            raise ValueError("training.learning_strategy=topology_conditioned requires structural_mode=true.")
        if _as_bool(config.get("struct_xattn_contrastive_enabled"), default=False):
            raise ValueError(
                "training.learning_strategy=topology_conditioned cannot be combined with "
                "training.struct_xattn_contrastive_enabled."
            )
        if support_level == FusionSupportLevel.FORBIDDEN:
            raise ValueError(
                "training.learning_strategy=topology_conditioned requires a structural fusion mode. "
                f"Got fusion_mode='{fusion_mode}'."
            )
        return cls(
            learning_strategy="topology_conditioned",
            fusion_support_level=support_level,
            wrong_version_negative_enabled=_as_bool(
                config.get("topology_conditioning_wrong_version_negative_enabled"),
                default=True,
            ),
            wrong_version_negative_weight=float(
                config.get("topology_conditioning_wrong_version_negative_weight", 0.10)
            ),
            wrong_version_margin=float(config.get("topology_conditioning_wrong_version_margin", 0.20)),
            drop_edges_negative_enabled=_as_bool(
                config.get("topology_conditioning_drop_edges_negative_enabled"),
                default=True,
            ),
            drop_edges_negative_weight=float(config.get("topology_conditioning_drop_edges_negative_weight", 0.05)),
            drop_edges_margin=float(config.get("topology_conditioning_drop_edges_margin", 0.10)),
            drop_edges_ratio=min(
                1.0,
                max(0.0, float(config.get("topology_conditioning_drop_edges_ratio", 0.15))),
            ),
            allowed_set_loss_enabled=_as_bool(
                config.get("topology_conditioning_allowed_set_loss_enabled"),
                default=True,
            ),
            allowed_set_loss_weight=float(config.get("topology_conditioning_allowed_set_loss_weight", 0.05)),
            retention_enabled=_as_bool(config.get("topology_conditioning_retention_enabled"), default=True),
            retention_policy=str(config.get("topology_conditioning_retention_policy", "version_decay")).strip().lower(),
            current_version_policy=str(
                config.get("topology_conditioning_current_version_policy", "latest_in_train")
            ).strip().lower(),
            retention_recent_versions=max(
                0,
                int(config.get("topology_conditioning_retention_recent_versions", 1)),
            ),
            retention_recent_weight=float(config.get("topology_conditioning_retention_recent_weight", 0.50)),
            retention_old_weight=float(config.get("topology_conditioning_retention_old_weight", 0.15)),
            retention_obsolete_weight=float(config.get("topology_conditioning_retention_obsolete_weight", 0.00)),
        )
