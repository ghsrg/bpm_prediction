"""Feature configuration DTOs parsed from YAML mapping.features."""

# Відповідно до:
# - AGENT_GUIDE.MD -> розділ 4 (DTO discipline) та розділ 5 (config-driven implementation)
# - DATA_FLOWS_MVP1.MD -> Encoder contract (feature representation is explicit)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence


FeatureSource = Literal["event", "trace"]
FeatureDType = Literal["string", "float", "int", "boolean", "timestamp"]


@dataclass(frozen=True)
class FeatureConfig:
    """Single feature specification from YAML mapping.features."""

    name: str
    source: FeatureSource
    dtype: FeatureDType
    fill_na: Any
    encoding: List[str]
    role: Optional[str] = None


@dataclass(frozen=True)
class FeatureLayout:
    """Layout contract passed into model for deterministic feature consumption."""

    cat_features: Dict[str, int]
    cat_feature_names: List[str]
    num_dim: int


def parse_feature_configs(mapping_config: Dict[str, Any]) -> List[FeatureConfig]:
    """Parse mapping.features list into validated FeatureConfig DTOs."""
    mapping = mapping_config.get("mapping", mapping_config)
    raw_features = mapping.get("features", [])
    configs: List[FeatureConfig] = []

    if not isinstance(raw_features, Sequence) or isinstance(raw_features, (str, bytes)):
        return configs

    for raw in raw_features:
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", "")).strip()
        source = str(raw.get("source", "event")).strip().lower()
        dtype = str(raw.get("dtype", "string")).strip().lower()
        encoding_raw = raw.get("encoding", [])
        role_raw = raw.get("role")

        if not name or source not in {"event", "trace"}:
            continue
        if dtype not in {"string", "float", "int", "boolean", "timestamp"}:
            continue

        if isinstance(encoding_raw, Sequence) and not isinstance(encoding_raw, (str, bytes)):
            encoding = [str(item).strip() for item in encoding_raw if str(item).strip()]
        else:
            encoding = []

        configs.append(
            FeatureConfig(
                name=name,
                source=source,  # type: ignore[arg-type]
                dtype=dtype,  # type: ignore[arg-type]
                fill_na=raw.get("fill_na"),
                encoding=encoding,
                role=str(role_raw).strip().lower() if role_raw is not None else None,
            )
        )

    return configs
