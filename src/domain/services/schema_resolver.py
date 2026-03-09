"""Canonical schema resolver for feature key aliases and future semantic mapping."""

# Відповідно до:
# - ARCHITECTURE_RULES.md -> розділ 2-3 (Domain service без інфраструктурних залежностей)
# - DATA_FLOWS_MVP1.MD -> розділ 2.1/2.3 (узгоджений mapping між ingestion та encoder)
# - TARGET_ARCHITECTURE.MD -> секція data contracts (єдина канонізація перед розширенням MVP2)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional

from src.domain.entities.feature_config import FeatureConfig


_MISSING = object()


@dataclass(frozen=True)
class SchemaResolver:
    """Resolve schema aliases consistently across adapters and domain encoders.

    MVP1 behavior is intentionally passive: key resolution only + identity value pass-through.
    Extension point: in MVP2 `resolve_value` will be delegated to a SemanticMapper
    for language/synonym normalization (e.g. "Старт" -> "Start") without changing
    adapter/encoder call sites.
    """

    fallback_keys: tuple[str, ...] = ()

    def resolve_keys(self, cfg: FeatureConfig) -> list[str]:
        """Return ordered candidate keys: canonical name, source alias, then configured fallbacks."""
        ordered: list[str] = [cfg.name]
        if cfg.source_key and cfg.source_key not in ordered:
            ordered.append(cfg.source_key)
        for key in self.fallback_keys:
            if key and key not in ordered:
                ordered.append(key)
        return ordered

    def resolve_from_mapping(self, cfg: FeatureConfig, payload: Mapping[str, Any], default: Any = None) -> Any:
        """Resolve value from mapping using unified key precedence."""
        for key in self.resolve_keys(cfg):
            if key in payload:
                return self.resolve_value(cfg, payload[key])
        return default

    def resolve_value(self, cfg: FeatureConfig, raw_value: Any) -> Any:
        """MVP1 identity transform; reserved extension point for MVP2 SemanticMapper."""
        _ = cfg
        return raw_value

    def first_existing_key(self, cfg: FeatureConfig, payload: Mapping[str, Any]) -> Optional[str]:
        """Return first matching key in payload, if any."""
        for key in self.resolve_keys(cfg):
            if key in payload:
                return key
        return None

    def normalize_fallbacks(self, keys: Iterable[str]) -> "SchemaResolver":
        """Return a copy with normalized fallback keys for explicit profiles."""
        normalized = tuple(str(key).strip() for key in keys if str(key).strip())
        return SchemaResolver(fallback_keys=normalized)
