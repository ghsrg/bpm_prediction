"""Domain feature encoder for mixed categorical embedding indices and numeric features."""

# Відповідно до:
# - AGENT_GUIDE.MD -> розділ 2 (MVP1 baseline), розділ 4 (strict DTO usage)
# - DATA_FLOWS_MVP1.MD -> розділ 2.3 (FeatureEncoder responsibility)
# - LLD_MVP1.MD -> розділ 4 (feature engineering)

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from math import cos, pi, sin
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.domain.entities.feature_config import FeatureConfig, FeatureLayout
from src.domain.entities.raw_trace import RawTrace


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EncodedNodeFeatures:
    """Encoded per-node feature split for model consumption."""

    cat_indices: List[int]
    num_values: List[float]


class FeatureEncoder:
    """Encodes node features according to mapping.features configuration."""

    def __init__(
        self,
        feature_configs: Sequence[FeatureConfig],
        traces: Optional[Sequence[RawTrace]] = None,
        state_dict: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.feature_configs = list(feature_configs)
        self.role_to_feature = {cfg.role: cfg.name for cfg in self.feature_configs if cfg.role}
        self.activity_feature_name = self._resolve_activity_feature_name()

        self.cat_feature_configs: List[FeatureConfig] = [
            cfg for cfg in self.feature_configs if "embedding" in cfg.encoding
        ]
        self.num_feature_configs: List[FeatureConfig] = [
            cfg for cfg in self.feature_configs if any(enc != "embedding" for enc in cfg.encoding)
        ]

        source_traces = [] if state_dict is not None else list(traces or [])
        self.categorical_vocabs: Dict[str, Dict[str, int]] = self._build_categorical_vocabs(source_traces)
        self.numeric_stats: Dict[str, Dict[str, float]] = self._build_numeric_stats(source_traces)
        self._imputation_warnings_emitted: set[str] = set()
        if state_dict is not None:
            self.load_state(state_dict)
        self.feature_layout = self._build_feature_layout()

    def get_state(self) -> Dict[str, Any]:
        """Export fitted categorical vocabularies and numeric scaler statistics."""
        return {
            "categorical_vocabs": {
                key: {str(token): int(index) for token, index in vocab.items()}
                for key, vocab in self.categorical_vocabs.items()
            },
            "numerical_scalers": {
                key: {"mu": float(stats.get("mu", 0.0)), "sigma": float(stats.get("sigma", 1.0))}
                for key, stats in self.numeric_stats.items()
            },
        }

    def load_state(self, state_dict: Dict[str, Any]) -> None:
        """Load categorical vocabularies and numeric scaler statistics from checkpoint state."""
        if not isinstance(state_dict, dict):
            raise ValueError("FeatureEncoder state_dict must be a dictionary.")

        raw_vocabs = state_dict.get("categorical_vocabs", {})
        if not isinstance(raw_vocabs, dict):
            raise ValueError("FeatureEncoder state key 'categorical_vocabs' must be a dictionary.")

        restored_vocabs: Dict[str, Dict[str, int]] = {}
        for cfg in self.cat_feature_configs:
            if cfg.name not in raw_vocabs:
                raise ValueError(f"Критична помилка: Фіча {cfg.name} відсутня у збереженому стані енкодера.")
            vocab_payload = raw_vocabs[cfg.name]
            if not isinstance(vocab_payload, dict):
                raise ValueError(f"FeatureEncoder vocab for '{cfg.name}' must be a dictionary.")
            restored_vocab = {str(token): int(index) for token, index in vocab_payload.items()}
            if "<UNK>" not in restored_vocab:
                restored_vocab["<UNK>"] = 0
            restored_vocabs[cfg.name] = restored_vocab
        self.categorical_vocabs = restored_vocabs

        raw_scalers = state_dict.get("numerical_scalers", state_dict.get("numeric_stats", {}))
        if not isinstance(raw_scalers, dict):
            raise ValueError("FeatureEncoder state key 'numerical_scalers' must be a dictionary.")
        restored_stats: Dict[str, Dict[str, float]] = {}
        for cfg in self.num_feature_configs:
            if "z-score" not in cfg.encoding:
                continue
            if cfg.name not in raw_scalers:
                raise ValueError(f"Критична помилка: Фіча {cfg.name} відсутня у збереженому стані енкодера.")
            scaler = raw_scalers[cfg.name]
            if not isinstance(scaler, dict):
                raise ValueError(f"FeatureEncoder scaler for '{cfg.name}' must be a dictionary.")
            sigma = float(scaler.get("sigma", 1.0))
            restored_stats[cfg.name] = {
                "mu": float(scaler.get("mu", 0.0)),
                "sigma": sigma if sigma > 1e-8 else 1.0,
            }
        self.numeric_stats = restored_stats
        self.feature_layout = self._build_feature_layout()

    def _resolve_activity_feature_name(self) -> str:
        """Resolve feature name used as target label vocabulary."""
        for cfg in self.feature_configs:
            if cfg.role == "activity":
                return cfg.name
        return "concept:name"

    def _build_feature_layout(self) -> FeatureLayout:
        """Build deterministic layout with categorical vocab sizes and numeric dimension."""
        cat_features = {
            cfg.name: len(self.categorical_vocabs.get(cfg.name, {"<UNK>": 0}))
            for cfg in self.cat_feature_configs
        }
        num_dim = 0
        for cfg in self.num_feature_configs:
            for enc in cfg.encoding:
                if enc == "embedding":
                    continue
                if enc.startswith("time2vec_"):
                    if enc == "time2vec_work_hours":
                        num_dim += 1
                    else:
                        num_dim += 2
                else:
                    num_dim += 1

        return FeatureLayout(
            cat_features=cat_features,
            cat_feature_names=[cfg.name for cfg in self.cat_feature_configs],
            num_dim=num_dim,
        )

    def _build_categorical_vocabs(self, traces: Sequence[RawTrace]) -> Dict[str, Dict[str, int]]:
        """Build vocabularies for embedding-encoded features with <UNK>=0."""
        vocabs: Dict[str, Dict[str, int]] = {cfg.name: {"<UNK>": 0} for cfg in self.cat_feature_configs}

        for trace in traces:
            for cfg in self.cat_feature_configs:
                values = self._iter_feature_values(trace, cfg)
                vocab = vocabs[cfg.name]
                for value in values:
                    token = str(value)
                    if token not in vocab:
                        vocab[token] = len(vocab)

        return vocabs

    def _build_numeric_stats(self, traces: Sequence[RawTrace]) -> Dict[str, Dict[str, float]]:
        """Build z-score stats for all features that request z-score encoding."""
        values_by_key: Dict[str, List[float]] = {}

        for cfg in self.num_feature_configs:
            if "z-score" not in cfg.encoding:
                continue
            values_by_key[cfg.name] = []

        for trace in traces:
            for cfg in self.num_feature_configs:
                if "z-score" not in cfg.encoding:
                    continue
                for value in self._iter_feature_values(trace, cfg):
                    as_num = self._to_numeric(value, cfg.dtype)
                    values_by_key[cfg.name].append(as_num)

        stats: Dict[str, Dict[str, float]] = {}
        for key, values in values_by_key.items():
            if not values:
                stats[key] = {"mu": 0.0, "sigma": 1.0}
                continue
            mu = float(sum(values) / len(values))
            var = float(sum((val - mu) ** 2 for val in values) / len(values))
            sigma = var ** 0.5
            stats[key] = {"mu": mu, "sigma": sigma if sigma > 1e-8 else 1.0}
        return stats

    def _iter_feature_values(self, trace: RawTrace, cfg: FeatureConfig) -> List[Any]:
        """Iterate values from a trace for a specific feature config."""
        if cfg.source == "trace":
            return [trace.trace_attributes.get(cfg.name, cfg.fill_na)]
        return [event.extra.get(cfg.name, cfg.fill_na) for event in trace.events]

    def encode_event(self, *, event_extra: Dict[str, Any]) -> EncodedNodeFeatures:
        """Encode one event payload into split categorical indices and numeric values."""
        cat_indices: List[int] = []
        num_values: List[float] = []

        for cfg in self.cat_feature_configs:
            raw = self._resolve_raw_value(event_extra=event_extra, cfg=cfg)
            token = str(raw)
            vocab = self.categorical_vocabs[cfg.name]
            cat_indices.append(vocab.get(token, 0))

        for cfg in self.num_feature_configs:
            raw = self._resolve_raw_value(event_extra=event_extra, cfg=cfg)
            for enc in cfg.encoding:
                if enc == "embedding":
                    continue
                if enc == "z-score":
                    z = self._zscore(raw, cfg)
                    num_values.append(max(-3.0, min(3.0, z)))
                elif enc.startswith("time2vec_"):
                    num_values.extend(self._encode_time2vec(raw, enc))

        return EncodedNodeFeatures(cat_indices=cat_indices, num_values=num_values)

    def _resolve_raw_value(self, event_extra: Dict[str, Any], cfg: FeatureConfig) -> Any:
        """Resolve raw feature value with schema-alignment imputations for missing keys."""
        if cfg.name in event_extra:
            return event_extra[cfg.name]

        if cfg.name not in self._imputation_warnings_emitted:
            logger.warning("Feature %s missing in data. Imputing with defaults.", cfg.name)
            self._imputation_warnings_emitted.add(cfg.name)

        if "embedding" in cfg.encoding:
            return "<UNK>"

        if "z-score" in cfg.encoding:
            stats = self.numeric_stats.get(cfg.name, {"mu": 0.0})
            return float(stats.get("mu", 0.0))

        return cfg.fill_na

    def _zscore(self, value: Any, cfg: FeatureConfig) -> float:
        """Apply z-score transformation with precomputed stats."""
        stats = self.numeric_stats.get(cfg.name, {"mu": 0.0, "sigma": 1.0})
        as_num = self._to_numeric(value, cfg.dtype)
        return (as_num - float(stats["mu"])) / float(stats["sigma"])

    def _to_numeric(self, value: Any, dtype: str) -> float:
        """Convert typed value to float for numeric processing."""
        if dtype == "boolean":
            return 1.0 if bool(value) else 0.0
        if dtype == "timestamp":
            return self._to_timestamp_epoch(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def _to_timestamp_epoch(self, value: Any) -> float:
        """Convert timestamp raw value to epoch seconds float."""
        if isinstance(value, (int, float)):
            return float(value)
        text = "" if value is None else str(value).strip()
        if not text:
            return 0.0
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            return 0.0
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return float(dt.timestamp())

    def _encode_time2vec(self, value: Any, mode: str) -> List[float]:
        """Encode timestamp into cyclic time channels (sin/cos pairs for cyclic modes)."""
        epoch = self._to_timestamp_epoch(value)
        dt = datetime.fromtimestamp(epoch, tz=timezone.utc)

        if mode == "time2vec_dayofweek":
            return self._cyclic_pair(dt.weekday(), 7.0)
        if mode == "time2vec_month":
            return self._cyclic_pair(dt.month - 1, 12.0)
        if mode == "time2vec_dayofmonth":
            return self._cyclic_pair(dt.day - 1, 31.0)
        if mode == "time2vec_weekofyear":
            return self._cyclic_pair(int(dt.isocalendar().week) - 1, 53.0)
        if mode == "time2vec_hour":
            return self._cyclic_pair(dt.hour, 24.0)
        if mode == "time2vec_work_hours":
            # Work-hours indicator (Mon-Fri, 09:00-18:00 UTC) as non-cyclic context channel.
            is_work_hour = 1.0 if dt.weekday() < 5 and 9 <= dt.hour < 18 else 0.0
            return [is_work_hour]
        return []

    def _cyclic_pair(self, value: float, period: float) -> List[float]:
        """Return sin/cos pair for one cyclic feature value."""
        angle = 2.0 * pi * (float(value) / float(period))
        return [float(sin(angle)), float(cos(angle))]
