import pytest

from src.domain.entities.feature_config import FeatureConfig
from src.domain.services.schema_resolver import SchemaResolver


def _cfg(source_key: str | None) -> FeatureConfig:
    return FeatureConfig(
        name="concept:name",
        source_key=source_key,
        source="event",
        dtype="string",
        fill_na="<UNK>",
        encoding=["embedding"],
        role="activity",
    )


@pytest.mark.parametrize(
    ("source_key", "fallback_keys", "expected"),
    [
        ("activity", (), ["concept:name", "activity"]),
        (None, (), ["concept:name"]),
        ("activity", ("concept:name", "legacy_activity"), ["concept:name", "activity", "legacy_activity"]),
    ],
)
def test_resolve_keys_order_name_source_fallback(source_key, fallback_keys, expected):
    resolver = SchemaResolver(fallback_keys=fallback_keys)
    assert resolver.resolve_keys(_cfg(source_key)) == expected


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"concept:name": "Start", "activity": "IgnoredAlias"}, "Start"),
        ({"activity": "Approve"}, "Approve"),
    ],
)
def test_resolve_from_mapping_uses_name_then_source_key(payload, expected):
    resolver = SchemaResolver()
    assert resolver.resolve_from_mapping(_cfg("activity"), payload, default=None) == expected


def test_resolve_from_mapping_returns_default_when_no_keys():
    resolver = SchemaResolver(fallback_keys=("legacy",))
    assert resolver.resolve_from_mapping(_cfg("activity"), payload={"other": 1}, default="N/A") == "N/A"


def test_resolve_value_identity_returns_same_object():
    resolver = SchemaResolver()
    cfg = _cfg("activity")
    raw = {"k": ["v"]}
    assert resolver.resolve_value(cfg, raw) is raw

