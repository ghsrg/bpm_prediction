from src.domain.services.feature_encoder import FeatureEncoder
from src.domain.entities.feature_config import FeatureConfig
from src.domain.entities.prefix_slice import PrefixSlice
from src.domain.services.baseline_graph_builder import BaselineGraphBuilder


def test_unknown_categorical_value_maps_to_unk_index_zero(mock_feature_configs, mock_raw_trace):
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=[mock_raw_trace])

    encoded = encoder.encode_event(
        event_extra={
            "concept:name": "CompletelyNewActivity",
            "org:resource": "R1",
            "amount": 100.0,
        }
    )

    assert encoded.cat_indices[0] == 0


def test_missing_numeric_value_uses_mu_and_yields_zero_zscore(mock_feature_configs, mock_raw_trace):
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=[mock_raw_trace])

    encoded = encoder.encode_event(
        event_extra={
            "concept:name": "Start",
            "org:resource": "R1",
        }
    )

    assert encoded.num_values == [0.0]


def test_missing_categorical_key_is_imputed_as_unk(mock_feature_configs, mock_raw_trace):
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=[mock_raw_trace])
    activity_cfg = mock_feature_configs[0]

    raw_value = encoder._resolve_raw_value(event_extra={"org:resource": "R1"}, cfg=activity_cfg)

    assert raw_value == "<UNK>"


def test_activity_target_vocab_is_built_when_activity_is_not_input_feature(mock_raw_trace):
    feature_configs = [
        FeatureConfig(
            name="org:resource",
            source_key=None,
            source="event",
            dtype="string",
            fill_na="<UNK>",
            encoding=["embedding"],
            role="resource",
        ),
    ]

    encoder = FeatureEncoder(
        feature_configs=feature_configs,
        traces=[mock_raw_trace],
        policy_config={"activity_fallback_feature": "concept:name"},
    )

    assert "concept:name" in encoder.categorical_vocabs
    assert encoder.categorical_vocabs["concept:name"]["Start"] > 0
    assert encoder.categorical_vocabs["concept:name"]["Approve"] > 0
    assert encoder.feature_layout.cat_feature_names == ["org:resource"]


def test_baseline_graph_target_uses_activity_vocab_without_activity_input_feature(mock_raw_trace):
    feature_configs = [
        FeatureConfig(
            name="org:resource",
            source_key=None,
            source="event",
            dtype="string",
            fill_na="<UNK>",
            encoding=["embedding"],
            role="resource",
        ),
    ]
    encoder = FeatureEncoder(
        feature_configs=feature_configs,
        traces=[mock_raw_trace],
        policy_config={"activity_fallback_feature": "concept:name"},
    )
    builder = BaselineGraphBuilder(feature_encoder=encoder)
    prefix = PrefixSlice(
        case_id=mock_raw_trace.case_id,
        process_version=mock_raw_trace.process_version,
        prefix_events=[mock_raw_trace.events[0]],
        target_event=mock_raw_trace.events[1],
    )

    graph = builder.build_graph(prefix)

    assert int(graph["y"].item()) == encoder.categorical_vocabs["concept:name"]["Approve"]
    assert int(graph["y"].item()) != 0
