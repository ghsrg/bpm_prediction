from src.domain.services.feature_encoder import FeatureEncoder


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

