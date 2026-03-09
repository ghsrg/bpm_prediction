from src.domain.services.feature_encoder import FeatureEncoder


def test_encoder_state_roundtrip_is_deterministic(mock_feature_configs, mock_raw_trace):
    encoder_a = FeatureEncoder(feature_configs=mock_feature_configs, traces=[mock_raw_trace])
    state = encoder_a.get_state()

    encoder_b = FeatureEncoder(feature_configs=mock_feature_configs, traces=None)
    encoder_b.load_state(state)

    event = {
        "concept:name": "End",
        "org:resource": "R1",
        "amount": 300.0,
    }
    encoded_a = encoder_a.encode_event(event_extra=event)
    encoded_b = encoder_b.encode_event(event_extra=event)

    assert encoded_a.cat_indices == encoded_b.cat_indices
    assert encoded_a.num_values == encoded_b.num_values

