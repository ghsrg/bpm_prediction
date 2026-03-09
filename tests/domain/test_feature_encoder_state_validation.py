import pytest

from src.domain.services.feature_encoder import FeatureEncoder


def test_load_state_raises_when_categorical_vocab_is_missing(mock_feature_configs, mock_raw_trace):
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=[mock_raw_trace])
    state = encoder.get_state()
    state["categorical_vocabs"].pop("org:resource")

    with pytest.raises(ValueError):
        encoder.load_state(state)


def test_load_state_raises_when_numeric_scaler_is_missing(mock_feature_configs, mock_raw_trace):
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=[mock_raw_trace])
    state = encoder.get_state()
    state["numerical_scalers"].pop("cost")

    with pytest.raises(ValueError):
        encoder.load_state(state)

