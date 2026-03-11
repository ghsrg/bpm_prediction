from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest
import torch
from torch.optim import Adam

from src.application.use_cases.trainer import ModelTrainer
from src.domain.entities.raw_trace import RawTrace
from src.domain.models.baseline_gcn import BaselineGCN
from src.domain.services.baseline_graph_builder import BaselineGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder
from src.domain.services.schema_resolver import SchemaResolver


class _DummyAdapter:
    def read(self, file_path: str, mapping_config: dict) -> Iterator[RawTrace]:
        _ = file_path
        _ = mapping_config
        return iter([])


class _DummyPrefixPolicy:
    def generate_slices(self, trace: RawTrace):
        _ = trace
        return []


def _build_trainer(checkpoint_path: Path, *, model: BaselineGCN, graph_builder: BaselineGraphBuilder) -> ModelTrainer:
    return ModelTrainer(
        xes_adapter=_DummyAdapter(),
        prefix_policy=_DummyPrefixPolicy(),
        graph_builder=graph_builder,
        model=model,
        log_path="in_memory.xes",
        config={
            "mode": "train",
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "checkpoint_path": str(checkpoint_path),
            "experiment_config": {"name": "mvp1_ckpt_test"},
        },
    )


@pytest.mark.mvp1_regression
def test_checkpoint_roundtrip_with_encoder_state(mock_feature_configs, mock_raw_trace, tmp_path):
    encoder = FeatureEncoder(
        feature_configs=mock_feature_configs,
        traces=[mock_raw_trace],
        schema_resolver=SchemaResolver(),
    )
    graph_builder = BaselineGraphBuilder(feature_encoder=encoder)
    model = BaselineGCN(feature_layout=encoder.feature_layout, hidden_dim=8, output_dim=3, dropout=0.0)
    trainer = _build_trainer(tmp_path / "mvp1_best.pth", model=model, graph_builder=graph_builder)

    optimizer = Adam(model.parameters(), lr=1e-3)
    trainer._save_checkpoint(Path(trainer.checkpoint_path), epoch=1, val_loss=0.123, optimizer=optimizer)

    checkpoint = trainer._load_checkpoint(Path(trainer.checkpoint_path), require_encoder_state=True)
    assert "encoder_state" in checkpoint

    new_encoder = FeatureEncoder(
        feature_configs=mock_feature_configs,
        traces=[mock_raw_trace],
        schema_resolver=SchemaResolver(),
    )
    new_graph_builder = BaselineGraphBuilder(feature_encoder=new_encoder)
    new_model = BaselineGCN(feature_layout=new_encoder.feature_layout, hidden_dim=8, output_dim=3, dropout=0.0)
    new_trainer = _build_trainer(tmp_path / "mvp1_best_restore.pth", model=new_model, graph_builder=new_graph_builder)

    new_trainer._restore_from_checkpoint(checkpoint, require_encoder_state=True)

    sample_event = {"concept:name": "Start", "org:resource": "R1", "amount": 10.0}
    old_encoded = encoder.encode_event(event_extra=sample_event)
    new_encoded = new_encoder.encode_event(event_extra=sample_event)
    assert old_encoded.cat_indices == new_encoded.cat_indices
    assert old_encoded.num_values == new_encoded.num_values


@pytest.mark.mvp1_regression
def test_checkpoint_loader_accepts_legacy_payload_without_encoder_state(tmp_path):
    path = tmp_path / "legacy_resume_checkpoint.pth"
    payload = {
        "epoch": 3,
        "model_state_dict": {},
        "optimizer_state_dict": {},
        "val_loss": 0.5,
    }
    torch.save(payload, path)

    model = BaselineGCN(
        feature_layout=FeatureEncoder(
            feature_configs=[],
            traces=[],
            schema_resolver=SchemaResolver(),
        ).feature_layout,
        hidden_dim=8,
        output_dim=2,
        dropout=0.0,
    )
    trainer = _build_trainer(path, model=model, graph_builder=BaselineGraphBuilder(feature_encoder=FeatureEncoder(feature_configs=[], traces=[])))

    checkpoint = trainer._load_checkpoint(path, require_encoder_state=False)
    assert checkpoint["epoch"] == 3

    with pytest.raises(ValueError, match="encoder_state"):
        trainer._load_checkpoint(path, require_encoder_state=True)

