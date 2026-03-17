from __future__ import annotations

from typing import Iterator

from torch_geometric.data import Data

from src.application.use_cases.trainer import ModelTrainer, SplitData
from src.domain.entities.raw_trace import RawTrace
from src.domain.models.baseline_gcn import BaselineGCN
from src.domain.services.baseline_graph_builder import BaselineGraphBuilder
from src.domain.services.feature_encoder import FeatureEncoder
from src.domain.services.prefix_policy import PrefixPolicy


class _FailOnReadAdapter:
    def read(self, file_path: str, mapping_config: dict) -> Iterator[RawTrace]:
        _ = file_path
        _ = mapping_config
        raise AssertionError("Adapter read should not be called when prepared_data is provided.")


def _contract_to_data(contract: dict) -> Data:
    payload = {
        "x_cat": contract["x_cat"],
        "x_num": contract["x_num"],
        "edge_index": contract["edge_index"],
        "edge_type": contract["edge_type"],
        "y": contract["y"],
        "num_nodes": int(contract["num_nodes"]),
    }
    return Data(**payload)


def test_trainer_run_uses_preloaded_data_without_second_read(mock_feature_configs, mock_raw_trace, tmp_path):
    traces = [mock_raw_trace, mock_raw_trace, mock_raw_trace]
    prefix_policy = PrefixPolicy()
    encoder = FeatureEncoder(feature_configs=mock_feature_configs, traces=traces)
    graph_builder = BaselineGraphBuilder(feature_encoder=encoder)

    contracts = []
    for trace in traces:
        for prefix in prefix_policy.generate_slices(trace):
            contracts.append(graph_builder.build_graph(prefix))
    assert contracts

    graph_dataset = [_contract_to_data(contract) for contract in contracts]

    model = BaselineGCN(
        feature_layout=encoder.feature_layout,
        hidden_dim=8,
        output_dim=len(encoder.categorical_vocabs[encoder.activity_feature_name]),
        dropout=0.0,
    )

    trainer = ModelTrainer(
        xes_adapter=_FailOnReadAdapter(),
        prefix_policy=prefix_policy,
        graph_builder=graph_builder,
        model=model,
        log_path="in_memory.xes",
        config={
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 0.001,
            "device": "cpu",
            "show_progress": False,
            "tqdm_disable": True,
            "checkpoint_dir": str(tmp_path),
            "experiment_config": {
                "name": "pytest_preloaded",
                "mode": "train",
                "fraction": 1.0,
                "split_strategy": "temporal",
                "train_ratio": 1.0,
                "split_ratio": [0.7, 0.2, 0.1],
            },
        },
        prepared_data={
            "prepared_traces": traces,
            "train_traces": traces,
            "val_traces": traces[:1],
            "test_traces": traces[:1],
            "train_dataset": graph_dataset,
            "val_dataset": graph_dataset[:1],
            "test_dataset": graph_dataset[:1],
            "idx_to_version": {0: "v1"},
        },
    )

    captured = {}

    def _fake_run_train_pipeline(**kwargs):
        captured["prebuilt_datasets"] = kwargs.get("prebuilt_datasets")
        split_data = kwargs.get("split_data")
        assert isinstance(split_data, SplitData)
        return {"history": [], "test_metrics": {}, "mode": "train"}

    trainer._run_train_pipeline = _fake_run_train_pipeline  # type: ignore[method-assign]
    trainer._prepare_checkpoint_state = lambda is_eval_mode: (None, 0, float("inf"), 0)  # type: ignore[assignment]

    result = trainer.run()

    assert result["mode"] == "train"
    assert captured["prebuilt_datasets"] is not None
    assert "train" in captured["prebuilt_datasets"]
