import sys
from src.cli import load_yaml_config, _build_trace_adapter, prepare_data, create_model, _build_model_factory_kwargs
from src.application.use_cases.trainer import ModelTrainer
from src.domain.services.prefix_policy import PrefixPolicy
from pathlib import Path

def main():
    config_path = "configs/experiments/ui_run_lhtcqucr.yaml"
    print(f"Loading config: {config_path}")
    config = load_yaml_config(config_path)
    
    # Set small fraction to speed up data prep
    config["experiment"]["fraction"] = 0.02
    config["experiment"]["graph_dataset_cache_policy"] = "off" # disable cache to measure prep time
    config["training"]["epochs"] = 1
    
    trace_adapter = _build_trace_adapter(config.get("mapping", {}))
    print("Preparing data...")
    prepared = prepare_data(config, trace_adapter=trace_adapter)
    
    activity_vocab = prepared["activity_vocab"]
    output_dim = len(activity_vocab)
    feature_layout = prepared["feature_layout"]
    
    model_kwargs = _build_model_factory_kwargs(
        model_cfg=config.get("model", {}),
        feature_layout=feature_layout,
        output_dim=output_dim,
    )
    model = create_model(**model_kwargs)
    
    trainer_experiment_cfg = dict(config.get("experiment", {}))
    trainer_experiment_cfg.update(prepared.get("experiment_split_config", {}))
    
    trainer_config = {
        **config.get("training", {}),
        "mapping_config": prepared["mapping_config"],
        "data_config": prepared["data_config"],
        "run_profile": prepared.get("run_profile", {}),
        "model_config": config.get("model", {}),
        "experiment_config": trainer_experiment_cfg,
        "tracking_config": config.get("tracking", {}),
        "config_path": str(config_path),
        "feature_configs": [
            {
                "name": item.name,
                "source_key": item.source_key,
                "encoding": list(item.encoding),
                "source": item.source,
                "dtype": item.dtype,
                "role": item.role,
            }
            for item in prepared["feature_configs"]
        ],
        "policy_config": config.get("policies", {}),
        "data_metrics": prepared["data_metrics"],
        "dataset_label": config.get("data", {}).get("dataset_label", "loan_v1_v4_simulated"),
        "model_label": config.get("model", {}).get("model_label", "GATv2"),
        "feature_layout": {
            "num_cat_features": len(prepared["feature_layout"].cat_feature_names),
            "num_num_channels": int(prepared["feature_layout"].num_dim),
            "cat_feature_names": list(prepared["feature_layout"].cat_feature_names),
        },
        "seed": 42,
        "retrain": True,
        "checkpoint_dir": Path("checkpoints"),
        "checkpoint_path": Path(config["experiment"]["load_checkpoint"]) if config["experiment"].get("load_checkpoint") else None,
        "resume_checkpoint_path": None,
        "mode": config["experiment"]["mode"],
        "drift_window_size": 100,
        "drift_window_sliding": 10,
    }
    
    trainer = ModelTrainer(
        xes_adapter=trace_adapter,
        prefix_policy=PrefixPolicy(),
        graph_builder=prepared["graph_builder"],
        model=model,
        log_path=prepared["log_path"],
        config=trainer_config,
        tracker=None,
        trace_recorder=None,
        class_weights=None,
        prepared_data=prepared,
    )
    
    print("Running evaluation...")
    results = trainer.run()
    print("Run completed successfully!")
    print("Keys in results:", results.keys())
    if "drift_metrics" in results:
        print(f"Number of drift windows evaluated: {len(results['drift_metrics'])}")

if __name__ == "__main__":
    main()
