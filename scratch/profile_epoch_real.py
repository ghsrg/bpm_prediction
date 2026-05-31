import os
import glob
import time
import yaml
import cProfile
import pstats
from pathlib import Path
from src.cli import load_yaml_config
from src.cli import _build_trace_adapter, prepare_data, create_model, _build_model_factory_kwargs, _parse_split_ratio, _trace_version_counts, _apply_cascade_prepare
from src.application.use_cases.trainer import ModelTrainer
from src.domain.services.prefix_policy import PrefixPolicy

def get_latest_ui_run_yaml():
    yaml_files = glob.glob("configs/experiments/ui_run_*.yaml")
    if not yaml_files:
        return None
    yaml_files.sort(key=os.path.getmtime, reverse=True)
    return yaml_files[0]

def profile_run():
    config_path = get_latest_ui_run_yaml()
    if not config_path:
        print("No ui_run_*.yaml files found.")
        return
    print(f"Profiling using config: {config_path}")
    
    config = load_yaml_config(config_path)
    
    # Force cpu and epochs=1 for profiling
    config["training"]["device"] = "cpu"
    config["training"]["epochs"] = 1
    
    # We want to measure the data loading and forward time
    # Build components
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
    
    # Build trainer config
    experiment_cfg = config.get("experiment", {})
    trainer_experiment_cfg = dict(experiment_cfg)
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
        "checkpoint_path": Path("checkpoints/temp.pth"),
        "resume_checkpoint_path": None,
        "mode": "train",
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
    
    print("Building loaders...")
    prebuilt_datasets = {
        "train": prepared.get("train_dataset"),
        "val": prepared.get("val_dataset"),
        "test": prepared.get("test_dataset"),
    }
    train_loader = trainer._build_loader_from_dataset(prebuilt_datasets.get("train"), shuffle=True)
    print(f"Number of train batches: {len(train_loader)}")
    
    # We will profile training exactly 5 batches
    print("Starting profiling of 5 batches...")
    
    profiler = cProfile.Profile()
    profiler.enable()
    
    t0 = time.perf_counter()
    trainer.model.train()
    iterator = iter(train_loader)
    for batch_idx in range(5):
        try:
            batch = next(iterator)
        except StopIteration:
            break
        print(f"Batch {batch_idx+1}/5 loaded.")
        batch = batch.to(trainer.device)
        contract = trainer._data_to_contract(batch)
        logits = trainer._forward_candidate_output(contract).candidate_logits
        targets = batch.y.view(-1).long()
        loss = trainer.criterion(logits, targets)
        # no backward step to keep it simple or we can add it
        
    t1 = time.perf_counter()
    profiler.disable()
    
    print(f"Profiled 5 batches in {t1 - t0:.4f} seconds.")
    stats = pstats.Stats(profiler).sort_stats('cumulative')
    stats.print_stats(30)

if __name__ == "__main__":
    profile_run()
