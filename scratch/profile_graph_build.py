import os
import sys
import psutil
import torch
import gc

# Add the project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.cli import load_yaml_config, _apply_experiment_switch_overrides, _build_trace_adapter, prepare_data
from src.domain.services.dynamic_graph_builder import DynamicGraphBuilder
from src.domain.entities.prefix_slice import PrefixSlice

def print_memory(label):
    process = psutil.Process()
    rss = process.memory_info().rss / (1024 * 1024 * 1024)
    vms = process.memory_info().vms / (1024 * 1024 * 1024)
    print(f"[{label}] RSS: {rss:.3f} GB | VMS: {vms:.3f} GB")

def main():
    config_path = "configs/experiments/ui_run_feacvhua.yaml"
    if not os.path.exists(config_path):
        print(f"Config path {config_path} does not exist!")
        return

    print(f"Loading config from {config_path}")
    config = load_yaml_config(config_path)
    
    # We want to run with fraction=0.2 (20% of dataset) to build faster but see memory trend
    config["experiment"]["fraction"] = 0.2
    config = _apply_experiment_switch_overrides(config)
    
    print_memory("Start")
    trace_adapter = _build_trace_adapter(config.get("mapping", {}))
    
    # To trace memory during prep, let's wrap prepare_data or see how it behaves
    # We can temporarily patch _build_graph_dataset_sharded or similar to print RSS every 1000 graphs
    import src.cli
    original_build_sharded = src.cli._build_graph_dataset_sharded
    
    def patched_build_sharded(*args, **kwargs):
        print_memory(f"Entering build_sharded for {kwargs.get('split_key')}")
        
        # We can print memory inside the loop or after
        # Let's let it run and monitor
        res = original_build_sharded(*args, **kwargs)
        print_memory(f"Exiting build_sharded for {kwargs.get('split_key')}")
        return res
        
    src.cli._build_graph_dataset_sharded = patched_build_sharded
    
    prepared = prepare_data(config, trace_adapter=trace_adapter)
    print_memory("After prepare_data")
    
    # Clean cache and trigger GC
    gc.collect()
    print_memory("After GC")

if __name__ == "__main__":
    main()
