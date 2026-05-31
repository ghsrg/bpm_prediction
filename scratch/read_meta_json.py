import json
from pathlib import Path

base_dir = Path(".cache/graph_datasets/loan_v1_v4_simulated")
for entry in base_dir.iterdir():
    if entry.is_dir():
        meta_file = entry / "meta.json"
        if meta_file.exists():
            with open(meta_file, "r") as f:
                meta = json.load(f)
            print(f"Directory: {entry.name}")
            print(f"  Saved At: {meta.get('saved_at') or meta.get('created_at')}")
            print(f"  Schema: {meta.get('schema_version') or meta.get('schema')}")
            print(f"  Config: {meta.get('config', {}).get('experiment_name') or meta.get('experiment_name')}")
            print(f"  Fraction: {meta.get('config', {}).get('fraction') or meta.get('fraction')}")
            print(f"  Split strategy: {meta.get('config', {}).get('split_strategy') or meta.get('split_strategy')}")
            print(f"  Train ratio: {meta.get('config', {}).get('train_ratio') or meta.get('train_ratio')}")
            # Count shard files
            train_shards = list((entry / "train_shards").glob("*.pt"))
            val_shards = list((entry / "validation_shards").glob("*.pt"))
            test_shards = list((entry / "test_shards").glob("*.pt"))
            print(f"  Shards - Train: {len(train_shards)} | Val: {len(val_shards)} | Test: {len(test_shards)}")
