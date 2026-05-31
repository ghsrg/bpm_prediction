from pathlib import Path

train_shards_dir = Path(".cache/graph_datasets/loan_v1_v4_simulated/942b0cd3613e821387558a22168d898f93a288a2/train_shards")
for f in train_shards_dir.glob("*.pt"):
    print(f"{f.name}: {f.stat().st_size / (1024*1024):.2f} MB")
