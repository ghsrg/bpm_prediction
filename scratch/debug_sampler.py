import yaml
import logging
from src.cli import prepare_data
from src.application.use_cases.trainer import _TopologyHomogeneousBatchSampler, ShardedGraphDataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load config from successful run 2c866c7723594324ae59d970b3288774
with open("mlruns/854778689611649472/2c866c7723594324ae59d970b3288774/artifacts/ui_run_jpv_4utw.yaml", "r") as f:
    config = yaml.safe_load(f)

# Override device to cpu for debugging
config["training"]["device"] = "cpu"

print("Preparing data...")
data_artifacts = prepare_data(config)
train_dataset_raw = data_artifacts["train_dataset"]

print(f"Train dataset raw type: {type(train_dataset_raw)}")

if isinstance(train_dataset_raw, dict) and train_dataset_raw.get("kind") == "sharded_cache_split":
    train_dataset = ShardedGraphDataset.from_payload(train_dataset_raw, max_cached_shards=2)
else:
    train_dataset = train_dataset_raw

print(f"Train dataset type: {type(train_dataset)}")
print(f"Train dataset length: {len(train_dataset)}")

# Resolve topology keys
print("Resolving topology keys...")
topology_keys = _TopologyHomogeneousBatchSampler._resolve_topology_keys(train_dataset)
print(f"Topology keys count: {len(topology_keys)}")

from collections import Counter
keys_counter = Counter(topology_keys)
print(f"Unique topology keys and counts: {dict(keys_counter)}")

# Create sampler with shuffle=True
sampler = _TopologyHomogeneousBatchSampler(
    train_dataset,
    batch_size=128,
    shuffle=True,
    seed=42
)

# Generate batches
batches = list(sampler)
print(f"Number of generated batches: {len(batches)}")

# Check topology homogeneity of each batch
mixed_batches_count = 0
for idx, batch in enumerate(batches):
    batch_keys = [topology_keys[i] for i in batch]
    unique_keys = set(batch_keys)
    if len(unique_keys) > 1:
        mixed_batches_count += 1
        if mixed_batches_count <= 5:
            print(f"Mixed batch {idx}: size={len(batch)}, unique_keys={unique_keys}")

print(f"Total mixed batches: {mixed_batches_count} out of {len(batches)}")
