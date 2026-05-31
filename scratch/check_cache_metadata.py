import os
import glob
import json

cache_dir = "c:/Users/korsr/PycharmProjects/bpm_prediction/.cache/graph_datasets"
summary_files = glob.glob(os.path.join(cache_dir, "**/summary.json"), recursive=True)
print(f"Found {len(summary_files)} summary files.")

for summary_file in summary_files:
    print(f"\nInspecting summary: {summary_file}")
    try:
        with open(summary_file, "r") as f:
            data = json.load(f)
        shards = data.get("shards", [])
        print(f"  Format: {data.get('format', 'N/A')}")
        print(f"  Number of shards: {len(shards)}")
        if shards:
            first_shard = shards[0]
            print("  First shard keys:", list(first_shard.keys()))
            topology_segments = first_shard.get("topology_segments")
            if topology_segments is None:
                print("  WARNING: topology_segments is MISSING!")
            else:
                print(f"  topology_segments present. Type: {type(topology_segments)}. Length: {len(topology_segments)}")
                if len(topology_segments) > 0:
                    print("    First segment:", topology_segments[0])
    except Exception as e:
        print(f"  Error reading summary: {e}")
