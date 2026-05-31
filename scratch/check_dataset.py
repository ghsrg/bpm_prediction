import json
from pathlib import Path

cache_dir = Path(".cache/graph_datasets/loan_v1_v4_simulated")
print("Scanning:", cache_dir.absolute())
if not cache_dir.exists():
    print("Does not exist!")
else:
    for entry in cache_dir.iterdir():
        print("Found entry:", entry.name)
        if entry.is_dir():
            meta_file = entry / "metadata.json"
            print("  meta_file exists:", meta_file.exists())
            if meta_file.exists():
                with open(meta_file, "r") as f:
                    meta = json.load(f)
                print(f"    Saved At: {meta.get('saved_at')}")
                print(f"    Schema: {meta.get('schema_version')}")
                # print number of shards
                shards = list(entry.glob("*.pt"))
                print(f"    Shards: {len(shards)} files")
