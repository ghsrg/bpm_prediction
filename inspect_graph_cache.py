from pathlib import Path
import torch

# Search in .cache/graph_datasets
cache_dir = Path(".cache/graph_datasets")
print("Cache dir exists:", cache_dir.exists())
for p in sorted(cache_dir.rglob("*.pt")):
    try:
        data = torch.load(p, map_location="cpu")
        has_audit = False
        sample = None
        if isinstance(data, list) and len(data) > 0:
            sample = data[0]
            has_audit = hasattr(sample, "audit_payload_json") and sample.audit_payload_json is not None
        print(f"File: {p} | items: {len(data) if isinstance(data, list) else type(data)} | has_audit: {has_audit}")
        if sample is not None:
            print(f"   sample keys: {list(sample.keys()) if hasattr(sample, 'keys') else dir(sample)}")
            if hasattr(sample, "audit_payload_json"):
                print(f"   audit_payload_json: {sample.audit_payload_json}")
    except Exception as e:
        print(f"File {p} error: {e}")
