import os
import glob
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def find_pt_files(root_dir):
    return glob.glob(os.path.join(root_dir, "**/*.pt"), recursive=True)

cache_dir = "c:/Users/korsr/PycharmProjects/bpm_prediction/.cache/graph_datasets"
pt_files = find_pt_files(cache_dir)
print(f"Found {len(pt_files)} .pt files in cache directory.")

if not pt_files:
    print("No .pt files found.")
    exit(0)

# Sort them so we get a consistent one
pt_files.sort()
first_pt_file = pt_files[0]
print(f"Loading pt file: {first_pt_file}")
try:
    loaded = torch.load(first_pt_file, weights_only=False)
    if isinstance(loaded, dict) and loaded.get("format") == "dedup_structural_payloads":
        graphs = loaded.get("graphs", [])
        registry = loaded.get("structural_payloads", {})
        print(f"Deduplicated format. Found {len(graphs)} graphs and {len(registry)} payloads.")
        data_list = []
        for graph in graphs:
            key = getattr(graph, "structural_payload_key", None)
            payload = registry.get(str(key)) if key is not None else None
            if isinstance(payload, dict):
                for name, value in payload.items():
                    setattr(graph, name, value)
            data_list.append(graph)
    else:
        data_list = loaded
        print(f"Standard format. Loaded {len(data_list)} graphs.")
        
    if not data_list:
        print("Data list is empty.")
        exit(0)
    # Check what keys are on the first graph
    g0 = data_list[0]
    print("Graph keys:", g0.keys())
    
    # Try creating a DataLoader and batching
    loader = DataLoader(data_list[:2], batch_size=2, shuffle=False)
    batch = next(iter(loader))
    print("Batch object type:", type(batch))
    
    print("Trying batch.get_example(0)...")
    try:
        first_ex = batch.get_example(0)
        print("get_example(0) succeeded!")
    except Exception as e:
        import traceback
        print("get_example(0) failed:")
        traceback.print_exc()
        
    print("Trying list(batch.to_data_list())...")
    try:
        data_list_sliced = list(batch.to_data_list())
        print("to_data_list() succeeded!")
    except Exception as e:
        import traceback
        print("to_data_list() failed:")
        traceback.print_exc()

except Exception as e:
    import traceback
    traceback.print_exc()
