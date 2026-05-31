import os
import glob
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

cache_dir = "c:/Users/korsr/PycharmProjects/bpm_prediction/.cache/graph_datasets"
pt_files = glob.glob(os.path.join(cache_dir, "**/*.pt"), recursive=True)
pt_files.sort()
first_pt_file = pt_files[0]
loaded = torch.load(first_pt_file, weights_only=False)

if isinstance(loaded, dict) and loaded.get("format") == "dedup_structural_payloads":
    graphs = loaded.get("graphs", [])
    registry = loaded.get("structural_payloads", {})
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

loader = DataLoader(data_list[:4], batch_size=4, shuffle=False)
batch = next(iter(loader))

print("Batch internal attributes:")
for k, v in batch.__dict__.items():
    if "slice" in k or "ptr" in k or "offset" in k or k.startswith("_"):
        if isinstance(v, dict):
            print(f"  {k}: dict with keys {list(v.keys())}")
        elif isinstance(v, torch.Tensor):
            print(f"  {k}: Tensor shape={list(v.shape)}")
        else:
            print(f"  {k}: {type(v)}")

if hasattr(batch, "_slice_dict") and batch._slice_dict is not None:
    print("\n_slice_dict keys:")
    for k, v in batch._slice_dict.items():
        print(f"  {k}: {type(v)} (len={len(v) if hasattr(v, '__len__') else 'N/A'})")
