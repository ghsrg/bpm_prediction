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

print("structural_edge_index max value:", batch.structural_edge_index.max().item())
print("structural_edge_index values in first graph area:")
edge_count = batch.structural_edge_index.size(1) // 4
first_edges = batch.structural_edge_index[:, :edge_count]
print("First graph edges max value:", first_edges.max().item())
print("First graph edges shape:", first_edges.shape)
