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

# Print slices before calling get_example
print("Is _slice_dict populated before calling get_example?")
print("batch._slice_dict is None:", getattr(batch, "_slice_dict", None) is None)

if hasattr(batch, "_slice_dict") and batch._slice_dict is not None:
    for k in [
        "structural_edge_index",
        "structural_edge_weight",
        "struct_node_to_class_index",
        "struct_node_to_candidate_index",
        "candidate_class_index",
        "candidate_is_unseen",
        "candidate_allowed_target_mask"
    ]:
        slices = batch._slice_dict.get(k)
        if slices is not None:
            print(f"  {k} slices:", slices.tolist() if isinstance(slices, torch.Tensor) else slices)
