import os
import glob
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def _slice_first(tensor, key, slice_dict, dim=0):
    if tensor is None:
        return None
    if not isinstance(tensor, torch.Tensor):
        return tensor
    if slice_dict is not None and key in slice_dict:
        slices = slice_dict[key]
        if isinstance(slices, torch.Tensor) and slices.numel() > 1:
            end_idx = int(slices[1].item())
            if dim == 0:
                return tensor[:end_idx]
            elif dim == 1:
                return tensor[:, :end_idx]
    return tensor

def get_first_list_attr(val):
    if val is not None:
        if len(val) > 0 and isinstance(val[0], (list, tuple)):
            return tuple(val[0])
        return tuple(val)
    return None

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

# 1. Test Homogeneous Batch
print("--- Homogeneous Batch ---")
loader = DataLoader(data_list[:4], batch_size=4, shuffle=False)
batch = next(iter(loader))
ref = batch.get_example(0)
slice_dict = getattr(batch, "_slice_dict", None)

struct_x = getattr(batch, "struct_x", None)
structural_edge_index = getattr(batch, "structural_edge_index", None)
structural_edge_weight = getattr(batch, "structural_edge_weight", None)
struct_node_to_class_index = getattr(batch, "struct_node_to_class_index", None)
struct_node_to_candidate_index = getattr(batch, "struct_node_to_candidate_index", None)
candidate_class_index = getattr(batch, "candidate_class_index", None)
candidate_is_unseen = getattr(batch, "candidate_is_unseen", None)
candidate_allowed_target_mask = getattr(batch, "candidate_allowed_target_mask", None)
candidate_ids = getattr(batch, "candidate_ids", None)

print("struct_x match:", torch.equal(_slice_first(struct_x, "struct_x", slice_dict, 0), ref.struct_x) if struct_x is not None else "None")
print("structural_edge_index match:", torch.equal(_slice_first(structural_edge_index, "structural_edge_index", slice_dict, 1), ref.structural_edge_index))
print("structural_edge_weight match:", torch.equal(_slice_first(structural_edge_weight, "structural_edge_weight", slice_dict, 0), ref.structural_edge_weight))
print("struct_node_to_class_index match:", torch.equal(_slice_first(struct_node_to_class_index, "struct_node_to_class_index", slice_dict, 0), ref.struct_node_to_class_index))
print("struct_node_to_candidate_index match:", torch.equal(_slice_first(struct_node_to_candidate_index, "struct_node_to_candidate_index", slice_dict, 0), ref.struct_node_to_candidate_index))
print("candidate_class_index match:", torch.equal(_slice_first(candidate_class_index, "candidate_class_index", slice_dict, 0), ref.candidate_class_index))
print("candidate_is_unseen match:", torch.equal(_slice_first(candidate_is_unseen, "candidate_is_unseen", slice_dict, 0), ref.candidate_is_unseen))
if candidate_allowed_target_mask is not None:
    sliced_allowed_mask = candidate_allowed_target_mask[0:1] if candidate_allowed_target_mask.dim() == 2 else _slice_first(candidate_allowed_target_mask, "candidate_allowed_target_mask", slice_dict, 0)
    print("candidate_allowed_target_mask match:", torch.equal(sliced_allowed_mask, ref.candidate_allowed_target_mask))
else:
    print("candidate_allowed_target_mask: None")
print("candidate_ids match:", tuple(get_first_list_attr(candidate_ids)) == tuple(ref.candidate_ids))

# 2. Test Heterogeneous Batch
print("\n--- Heterogeneous Batch ---")
g_a = data_list[0]
g_b = None
for g in data_list[1:]:
    if g.structural_edge_index.size(1) != g_a.structural_edge_index.size(1):
        g_b = g
        break
if g_b is None:
    print("Could not find different topology graphs in cache, using index 100")
    g_b = data_list[min(100, len(data_list)-1)]

loader = DataLoader([g_a, g_b], batch_size=2, shuffle=False)
batch_hetero = next(iter(loader))
ref_hetero = batch_hetero.get_example(0)
slice_dict_hetero = getattr(batch_hetero, "_slice_dict", None)

struct_x_het = getattr(batch_hetero, "struct_x", None)
structural_edge_index_het = getattr(batch_hetero, "structural_edge_index", None)
structural_edge_weight_het = getattr(batch_hetero, "structural_edge_weight", None)
struct_node_to_class_index_het = getattr(batch_hetero, "struct_node_to_class_index", None)
struct_node_to_candidate_index_het = getattr(batch_hetero, "struct_node_to_candidate_index", None)
candidate_class_index_het = getattr(batch_hetero, "candidate_class_index", None)
candidate_is_unseen_het = getattr(batch_hetero, "candidate_is_unseen", None)
candidate_allowed_target_mask_het = getattr(batch_hetero, "candidate_allowed_target_mask", None)
candidate_ids_het = getattr(batch_hetero, "candidate_ids", None)

print("struct_x match:", torch.equal(_slice_first(struct_x_het, "struct_x", slice_dict_hetero, 0), ref_hetero.struct_x) if struct_x_het is not None else "None")
print("structural_edge_index match:", torch.equal(_slice_first(structural_edge_index_het, "structural_edge_index", slice_dict_hetero, 1), ref_hetero.structural_edge_index))
print("structural_edge_weight match:", torch.equal(_slice_first(structural_edge_weight_het, "structural_edge_weight", slice_dict_hetero, 0), ref_hetero.structural_edge_weight))
print("struct_node_to_class_index match:", torch.equal(_slice_first(struct_node_to_class_index_het, "struct_node_to_class_index", slice_dict_hetero, 0), ref_hetero.struct_node_to_class_index))
print("struct_node_to_candidate_index match:", torch.equal(_slice_first(struct_node_to_candidate_index_het, "struct_node_to_candidate_index", slice_dict_hetero, 0), ref_hetero.struct_node_to_candidate_index))
print("candidate_class_index match:", torch.equal(_slice_first(candidate_class_index_het, "candidate_class_index", slice_dict_hetero, 0), ref_hetero.candidate_class_index))
print("candidate_is_unseen match:", torch.equal(_slice_first(candidate_is_unseen_het, "candidate_is_unseen", slice_dict_hetero, 0), ref_hetero.candidate_is_unseen))
if candidate_allowed_target_mask_het is not None:
    sliced_allowed_mask_het = candidate_allowed_target_mask_het[0:1] if candidate_allowed_target_mask_het.dim() == 2 else _slice_first(candidate_allowed_target_mask_het, "candidate_allowed_target_mask", slice_dict_hetero, 0)
    print("candidate_allowed_target_mask match:", torch.equal(sliced_allowed_mask_het, ref_hetero.candidate_allowed_target_mask))
else:
    print("candidate_allowed_target_mask: None")
print("candidate_ids match:", tuple(get_first_list_attr(candidate_ids_het)) == tuple(ref_hetero.candidate_ids))
