import os
import torch
import sys

# Ensure project root is in path in case loading needs project imports
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from torch_geometric.data import Data

cache_dir = ".cache/graph_datasets"
found = False

for root, dirs, files in os.walk(cache_dir):
    for file in files:
        if file.endswith(".pt"):
            path = os.path.join(root, file)
            try:
                # Load with weights_only=False since these are trusted local shards
                loaded = torch.load(path, map_location="cpu", weights_only=False)
                print(f"Successfully loaded shard: {path}")
                found = True
                if isinstance(loaded, dict):
                    print("Keys in loaded dict:", list(loaded.keys()))
                    if "structural_payloads" in loaded:
                        payloads = loaded["structural_payloads"]
                        print(f"  Deduplicated payloads count: {len(payloads)}")
                        for pk, pv in list(payloads.items())[:1]:
                            print(f"  Sample payload keys for {pk}:", list(pv.keys()))
                            for name in ["candidate_ids", "candidate_labels", "struct_x", "structural_edge_index"]:
                                if name in pv:
                                    val = pv[name]
                                    if isinstance(val, (list, tuple)):
                                        print(f"    {name} (len {len(val)}):", val[:5])
                                    else:
                                        print(f"    {name}: type {type(val)}")
                    if "graphs" in loaded and len(loaded["graphs"]) > 0:
                        g = loaded["graphs"][0]
                        print("  Sample graph keys:", sorted(list(g.__dict__.keys()) if hasattr(g, "__dict__") else []))
                        print("  Sample graph target_label:", getattr(g, "target_label", None))
                        print("  Sample graph process_version_idx:", getattr(g, "process_version_idx", None))
                        print("  Sample graph structural_payload_key:", getattr(g, "structural_payload_key", None))
                        print("  Sample graph candidate_ids:", getattr(g, "candidate_ids", None))
                        print("  Sample graph candidate_labels:", getattr(g, "candidate_labels", None))
                break
            except Exception as e:
                # Silently skip non-matching files
                pass
    if found:
        break
