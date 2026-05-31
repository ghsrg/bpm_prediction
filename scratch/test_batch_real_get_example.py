import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

# Mock custom diagnostics class
class DummyDiagnostics:
    def __init__(self):
        self.is_aligned = True

g1 = Data(
    x_cat=torch.zeros((1, 0), dtype=torch.long),
    x_num=torch.ones((1, 1), dtype=torch.float32),
    edge_index=torch.zeros((2, 0), dtype=torch.long),
    y=torch.tensor([0], dtype=torch.long),
    num_nodes=1,
    allowed_masks_by_src={"a": [True, False]},
    candidate_ids=("id1", "id2"),
    candidate_labels=("lbl1", "lbl2"),
    candidate_class_index=torch.tensor([0, 1], dtype=torch.long),
    candidate_is_unseen=torch.tensor([False, True], dtype=torch.bool),
    struct_node_to_candidate_index=torch.tensor([0, 1], dtype=torch.long),
    topology_projection_diagnostics=DummyDiagnostics()
)

g2 = Data(
    x_cat=torch.zeros((1, 0), dtype=torch.long),
    x_num=torch.ones((1, 1), dtype=torch.float32),
    edge_index=torch.zeros((2, 0), dtype=torch.long),
    y=torch.tensor([1], dtype=torch.long),
    num_nodes=1,
    allowed_masks_by_src={"a": [True, False]},
    candidate_ids=("id1", "id2"),
    candidate_labels=("lbl1", "lbl2"),
    candidate_class_index=torch.tensor([0, 1], dtype=torch.long),
    candidate_is_unseen=torch.tensor([False, True], dtype=torch.bool),
    struct_node_to_candidate_index=torch.tensor([0, 1], dtype=torch.long),
    topology_projection_diagnostics=DummyDiagnostics()
)

loader = DataLoader([g1, g2], batch_size=2, shuffle=False)
for batch in loader:
    try:
        first = batch.get_example(0)
        print("get_example(0) succeeded!")
    except Exception as e:
        import traceback
        print("get_example(0) failed:")
        traceback.print_exc()
