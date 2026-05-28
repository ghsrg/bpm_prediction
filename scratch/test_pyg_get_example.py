import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

g1 = Data(x=torch.zeros((2, 1)), edge_index=torch.zeros((2, 2), dtype=torch.long))
g1.candidate_ids = ['id1', 'id2']

g2 = Data(x=torch.zeros((2, 1)), edge_index=torch.zeros((2, 2), dtype=torch.long))
g2.candidate_ids = ['id3', 'id4', 'id5']

loader = DataLoader([g1, g2], batch_size=2, shuffle=False)
for batch in loader:
    # Test get_example(0)
    try:
        first_example = batch.get_example(0)
        print("get_example(0) candidate_ids:", getattr(first_example, "candidate_ids", None))
    except Exception as e:
        print("get_example(0) failed:", e)
        
    # Test to_data_list()[0]
    try:
        data_list = batch.to_data_list()
        print("to_data_list()[0] candidate_ids:", getattr(data_list[0], "candidate_ids", None))
    except Exception as e:
        print("to_data_list() failed:", e)
