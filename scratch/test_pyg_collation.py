import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

g1 = Data(x=torch.zeros((2, 1)), edge_index=torch.zeros((2, 2), dtype=torch.long))
g1.candidate_ids = ['id1', 'id2']

g2 = Data(x=torch.zeros((2, 1)), edge_index=torch.zeros((2, 2), dtype=torch.long))
g2.candidate_ids = ['id3', 'id4', 'id5']

loader = DataLoader([g1, g2], batch_size=2, shuffle=False)
for batch in loader:
    print("Batch type:", type(batch))
    print("Batch candidate_ids:", batch.candidate_ids)
    print("Batch candidate_ids type:", type(batch.candidate_ids))
    if isinstance(batch.candidate_ids, list):
        print("First element of batch.candidate_ids:", batch.candidate_ids[0])
        print("First element type:", type(batch.candidate_ids[0]))
