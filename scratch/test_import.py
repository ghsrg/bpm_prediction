print("Starting import test...")
import time
t0 = time.perf_counter()
from src.application.use_cases.trainer import ShardedGraphDataset
t1 = time.perf_counter()
print(f"Import ShardedGraphDataset took: {t1-t0:.4f}s")
