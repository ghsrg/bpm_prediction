import os
from pathlib import Path

base_dir = Path(".cache/graph_datasets")
for root, dirs, files in os.walk(base_dir):
    if files:
        print(f"Directory: {root}")
        print(f"  Files: {files[:5]} ... (total {len(files)})")
