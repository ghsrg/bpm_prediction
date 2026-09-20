from pathlib import Path
import json

kg_dir = Path("data/knowledge_graph")
for p in kg_dir.iterdir():
    if "loan" in p.name.lower():
        print("Found loan dir:", p.name)
        for sub in p.iterdir():
            print("  sub:", sub.name)
            for f in sub.iterdir():
                print("    file:", f.name, f.stat().st_size)
