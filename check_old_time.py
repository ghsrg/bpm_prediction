from pathlib import Path
from datetime import datetime

for exp_dir in Path("mlruns").iterdir():
    if not exp_dir.is_dir():
        continue
    rd = exp_dir / "302fb003f78d468b867a9477e40e9d50"
    if rd.is_dir():
        meta = (rd / "meta.yaml").read_text(encoding="utf-8")
        print("meta.yaml of 302fb003f78d468b867a9477e40e9d50:")
        print(meta)
        for line in meta.splitlines():
            if "start_time:" in line:
                st = int(line.split("start_time:")[1].strip())
                print("start_time readable:", datetime.fromtimestamp(st / 1000))
