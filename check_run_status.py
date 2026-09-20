from pathlib import Path

for exp_dir in Path("mlruns").iterdir():
    if not exp_dir.is_dir():
        continue
    rd = exp_dir / "f71502fd16624ce2b838f50f91dd160a"
    if rd.is_dir():
        meta = (rd / "meta.yaml").read_text(encoding="utf-8")
        print("meta.yaml:")
        print(meta)
