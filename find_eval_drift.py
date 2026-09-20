from pathlib import Path

txt = Path("src/cli.py").read_text(encoding="utf-8")
for i, line in enumerate(txt.splitlines(), 1):
    if "eval_drift" in line:
        print(f"{i}: {line}")
