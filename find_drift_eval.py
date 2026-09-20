from pathlib import Path

txt = Path("src/application/use_cases/trainer.py").read_text(encoding="utf-8")
for i, line in enumerate(txt.splitlines(), 1):
    if "def _evaluate_drift" in line:
        print(f"{i}: {line}")
