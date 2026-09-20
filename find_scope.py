from pathlib import Path

txt = Path("src/cli.py").read_text(encoding="utf-8")
for i, line in enumerate(txt.splitlines(), 1):
    if "_apply_version_scope" in line:
        print(f"{i}: {line}")
