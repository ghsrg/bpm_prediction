from pathlib import Path

logs_dir = Path("logs")
if logs_dir.exists():
    for f in sorted(logs_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]:
        print(f"Log: {f} | mtime: {f.stat().st_mtime} | size: {f.stat().st_size}")
