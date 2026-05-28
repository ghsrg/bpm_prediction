import json
from pathlib import Path

def search_in_file(path: Path):
    if not path.exists():
        return
    print(f"\nSearching in: {path}")
    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            try:
                data = json.loads(line)
            except Exception:
                continue
            content = str(data.get("content", ""))
            tool_calls = data.get("tool_calls", [])
            
            # Search in tool_calls to see what args were passed
            for tc in tool_calls:
                args = str(tc.get("arguments", ""))
                if "DARK_SLATE_THEME" in args or "PresetDrawer" in args or "RunMonitorWidget" in args:
                    print(f"  Step {data.get('step_index')}: found in tool call args of {tc.get('name')}")
                    # Write args to file
                    out_path = Path(f"scratch/recovered_args_{path.parent.parent.name}_{data.get('step_index')}.txt")
                    with open(out_path, "w", encoding="utf-8") as out_f:
                        out_f.write(args)
                    print(f"  Saved args to {out_path}")
            
            if "DARK_SLATE_THEME" in content or "PresetDrawer" in content or "RunMonitorWidget" in content:
                print(f"  Step {data.get('step_index')}: found in content (type: {data.get('type')})")

def main():
    brain_dir = Path(r"C:\Users\korsr\.gemini\antigravity\brain")
    for transcript_path in brain_dir.rglob("transcript.jsonl"):
        search_in_file(transcript_path)

if __name__ == "__main__":
    main()
