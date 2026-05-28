import json
from pathlib import Path

def main():
    transcript_path = Path(r"C:\Users\korsr\.gemini\antigravity\brain\aeaec5f7-b6a5-40d8-8b96-1bb12c4cfe30\.system_generated\logs\transcript.jsonl")
    if not transcript_path.exists():
        print(f"Transcript path not found: {transcript_path}")
        return
        
    print(f"Reading transcript: {transcript_path}")
    # Let's read all lines and find where the content contains the files we need.
    # We will search for 'class PresetDrawer' or 'class RunMonitorWidget'.
    found = False
    with open(transcript_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
            except Exception:
                continue
            content = str(data.get("content", ""))
            
            # Look for preset drawer and run monitor definitions
            if "class PresetDrawer" in content or "class RunMonitorWidget" in content or "DARK_SLATE_THEME" in content:
                # We want to check if it has the full plan or files
                # Wait, let's print some info
                print(f"Found step index: {data.get('step_index')} (type: {data.get('type')})")
                out_path = Path(f"scratch/recovered_plan_prev_{data.get('step_index')}.txt")
                with open(out_path, "w", encoding="utf-8") as out_f:
                    out_f.write(content)
                print(f"Saved content to {out_path}")
                found = True
                
    if not found:
        print("Nothing found in previous transcript.")

if __name__ == "__main__":
    main()
