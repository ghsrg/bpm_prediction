import re
from pathlib import Path

def main():
    recovered_path = Path("scratch/recovered_plan.txt")
    if not recovered_path.exists():
        print(f"File not found: {recovered_path}")
        return
        
    with open(recovered_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # We find all python blocks in the markdown
    # Note that markdown code blocks start with ```python and end with ```
    blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
    
    for block in blocks:
        lines = block.splitlines()
        if not lines:
            continue
        first_line = lines[0].strip()
        print(f"Found block starting with: {first_line}")
        
        # Match "# tools/desktop_ui/..." or similar comment
        m = re.match(r'^#\s*(tools/desktop_ui/\w+\.py)', first_line)
        if m:
            filepath = m.group(1)
            target_path = Path(filepath)
            print(f"Writing to {target_path}")
            target_path.parent.mkdir(parents=True, exist_ok=True)
            with open(target_path, "w", encoding="utf-8") as f:
                f.write(block)
                f.write("\n")

if __name__ == "__main__":
    main()
