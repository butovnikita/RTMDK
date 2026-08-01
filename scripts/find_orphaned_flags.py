import re
from pathlib import Path

text = Path("rtmdk/memory/config.py").read_text(encoding="utf-8")
start = text.find("_FIELD_GROUPS: Dict[str, str] = {")
end = start
brace_count = 0
for i, ch in enumerate(text[start:]):
    if ch == "{":
        brace_count += 1
    elif ch == "}":
        brace_count -= 1
        if brace_count == 0:
            end = start + i
            break

block = text[start : end + 1]
keys = re.findall(r'"([A-Za-z_][A-Za-z0-9_]*)"\s*:', block)
seen = set()
unique_keys = [k for k in keys if not (k in seen or seen.add(k))]

orphaned = []
for key in unique_keys:
    found = False
    for folder in ["rtmdk", "tests", "scripts"]:
        root = Path(folder)
        if not root.exists():
            continue
        for pyfile in root.rglob("*.py"):
            if pyfile.name == "config.py":
                continue
            content = pyfile.read_text(encoding="utf-8")
            if key in content:
                found = True
                break
        if found:
            break
    if not found:
        orphaned.append(key)

print("ORPHANED_FLAGS = [")
for k in orphaned:
    print(f'    "{k}",')
print("]")
