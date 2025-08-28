import re, sys, pathlib

ROOT = pathlib.Path(".")
FILES = [p for p in ROOT.rglob("*.py") if ".venv" not in p.parts]

def ensure_optional_import(text: str) -> str:
    if "from typing import Optional" in text:
        return text
    lines = text.splitlines()
    for i, line in enumerate(lines[:30]):
        if line.strip().startswith("from typing import"):
            if "Optional" not in line:
                lines[i] = line.rstrip() + (", Optional" if line.strip().endswith("import") is False else " Optional")
            return "\n".join(lines)
        if line.strip().startswith("import typing"):
            return "\n".join(["from typing import Optional", *lines])
    return "from typing import Optional\n" + "\n".join(lines)

def fix_unions(text: str) -> str:
    # param annotations: ": Optional[X]"
    text = re.sub(r":\s*([^:\n]+?)\s*\|\s*None\b", r": Optional[\1]", text)
    # return annotations: Optional["-> X]"
    text = re.sub(r"->\s*([^:\n]+?)\s*\|\s*None\b", r"-> Optional[\1]", text)
    return text

changes = 0
for p in FILES:
    src = p.read_text(encoding="utf-8")
    new = fix_unions(src)
    if new != src:
        new = ensure_optional_import(new)
        p.write_text(new, encoding="utf-8")
        print(f"fixed: {p}")
        changes += 1

print(f"done. files_changed={changes}")
