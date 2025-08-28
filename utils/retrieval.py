from typing import Optional
import subprocess, sys, json, re

SUBJECT_MARKERS = {
    "IB Economics": ["/IB/Economics/", "economics", "tragakes"],
    "IB Business Management": ["/IB/BusinessManagement/", "business", "hoang"],
    "IB Math AA": ["/IB/Mathematics_AI/", "mathematics", "math"],
    "History": ["/IB/History/", "history"],
    "French AB": ["/IB/French_AbInitio/", "french"],
    "Finance": ["/UG/Finance/", "finance"],
}

from typing import Optional

def _matches_subject(src: str, subject: Optional[str]) -> bool:

    if not subject:
        return True
    markers = SUBJECT_MARKERS.get(subject, [])
    s = (src or "").lower()
    return any(m.lower() in s for m in markers)

def search_snippets(query: str, subject: Optional[str] = None, k: int = 4):
    r = subprocess.run([sys.executable, "tools/content_bank.py", "search", query], capture_output=True, text=True)
    out = r.stdout or r.stderr
    hits = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            src = obj.get("source", "")  # prefer full path if provided by content_bank
            page = obj.get("page")
            text = obj.get("text", "")
        except Exception:
            m = re.match(r"(.+?):(\d+):\s*(.+)", line)
            if not m:
                continue
            src, page, text = m.group(1), int(m.group(2)), m.group(3)
        if _matches_subject(src, subject):
            hits.append({"source": src, "page": page, "text": text})
    return hits[:k]
