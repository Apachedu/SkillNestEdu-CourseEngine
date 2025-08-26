import subprocess, sys, json, re

def search_snippets(query:str, k:int=4):
    r = subprocess.run([sys.executable, "tools/content_bank.py", "search", query], capture_output=True, text=True)
    out = r.stdout or r.stderr
    # tools/search prints JSONL or plain lines; try JSONL first
    hits = []
    for line in out.splitlines():
        line=line.strip()
        if not line: continue
        try:
            obj=json.loads(line)
            src = obj.get("source","")
            page = obj.get("page")
            text = obj.get("text","")
            hits.append({"source":src, "page":page, "text":text})
        except Exception:
            m=re.match(r"(.+?):(\d+):\s*(.+)", line)
            if m:
                src,pg,tx=m.group(1), int(m.group(2)), m.group(3)
                hits.append({"source":src, "page":pg, "text":tx})
    return hits[:k]
