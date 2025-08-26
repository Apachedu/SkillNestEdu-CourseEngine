import pathlib, json, textwrap

ROOT = pathlib.Path(".").resolve()

FILES = {
"tools/content_bank.py": r'''import os, json, re, pathlib, glob
from typing import List, Dict, Tuple
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib

BASE = pathlib.Path(".").resolve()
SRC = BASE / "source_pdfs"
BANK = BASE / "content_bank"
IDX  = BANK / "index"
BANK.mkdir(parents=True, exist_ok=True)
IDX.mkdir(parents=True, exist_ok=True)

def _clean(txt:str)->str:
    t = re.sub(r"\\s+", " ", txt or " ").strip()
    return t

def _chunks(text:str, target_chars:int=1200) -> List[str]:
    parts = re.split(r"(\\n{2,}|[.!?])", text)
    buf, out = "", []
    for p in parts:
        if p is None: continue
        buf += p
        if len(buf) >= target_chars:
            out.append(_clean(buf)); buf = ""
    if _clean(buf): out.append(_clean(buf))
    return [c for c in out if len(c) > 60]

def _coll_name(board:str, subject:str)->str:
    return f"{board.strip().lower().replace(' ','_')}__{subject.strip().lower().replace(' ','_')}"

def ingest_pdf(pdf_path:str, board:str, subject:str, book_title:str)->int:
    coll = _coll_name(board, subject)
    out_path = BANK / f"{coll}.jsonl"
    count = 0
    with fitz.open(pdf_path) as doc, open(out_path, "a", encoding="utf-8") as f:
        for i, page in enumerate(doc, start=1):
            text = page.get_text("text")
            for ch in _chunks(text):
                rec = {
                    "board": board, "subject": subject, "book": book_title,
                    "page": i, "content": ch
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                count += 1
    return count

def ingest_all()->List[Tuple[str,int]]:
    results=[]
    for board_dir in sorted(glob.glob(str(SRC / "*"))):
        board = pathlib.Path(board_dir).name
        for subj_dir in sorted(glob.glob(str(pathlib.Path(board_dir) / "*"))):
            subject = pathlib.Path(subj_dir).name
            pdfs = sorted(glob.glob(str(pathlib.Path(subj_dir) / "*.pdf")))
            for pdf in pdfs:
                book_title = pathlib.Path(pdf).stem
                n = ingest_pdf(pdf, board, subject, book_title)
                results.append((f"{board}/{subject}/{book_title}", n))
    return results

def _load_corpus(coll:str)->List[Dict]:
    path = BANK / f"{coll}.jsonl"
    if not path.exists(): return []
    rows=[]
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try: rows.append(json.loads(line))
            except: pass
    return rows

def build_index(coll:str)->Tuple[str,int]:
    rows = _load_corpus(coll)
    texts = [r["content"] for r in rows]
    if not texts: return (coll,0)
    vec = TfidfVectorizer(max_features=40000, ngram_range=(1,2))
    X = vec.fit_transform(texts)
    joblib.dump({"vectorizer": vec, "matrix": X, "rows": rows}, IDX / f"{coll}.pkl")
    return (coll, len(rows))

def build_index_all()->List[Tuple[str,int]]:
    built=[]
    for p in BANK.glob("*.jsonl"):
        coll = p.stem
        built.append(build_index(coll))
    return built

def search(board:str, subject:str, query:str, topk:int=5)->List[Dict]:
    coll = _coll_name(board, subject)
    idx_path = IDX / f"{coll}.pkl"
    if not idx_path.exists(): return []
    pack = joblib.load(idx_path)
    vec, X, rows = pack["vectorizer"], pack["matrix"], pack["rows"]
    q = vec.transform([query])
    sims = cosine_similarity(q, X)[0]
    top = sims.argsort()[-topk:][::-1]
    out=[]
    for i in top:
        r = rows[i].copy()
        r["score"] = float(sims[i])
        out.append(r)
    return out

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python tools/content_bank.py [ingest_all|build_index_all|search <board> <subject> <query>]"); sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "ingest_all":
        SRC.mkdir(parents=True, exist_ok=True)
        res = ingest_all()
        for name,count in res: print(f"{name}: {count} chunks")
        sys.exit(0)
    if cmd == "build_index_all":
        built = build_index_all()
        for coll,n in built: print(f"{coll}: {n} chunks indexed")
        sys.exit(0)
    if cmd == "search":
        board, subject, query = sys.argv[2], sys.argv[3], sys.argv[4]
        hits = search(board, subject, query, topk=5)
        for h in hits:
            print(json.dumps(h, ensure_ascii=False))
        sys.exit(0)
    print("Unknown command"); sys.exit(1)
'''
}


def main():
    for rel, content in FILES.items():
        path = ROOT / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        print("Wrote", rel)

if __name__ == "__main__":
    main()

