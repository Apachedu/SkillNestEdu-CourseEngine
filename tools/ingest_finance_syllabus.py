import sys, json, re, pathlib
import pandas as pd

def norm(s):
    if s is None: return ""
    return re.sub(r"\s+", " ", str(s)).strip()

def slug(s):
    return re.sub(r"[^a-z0-9]+","-", norm(s).lower()).strip("-")

# map many possible header spellings -> canonical keys
HEADER_MAP = {
    "sem":"semester","semester":"semester","term":"semester",
    "sem no.":"semester","sem_no":"semester","sem_number":"semester",
    "code":"course_code","course code":"course_code","subject code":"course_code",
    "name":"course_name","course name":"course_name","subject name":"course_name","title":"course_name",
    "credit":"credits","credits":"credits","credit hours":"credits",
    "category":"category","type":"category",
    "prereq":"prerequisites","prerequisite":"prerequisites","prerequisites":"prerequisites",
    "topics":"topics","syllabus":"topics","contents":"topics","units":"topics",
    "outcome":"outcomes","outcomes":"outcomes","learning outcomes":"outcomes","lo":"outcomes",
}

def canonicalize_headers(cols):
    can = []
    for c in cols:
        key = norm(c).lower()
        key = HEADER_MAP.get(key, key)
        can.append(key)
    return can

def read_all_sheets(xlsx_path):
    xl = pd.ExcelFile(xlsx_path)
    frames = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        if df.empty: 
            continue
        df.columns = canonicalize_headers(df.columns)
        df["__sheet"] = sheet
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def coerce_sem(val):
    s = norm(val)
    m = re.search(r"\d+", s)
    return int(m.group(0)) if m else None

def clean_row(r):
    return {
        "semester": coerce_sem(r.get("semester")),
        "course_code": norm(r.get("course_code")),
        "course_name": norm(r.get("course_name")),
        "credits": norm(r.get("credits")),
        "category": norm(r.get("category")),
        "prerequisites": norm(r.get("prerequisites")),
        "topics": norm(r.get("topics")),
        "outcomes": norm(r.get("outcomes")),
        "sheet": norm(r.get("__sheet")),
    }

def build_index(df):
    # keep rows that have either a name or code
    df = df[[c for c in df.columns if c in {
        "semester","course_code","course_name","credits","category",
        "prerequisites","topics","outcomes","__sheet"}]]
    rows = [clean_row(r) for r in df.to_dict("records")]
    rows = [r for r in rows if r["course_name"] or r["course_code"]]

    by_sem = {}
    for r in rows:
        sem = r["semester"] or 0
        by_sem.setdefault(str(sem), []).append({
            k:v for k,v in r.items() if k!="semester"
        })

    catalog = []
    for r in rows:
        code = r["course_code"] or slug(r["course_name"])
        catalog.append({
            "id": slug(f"{r['semester']}-{code}"),
            "semester": r["semester"],
            "code": r["course_code"],
            "name": r["course_name"],
            "credits": r["credits"],
            "category": r["category"],
        })

    return {"program":"UG-Finance","version":1,
            "semesters": by_sem,
            "catalog": catalog}

def main():
    if len(sys.argv) < 3:
        print("usage: python tools/ingest_finance_syllabus.py <input.xlsx> <output.json>")
        sys.exit(1)
    inp = pathlib.Path(sys.argv[1]).expanduser().resolve()
    outp = pathlib.Path(sys.argv[2]).expanduser().resolve()
    outp.parent.mkdir(parents=True, exist_ok=True)

    df = read_all_sheets(inp)
    if df.empty:
        print("No data found in workbook.")
        sys.exit(2)
    idx = build_index(df)
    outp.write_text(json.dumps(idx, indent=2), encoding="utf-8")
    print(f"Wrote {outp} (semesters: {len(idx['semesters'])}, courses: {len(idx['catalog'])})")

if __name__ == "__main__":
    main()
