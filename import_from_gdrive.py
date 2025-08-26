#!/usr/bin/env python3
import os, sys, shutil, pathlib, fnmatch, subprocess

PROJECT = pathlib.Path(".").resolve()
DEST = PROJECT / "source_pdfs"
DEFAULT_DRIVE_ROOTS = [
    pathlib.Path.home() / "Library/CloudStorage",
    pathlib.Path.home() / "Google Drive",
]

MAPPING = {
    "ib_econ":   ("IB", "Economics"),
    "ib_bm":     ("IB", "Business Management"),
    "ib_math":   ("IB", "Math AA"),
    "ib_cs":     ("IB", "Computer Science"),
    "ib_phys":   ("IB", "Physics"),
    "ib_chem":   ("IB", "Chemistry"),
    "ib_bio":    ("IB", "Biology"),
    "ug_fin":    ("UG", "Finance"),
    "cbse_econ": ("CBSE","Economics"),
    "icse_econ": ("ICSE","Economics"),
}

def find_drive_roots():
    roots = []
    for base in DEFAULT_DRIVE_ROOTS:
        if base.exists():
            for p in base.iterdir():
                try:
                    if p.is_dir() and "GoogleDrive" in p.name or "Drive" in p.name or "My Drive" in p.name:
                        roots.append(p)
                except Exception:
                    pass
    # Also consider “My Drive” under top-level matches
    for base in DEFAULT_DRIVE_ROOTS:
        md = base / "My Drive"
        if md.exists():
            roots.append(md)
    return list(dict.fromkeys(roots))

def copy_pdf(src_path: pathlib.Path, board: str, subject: str):
    dest_dir = DEST / board / subject
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dest_dir / src_path.name)
    return str(dest_dir / src_path.name)

def main():
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python import_from_gdrive.py <pattern> <target_key>")
        print("  where <pattern> is a filename pattern like '*Tragakes*.pdf'")
        print("  and <target_key> is one of:", ", ".join(MAPPING.keys()))
        sys.exit(1)

    pattern = sys.argv[1]
    key = sys.argv[2]
    if key not in MAPPING:
        print("Unknown target_key. Choose from:", ", ".join(MAPPING.keys()))
        sys.exit(1)

    board, subject = MAPPING[key]
    roots = find_drive_roots()
    if not roots:
        print("Could not find Google Drive mount. Try passing absolute paths or mount Drive for desktop.")
        sys.exit(1)

    matches = []
    for root in roots:
        for path, _, files in os.walk(root):
            for name in files:
                if fnmatch.fnmatch(name, pattern) and name.lower().endswith(".pdf"):
                    matches.append(pathlib.Path(path) / name)

    if not matches:
        print("No PDFs matched your pattern in detected Drive roots:")
        for r in roots: print(str(r))
        sys.exit(1)

    copied = []
    for m in matches:
        copied.append(copy_pdf(m, board, subject))
    print("Copied:")
    for c in copied: print(c)

    print("Ingesting and indexing…")
    subprocess.check_call([sys.executable, "tools/content_bank.py", "ingest_all"])
    subprocess.check_call([sys.executable, "tools/content_bank.py", "build_index_all"])
    print("Done.")

if __name__ == "__main__":
    main()

