import os, csv, datetime, pathlib
DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
ATTEMPTS_CSV = DATA_DIR / "attempts.csv"
BOOKINGS_CSV = DATA_DIR / "bookings.csv"

def _append_row(path, header, rowdict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists: w.writeheader()
        w.writerow(rowdict)

def log_attempt(role, email, subject, topic, qtype, payload, score=None, feedback=None):
    row = {
        "ts": datetime.datetime.utcnow().isoformat()+"Z",
        "role": role, "email": email, "subject": subject, "topic": topic,
        "qtype": qtype, "payload": payload, "score": "" if score is None else score,
        "feedback": "" if feedback is None else feedback
    }
    _append_row(ATTEMPTS_CSV, list(row.keys()), row)

def log_booking(name, email, slot_iso, notes, price="500", minutes="45"):
    row = {
        "ts": datetime.datetime.utcnow().isoformat()+"Z",
        "name": name, "email": email, "slot_iso": slot_iso, "minutes": minutes, "price": price, "notes": notes
    }
    _append_row(BOOKINGS_CSV, list(row.keys()), row)
