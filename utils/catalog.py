import json, pathlib

_FIN_IDX = pathlib.Path("data/finance_index.json")

def load_finance_index():
    if not _FIN_IDX.exists():
        return {"semesters": {}, "catalog": []}
    with _FIN_IDX.open("r", encoding="utf-8") as f:
        return json.load(f)
