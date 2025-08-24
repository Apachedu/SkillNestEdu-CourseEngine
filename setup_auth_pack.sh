#!/bin/bash
set -e
echo "🔐 Installing Auth Pack (email+password, single seat, expiry, board)"

# 0) Folders
mkdir -p .license utils data pages boards/ib

# 1) Standards (ensure expiry source exists)
if [ ! -f boards/ib/standards.json ]; then
  cat > boards/ib/standards.json <<'JSON'
{
  "paper_map": { "short_answer_marks": [2,4], "long_answer_marks": [10,15,20], "numericals": false },
  "command_terms": ["define","explain","analyse","evaluate","discuss"],
  "expiry": "2026-05-31"
}
JSON
  echo "🗂️  wrote boards/ib/standards.json"
fi

# 2) Default license (single seat) with salted hash placeholder
cat > .license/license.json <<'JSON'
{
  "email": "contact@skillnestedu.com",
  "salt": "skillnest-salt-v1",
  "password_hash": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
  "board": "IB",
  "expiry": "2026-05-31"
}
JSON
# Note: password_hash = sha256(salt + password). Default (empty) until first set.

# 3) Utils: auth helpers
cat > utils/auth.py <<'PY'
import json, hashlib, time, os
from typing import Tuple, List, Optional

LICENSE_PATH = ".license/license.json"
STANDARDS_PATH = "boards/ib/standards.json"

def _read_json(path: str) -> dict:
    try:
        with open(path, "r") as f: return json.load(f)
    except Exception: return {}

def _write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f: json.dump(obj, f, indent=2)

def hash_password(salt: str, password: str) -> str:
    m = hashlib.sha256()
    m.update((salt + (password or "")).encode("utf-8"))
    return m.hexdigest()

def get_license_info() -> Tuple[str, List[str], Optional[int]]:
    lic = _read_json(LICENSE_PATH)
    std = _read_json(STANDARDS_PATH)
    email = lic.get("email", "contact@skillnestedu.com")
    boards = [lic.get("board", "IB")]
    # expiry priority: license.expiry, else standards.expiry
    exp = lic.get("expiry") or std.get("expiry")
    # return unix ts (None if not parseable)
    try:
        import datetime
        ts = int(time.mktime(datetime.datetime.fromisoformat(exp).timetuple())) if exp else None
    except Exception:
        ts = None
    return email, boards, ts

def verify_login(input_email: str, input_password: str) -> Tuple[bool, str]:
    lic = _read_json(LICENSE_PATH)
    if not lic:
        return False, "License file missing."
    if input_email.strip().lower() != lic.get("email","").lower():
        return False, "Email not licensed."
    # expiry check
    exp = lic.get("expiry")
    if exp:
        try:
            import datetime
            exp_dt = datetime.datetime.fromisoformat(exp)
            if datetime.datetime.utcnow() > exp_dt:
                return False, f"License expired on {exp}."
        except Exception:
            pass
    # password check
    salt = lic.get("salt","skillnest-salt-v1")
    expected = lic.get("password_hash","")
    actual = hash_password(salt, input_password or "")
    if expected == "" or expected == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855":
        return False, "Password not set. Use 'Set/Reset Password' below."
    if actual != expected:
        return False, "Incorrect password."
    return True, "Login successful."

def set_password(new_password: str) -> Tuple[bool, str]:
    if not new_password or len(new_password) < 6:
        return False, "Choose a password with at least 6 characters."
    lic = _read_json(LICENSE_PATH)
    if not lic: return False, "License file missing."
    salt = lic.get("salt","skillnest-salt-v1")
    lic["password_hash"] = hash_password(salt, new_password)
    _write_json(LICENSE_PATH, lic)
    return True, "Password set."

PY

# 4) Pages: Login
cat > pages/00_Login.py <<'PY'
import streamlit as st
from utils.auth import get_license_info, verify_login, set_password

st.set_page_config(page_title="SkillNestEdu — Login", layout="wide")
lic_email, boards, expiry = get_license_info()

st.title("Login")
with st.form("login_form", clear_on_submit=False):
    email = st.text_input("Licensed Email", value=lic_email)
    password = st.text_input("Password", type="password")
    board = st.selectbox("Board", boards or ["IB"])
    submit = st.form_submit_button("Login", use_container_width=True)

if submit:
    ok, msg = verify_login(email, password)
    if ok:
        st.session_state["auth_email"] = email
        st.session_state["auth_board"] = board
        st.success("Welcome! Use the sidebar to open Student Mode / Creator Mode / Booking.")
    else:
        st.error(msg)

st.divider()
st.subheader("Set/Reset Password (Admin)")
with st.form("set_pw", clear_on_submit=True):
    new_pw = st.text_input("New Password", type="password")
    set_btn = st.form_submit_button("Set/Reset Password")
    if set_btn:
        ok, msg = set_password(new_pw)
        st.success(msg) if ok else st.error(msg)

# Helper text
st.caption("This is a single-seat license. To change the licensed email or expiry, edit .license/license.json.")
PY

# 5) Ensure Attempts log exists for Student Mode
if [ ! -f data/attempts.csv ]; then
  echo "timestamp,email,subject,level,board,topic,qtype,max_marks,score,feedback" > data/attempts.csv
fi

echo "✅ Auth Pack installed."
echo "• Default licensed email: contact@skillnestedu.com"
echo "• First-time step: go to Pages → Login → Set/Reset Password (choose one), then Login."
