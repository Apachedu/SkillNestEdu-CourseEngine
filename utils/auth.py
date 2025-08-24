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

