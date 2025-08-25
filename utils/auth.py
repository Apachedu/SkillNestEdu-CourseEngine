import json, hashlib, os
from typing import Tuple, List, Optional

# Streamlit guarded import (so local linters don't break)
try:
    import streamlit as st  # type: ignore
except Exception:
    class _Dummy: secrets = {}
    st = _Dummy()  # type: ignore

LICENSE_PATH = ".license/license.json"

def _read_json(path: str) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _write_json(path: str, obj: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def hash_password(salt: str, password: str) -> str:
    m = hashlib.sha256()
    m.update((salt + (password or "")).encode("utf-8"))
    return m.hexdigest()

def _read_license() -> dict:
    """
    If FORCE_FILE_LICENSE=1 -> use file.
    Else try Secrets [license], then fallback to file.
    """
    if os.getenv("FORCE_FILE_LICENSE", "0") == "1":
        return _read_json(LICENSE_PATH)
    try:
        if hasattr(st, "secrets") and "license" in st.secrets:
            return dict(st.secrets["license"])
    except Exception:
        pass
    return _read_json(LICENSE_PATH)

def get_license_info() -> Tuple[str, List[str], Optional[int]]:
    lic = _read_license() or {}
    email = lic.get("email", "contact@skillnestedu.com")
    boards = [lic.get("board", "IB")]
    expiry = lic.get("expiry") or "2026-05-31"
    try:
        import datetime
        ts = int(datetime.datetime.fromisoformat(expiry).timestamp())
    except Exception:
        ts = None
    return email, boards, ts

def verify_login(input_email: str, input_password: str) -> Tuple[bool, str]:
    lic = _read_license()
    if not lic:
        return False, "License not found. Add .license/license.json (committed) or Secrets [license]."
    if (input_email or "").strip().lower() != (lic.get("email","") or "").lower():
        return False, "Email not licensed."
    # expiry
    exp = lic.get("expiry")
    if exp:
        try:
            import datetime
            if datetime.datetime.utcnow() > datetime.datetime.fromisoformat(exp):
                return False, f"License expired on {exp}."
        except Exception:
            pass
    # password
    salt = lic.get("salt", "skillnest-salt-v1")
    expected = lic.get("password_hash", "")
    if not expected:
        return False, "Password not set. In Cloud use Secrets; in local use Set/Reset."
    actual = hash_password(salt, input_password or "")
    if actual != expected:
        return False, "Incorrect password."
    return True, "Login successful."

def set_password(new_password: str) -> Tuple[bool, str]:
    """
    Setter works only in file mode (not Secrets).
    """
    # If secrets present, we don’t mutate at runtime
    try:
        if os.getenv("FORCE_FILE_LICENSE", "0") != "1" and hasattr(st, "secrets") and "license" in st.secrets:
            return False, "Secrets mode: update [license].password_hash in Secrets."
    except Exception:
        pass
    if not new_password or len(new_password) < 6:
        return False, "Choose a password with at least 6 characters."
    lic = _read_json(LICENSE_PATH)
    if not lic:
        return False, "License file missing."
    salt = lic.get("salt", "skillnest-salt-v1")
    lic["password_hash"] = hash_password(salt, new_password)
    _write_json(LICENSE_PATH, lic)
    return True, "Password set."
