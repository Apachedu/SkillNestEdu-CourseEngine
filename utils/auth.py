import os, json, hashlib
from typing import Tuple, List, Optional

try:
    import streamlit as st  # type: ignore
except Exception:
    class _Dummy: secrets = {}
    st = _Dummy()  # type: ignore

def _get_block(role: str) -> dict:
    try:
        if hasattr(st, "secrets") and role in st.secrets:
            return dict(st.secrets[role])
    except Exception:
        pass
    return {}

def get_roles() -> List[str]:
    roles = []
    try:
        for k in st.secrets.keys():
            if k == "admin" or k.startswith("student_"):
                roles.append(k)
    except Exception:
        pass
    return sorted(roles)

def get_license_info(role: str = "admin") -> Tuple[str, List[str], Optional[int]]:
    blk = _get_block(role) or {}
    email = (blk.get("email") or "contact@skillnestedu.com")
    boards = [blk.get("role") or role]
    expiry = blk.get("expiry")
    ts = None
    if expiry:
        try:
            import datetime
            ts = int(datetime.datetime.fromisoformat(expiry).timestamp())
        except Exception:
            ts = None
    return email, boards, ts

def verify_login(role: str, input_email: str, input_password: str) -> Tuple[bool, str]:
    blk = _get_block(role)
    if not blk:
        return False, f"Role [{role}] not configured."
    email_in = (input_email or "").strip().lower()
    email_ok = (blk.get("email") or "").strip().lower()
    if email_in != email_ok:
        return False, "Email not licensed."
    exp = (blk.get("expiry") or "").strip()
    if exp:
        try:
            import datetime
            if datetime.datetime.utcnow() > datetime.datetime.fromisoformat(exp):
                return False, f"License expired on {exp}."
        except Exception:
            pass
    salt = blk.get("salt", "skillnest-salt-v1")
    expected = blk.get("password_hash") or ""
    if not expected:
        return False, "Password not set."
    actual = hashlib.sha256((salt + (input_password or "").strip()).encode()).hexdigest()
    if actual != expected:
        return False, "Incorrect password."
    return True, "Login successful."
