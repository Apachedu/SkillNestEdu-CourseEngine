import json, hashlib
from typing import Tuple, Optional, List

LICENSE_PATH = ".license/license.json"

# ---- IO helpers ----
def _read_json(path: str) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}

def _write_json(path: str, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def _read_license() -> dict:
    return _read_json(LICENSE_PATH)

# ---- Crypto ----
def hash_password(salt: str, password: str) -> str:
    m = hashlib.sha256()
    m.update((salt + (password or "")).encode("utf-8"))
    return m.hexdigest()

# ---- Roles (discovered from license) ----
def get_roles() -> List[str]:
    lic = _read_license() or {}
    roles = list(lic.keys())
    # put admin first if present
    roles.sort(key=lambda r: (r != "admin", r))
    return roles

def _board_label_for_role(role: str) -> str:
    mapping = {
        "admin": "All Boards",
        "student_ib": "IB",
        "student_cbse": "CBSE",
        "student_icse": "ICSE",
        "student_ug": "UG",
        "student_ielts": "IELTS",
        "student_pte": "PTE",
        "student_softskills": "Soft Skills",
        "student_spokenenglish": "Spoken English",
    }
    return mapping.get(role, role.replace("_"," ").title())

# ---- Public API ----
def get_license_info(role: str) -> Tuple[str, str, str, Optional[str]]:
    """
    Returns (email, board_label, expiry, password_hash) for the selected role.
    """
    lic = _read_license() or {}
    acc = lic.get(role, {})
    email = acc.get("email", f"{role}@example.com")
    board_label = _board_label_for_role(role)
    expiry = acc.get("expiry", "2026-05-31")
    return email, board_label, expiry, acc.get("password_hash")

def verify_login(role: str, input_email: str, input_password: str) -> Tuple[bool, str]:
    lic = _read_license()
    if role not in lic:
        return False, f"Role '{role}' not licensed. Ask admin to add it in .license/license.json."

    acc = lic[role]
    if (input_email or "").strip().lower() != (acc.get("email","") or "").lower():
        return False, "Email not licensed for this role."

    exp = acc.get("expiry")
    if exp:
        try:
            import datetime
            if datetime.datetime.utcnow() > datetime.datetime.fromisoformat(exp):
                return False, f"License expired on {exp}."
        except Exception:
            pass

    salt = acc.get("salt", "skillnest-salt-v1")
    expected = acc.get("password_hash", "")
    if not expected:
        return False, "Password not set for this role."
    actual = hash_password(salt, input_password or "")
    if actual != expected:
        return False, "Incorrect password."
    return True, "Login successful."

def set_password(role: str, new_password: str) -> Tuple[bool, str]:
    """
    Update password for a given role in .license/license.json.
    (On Cloud, file writes won't persist after reboot—change locally and push.)
    """
    if not new_password or len(new_password) < 6:
        return False, "Choose a password with at least 6 characters."
    lic = _read_license()
    if role not in lic:
        return False, f"Role '{role}' not found in license."

    salt = lic[role].get("salt", "skillnest-salt-v1")
    lic[role]["password_hash"] = hash_password(salt, new_password)
    try:
        _write_json(LICENSE_PATH, lic)
        return True, f"Password updated for role '{role}'."
    except Exception as e:
        return False, f"Write failed: {e}"
