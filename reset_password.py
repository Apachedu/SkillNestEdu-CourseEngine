#!/usr/bin/env python3
import os, sys, json, hashlib, re
from datetime import datetime

LICENSE_PATH = ".license/license.json"
SALT = "skillnest-salt-v1"

def load_license() -> dict:
    if not os.path.exists(LICENSE_PATH):
        print(f"❌ License file not found at {LICENSE_PATH}")
        sys.exit(1)
    with open(LICENSE_PATH, "r") as f:
        return json.load(f)

def save_license(lic: dict):
    os.makedirs(os.path.dirname(LICENSE_PATH), exist_ok=True)
    with open(LICENSE_PATH, "w") as f:
        json.dump(lic, f, indent=2)
    print(f"✅ Updated {LICENSE_PATH}")

def hash_password(password: str) -> str:
    m = hashlib.sha256()
    m.update((SALT + password).encode("utf-8"))
    return m.hexdigest()

def valid_date(s: str) -> bool:
    try:
        datetime.fromisoformat(s)
        return True
    except Exception:
        return False

def list_accounts(lic: dict):
    rows = [(r, a.get("email",""), a.get("expiry","")) for r,a in lic.items()]
    if not rows:
        print("No roles found."); return
    w1 = max(len("role"), max(len(r[0]) for r in rows))
    w2 = max(len("email"), max(len(r[1]) for r in rows))
    w3 = max(len("expiry"), max(len(r[2]) for r in rows))
    print(f"{'role'.ljust(w1)}  {'email'.ljust(w2)}  {'expiry'.ljust(w3)}")
    print(f"{'-'*w1}  {'-'*w2}  {'-'*w3}")
    for r in rows:
        print(f"{r[0].ljust(w1)}  {r[1].ljust(w2)}  {r[2].ljust(w3)}")

def set_password(lic: dict, role: str, new_pw: str):
    if role not in lic:
        print(f"❌ Role '{role}' not found"); sys.exit(1)
    if len(new_pw) < 6:
        print("❌ Password must be at least 6 characters"); sys.exit(1)
    salt = lic[role].get("salt", SALT)
    lic[role]["salt"] = salt
    lic[role]["password_hash"] = hash_password(new_pw)
    save_license(lic)
    print(f"🔑 Password for '{role}' updated to: {new_pw}")

def set_email(lic: dict, role: str, new_email: str):
    if role not in lic:
        print(f"❌ Role '{role}' not found"); sys.exit(1)
    if "@" not in new_email:
        print("❌ Invalid email"); sys.exit(1)
    lic[role]["email"] = new_email
    save_license(lic)
    print(f"📧 Email for '{role}' updated to: {new_email}")

def set_expiry(lic: dict, role: str, new_expiry: str):
    if role not in lic:
        print(f"❌ Role '{role}' not found"); sys.exit(1)
    if not valid_date(new_expiry):
        print("❌ Expiry must be ISO date, e.g. 2026-05-31"); sys.exit(1)
    lic[role]["expiry"] = new_expiry
    save_license(lic)
    print(f"🗓️ Expiry for '{role}' updated to: {new_expiry}")

def add_role(lic: dict, role: str, email: str, password: str, expiry: str):
    if role in lic:
        print(f"❌ Role '{role}' already exists"); sys.exit(1)
    if "@" not in email:
        print("❌ Invalid email"); sys.exit(1)
    if len(password) < 6:
        print("❌ Password must be at least 6 characters"); sys.exit(1)
    if not valid_date(expiry):
        print("❌ Expiry must be ISO date, e.g. 2026-05-31"); sys.exit(1)
    lic[role] = {
        "email": email,
        "salt": SALT,
        "password_hash": hash_password(password),
        "expiry": expiry
    }
    save_license(lic)
    print(f"➕ Added role '{role}' with email {email}")

def del_role(lic: dict, role: str):
    if role not in lic:
        print(f"❌ Role '{role}' not found"); sys.exit(1)
    if role == "admin":
        print("❌ Refusing to delete 'admin' role"); sys.exit(1)
    lic.pop(role)
    save_license(lic)
    print(f"🗑️ Deleted role '{role}'")

def help_text():
    print("Usage:")
    print("  python reset_password.py                # list roles")
    print("  python reset_password.py setpass <role> <new_password>")
    print("  python reset_password.py setemail <role> <new_email>")
    print("  python reset_password.py setexpiry <role> <YYYY-MM-DD>")
    print("  python reset_password.py addrole <role> <email> <password> <YYYY-MM-DD>")
    print("  python reset_password.py delrole <role>")

if __name__ == "__main__":
    if not os.path.exists(LICENSE_PATH):
        print(f"❌ License file not found at {LICENSE_PATH}")
        sys.exit(1)

    lic = load_license()
    if len(sys.argv) == 1:
        list_accounts(lic); sys.exit(0)

    cmd = sys.argv[1].lower()
    if cmd == "setpass" and len(sys.argv) == 4:
        set_password(lic, sys.argv[2], sys.argv[3]); sys.exit(0)
    if cmd == "setemail" and len(sys.argv) == 4:
        set_email(lic, sys.argv[2], sys.argv[3]); sys.exit(0)
    if cmd == "setexpiry" and len(sys.argv) == 4:
        set_expiry(lic, sys.argv[2], sys.argv[3]); sys.exit(0)
    if cmd == "addrole" and len(sys.argv) == 6:
        add_role(lic, sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]); sys.exit(0)
    if cmd == "delrole" and len(sys.argv) == 3:
        del_role(lic, sys.argv[2]); sys.exit(0)

    help_text(); sys.exit(1)
