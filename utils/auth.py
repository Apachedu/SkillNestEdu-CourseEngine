import json, os, datetime
import streamlit as st

LICENSE_PATH = ".license/license.json"

def load_license():
    if not os.path.exists(LICENSE_PATH):
        st.stop()
    with open(LICENSE_PATH, "r") as f:
        return json.load(f)

def validate(email:str, password:str, board:str):
    lic = load_license()
    if email != lic.get("email") or password != lic.get("password"):
        return False, "Invalid credentials."
    if board not in lic.get("boards", []):
        return False, f"Board '{board}' not permitted."
    expiry = lic.get("expiry")
    if expiry:
        try:
            if datetime.date.today() > datetime.date.fromisoformat(expiry):
                return False, "License expired."
        except Exception:
            return False, "License expiry invalid."
    return True, "OK"

def get_license_info():
    lic = load_license()
    return lic.get("email"), lic.get("boards", []), lic.get("expiry")
