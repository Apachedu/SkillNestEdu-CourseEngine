import streamlit as st, subprocess, sys, json, os
from utils.auth import verify_login
import sys, os
def logout_fix():
    import streamlit as st
    for k in [k for k in list(st.session_state.keys()) if k.startswith("auth_") or k in ("role","email","auth_board")]:
        st.session_state.pop(k, None)
    st.rerun()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


st.set_page_config(page_title="Admin Tools", layout="wide")
role = st.session_state.get("role","")
email = st.session_state.get("email","")

if role.lower() != "admin":
    st.error("Admins only."); st.stop()

st.success(f"Logged in as {email} • Role: Admin")

colA, colB, colC = st.columns(3)

with colA:
    if st.button("Ingest PDFs", use_container_width=True):
        r = subprocess.run([sys.executable, "tools/content_bank.py", "ingest_all"], capture_output=True, text=True)
        st.code(r.stdout or r.stderr)
with colB:
    if st.button("Build Index", use_container_width=True):
        r = subprocess.run([sys.executable, "tools/content_bank.py", "build_index_all"], capture_output=True, text=True)
        st.code(r.stdout or r.stderr)
with colC:
    if st.button("Show Sources", use_container_width=True):
        roots = []
        for p in [
            "source_pdfs",
            os.path.expanduser("~/Library/CloudStorage/GoogleDrive-contact@skillnestedu.com"),
            os.path.expanduser("~/Library/CloudStorage/GoogleDrive-eduskillnest@gmail.com"),
        ]:
            if os.path.exists(p): roots.append(p)
        st.write(roots if roots else "No source roots found.")

st.divider()
q = st.text_input("Test retrieval (enter topic/keywords)", value="Price Elasticity of Demand")
if st.button("Search Now"):
    r = subprocess.run([sys.executable, "tools/content_bank.py", "search", q], capture_output=True, text=True)
    st.code(r.stdout or r.stderr)
