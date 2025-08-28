import streamlit as st
from utils.auth import verify_login
import sys, os
def logout_fix():
    import streamlit as st
    for k in [k for k in list(st.session_state.keys()) if k.startswith("auth_") or k in ("role","email","auth_board")]:
        st.session_state.pop(k, None)
    st.rerun()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

st.set_page_config(page_title="Login", layout="centered")

roles = ["student_ib","admin"]
role = st.selectbox("Role", roles, index=0)
default_email = "ib_student@skillnestedu.com" if role=="student_ib" else "admin@skillnestedu.com"

email = st.text_input("Email", value=default_email)
password = st.text_input("Password", type="password")

if st.button("Login", use_container_width=True):
    ok, msg = verify_login(role, email.strip(), password.strip())
    if ok:
        st.session_state["role"] = role
        st.session_state["email"] = email.strip()
        st.success("Login successful.")
        dest = "pages/01_Student_Mode.py" if role.startswith("student") else "pages/02_Creator_Mode.py"
        st.switch_page(dest)
    else:
        st.error(msg)

if "role" in st.session_state and st.button("Logout", use_container_width=True):
    for k in ("role","email"):
        st.session_state.pop(k, None)
    st.rerun()
