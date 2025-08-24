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
