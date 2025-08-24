import streamlit as st
from utils.auth import validate, get_license_info
from ui_branding import sidebar_branding, page_watermark

st.set_page_config(page_title="SkillNestEdu — Login", layout="wide")

lic_email, boards, expiry = get_license_info()
sidebar_branding(lic_email, boards[0] if boards else "—", expiry)
page_watermark(lic_email, expiry)

st.title("Login")
email = st.text_input("Licensed Email")
password = st.text_input("Password", type="password")
board = st.selectbox("Board", boards if boards else ["IB"])

if st.button("Login", use_container_width=True):
    ok, msg = validate(email, password, board)
    if ok:
        st.session_state["auth_email"] = email
        st.session_state["auth_board"] = board
        st.success("Logged in. Use the sidebar ➜ Pages.")
    else:
        st.error(msg)
