import streamlit as st
from utils.auth import get_license_info, verify_login, get_roles

st.set_page_config(page_title="SkillNestEdu — Login", layout="wide")
st.title("Login")

ROLES = get_roles()
role = st.selectbox("Role", ROLES, index=(ROLES.index("student_ib") if "student_ib" in ROLES else 0))
lic_email, board_label, expiry, _ = get_license_info(role)

with st.form("login_form"):
    email = st.text_input("Licensed Email", value=lic_email)
    password = st.text_input("Password", type="password")
    submit = st.form_submit_button("Login", use_container_width=True)

if submit:
    ok, msg = verify_login(role, email, password)
    if ok:
        st.session_state["auth_email"] = email
        st.session_state["auth_role"] = role
        st.session_state["auth_board"] = board_label
        st.success(f"Welcome {role.replace('_',' ').title()}! Open the sidebar pages.")
    else:
        st.error(msg)

st.caption("Admins unlock Creator/Booking; Students only see Student Mode.")
