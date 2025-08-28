import streamlit as st
from ui_branding import sidebar_branding, page_watermark
from engine_v2 import generate_package
import sys, os
def logout_fix():
    import streamlit as st
    for k in [k for k in list(st.session_state.keys()) if k.startswith("auth_") or k in ("role","email","auth_board")]:
        st.session_state.pop(k, None)
    st.rerun()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


st.set_page_config(page_title="SkillNestEdu — Creator Mode", layout="wide")

role = st.session_state.get("role")
if role != "admin":
    st.error("Admins only. Please login as Admin (Pages → Login)."); st.stop()

def logout():
    for k in [k for k in st.session_state.keys() if k.startswith("auth_")]: st.session_state.pop(k, None)
    try: st.rerun()
    except Exception: st.rerun()

with st.container(border=True):
    st.write(f"🔐 Logged in as **{st.session_state.get('email','?')}** • Role: **{st.session_state.get('role','?').replace('_',' ').title()}**")
    st.button("Logout", key="logout_02_creator_mode", on_click=logout_fix)
EMAIL = st.session_state.get("email", "admin@skillnestedu.com")
sidebar_branding(EMAIL, "All Boards", None); page_watermark(EMAIL, None)

st.title("Creator Mode")
subject = st.selectbox("Subject", ["IB Economics","IB Math AA","IELTS","PTE","Soft Skills","Spoken English"])
level = st.selectbox("Level", ["SL","HL","General"])
topic = st.text_input("Topic")
if st.button("Generate"):
    pkg = generate_package(subject, level, "Any", topic)
    st.success("Package generated")
    st.subheader("Carousel Points")
    for s in pkg["study_guide"]["key_points"]: st.write("• " + s)
    st.subheader("Reel Script"); st.write(f"Hook → 3 steps → CTA about {topic}")
    st.subheader("Blog Outline"); st.write("Intro → Concept → Examples → Practice → Summary")
