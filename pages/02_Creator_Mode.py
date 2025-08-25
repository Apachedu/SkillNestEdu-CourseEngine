import streamlit as st
from ui_branding import sidebar_branding, page_watermark
from engine_v2 import generate_package

st.set_page_config(page_title="SkillNestEdu — Creator Mode", layout="wide")

role = st.session_state.get("auth_role")
if role != "admin":
    st.error("Admins only. Please login as Admin (Pages → Login)."); st.stop()

def logout():
    for k in [k for k in st.session_state.keys() if k.startswith("auth_")]: st.session_state.pop(k, None)
    try: st.rerun()
    except Exception: st.experimental_rerun()

with st.container(border=True):
    st.write(f"🔐 Logged in as **{st.session_state.get('auth_email','?')}** • Role: **{st.session_state.get('auth_role','?').replace('_',' ').title()}**")
    st.button("Logout", on_click=logout)

EMAIL = st.session_state.get("auth_email", "admin@skillnestedu.com")
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
