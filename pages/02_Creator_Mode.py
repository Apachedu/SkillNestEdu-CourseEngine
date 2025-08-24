import streamlit as st
from ui_branding import sidebar_branding, page_watermark
from engine_v2 import generate_package
from utils.auth import get_license_info

st.set_page_config(page_title="SkillNestEdu — Creator Mode", layout="wide")

if "auth_email" not in st.session_state:
    st.warning("Please login first (Pages ➜ Login)."); st.stop()

auth_email = st.session_state["auth_email"]
auth_board = st.session_state.get("auth_board", "IB")
lic_email, boards, expiry = get_license_info()
sidebar_branding(lic_email, auth_board, expiry); page_watermark(lic_email, expiry)

st.title("Creator Mode")
subject = st.selectbox("Subject", ["IB Economics","IB Math AA"])
level = st.selectbox("Level", ["SL","HL"])
topic = st.text_input("Topic", value="Price Elasticity of Demand")

if st.button("Generate"):
    pkg = generate_package(subject, level, "IB", topic)
    st.success("Package generated")
    st.subheader("Carousel Points")
    for s in pkg["study_guide"]["key_points"]: st.write("• " + s)
    st.subheader("Reel Script")
    st.write(f"Hook → 3 steps → CTA about {topic}")
    st.subheader("Blog Outline")
    st.write("Intro → Concept → Examples → Practice → Summary")
