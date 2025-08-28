import streamlit as st, json, glob, os
from ui_branding import sidebar_branding, page_watermark
from utils.auth import get_license_info
import sys, os
def logout_fix():
    import streamlit as st
    for k in [k for k in list(st.session_state.keys()) if k.startswith("auth_") or k in ("role","email","auth_board")]:
        st.session_state.pop(k, None)
    st.rerun()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


st.set_page_config(page_title="SkillNestEdu — Sample Units", layout="wide")

if "email" not in st.session_state:
    st.warning("Please login first (Pages ➜ Login)."); st.stop()

lic_email, boards, expiry = get_license_info()
sidebar_branding(lic_email, boards[0] if boards else "IB", expiry); page_watermark(lic_email, expiry)

st.title("Sample Units")
cols = st.columns(2)

def render_card(path):
    data = json.load(open(path))
    with st.container(border=True):
        st.subheader(f"{data['subject']} — {data['topic']}")
        st.caption(f"Level: {data.get('level','')} • Difficulty: {data.get('difficulty','—')}")
        st.write(data["study_guide"]["intro"])
        if st.button(f"Prefill Student Mode with '{data['topic']}'", key=path):
            st.session_state["prefill_subject"] = data["subject"]
            st.session_state["prefill_level"] = data.get("level","SL")
            st.session_state["prefill_board"] = "IB"
            st.session_state["prefill_topic"] = data["topic"]
            st.success("Open Pages ➜ Student Mode and click Generate.")
        with st.expander("Open details"):
            st.write("**Key Points**")
            for k in data["study_guide"]["key_points"]:
                st.write("• " + k)
            st.write("**Sample MCQ**")
            for q in data["practice"].get("mcq", [])[:2]:
                st.write("• " + q["stem"])

paths = sorted(glob.glob("content/ib_econ/*.json")) + sorted(glob.glob("content/ib_math_aa/*.json"))
for i,p in enumerate(paths):
    with cols[i%2]:
        render_card(p)
