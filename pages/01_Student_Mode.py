import streamlit as st
from ui_branding import sidebar_branding, page_watermark
from engine_v2 import generate_package

st.set_page_config(page_title="SkillNestEdu — Student Mode", layout="wide")

# ------- Auth guard (students + admin allowed) -------
role = st.session_state.get("auth_role")
if role not in ("admin", "student_ib", "student_cbse", "student_icse", "student_ug",
                "student_ielts", "student_pte", "student_softskills", "student_spokenenglish"):
    st.warning("Please login first (Pages ➜ Login).")
    st.stop()

# ------- Header status + logout -------
def logout():
    for k in [k for k in st.session_state.keys() if k.startswith("auth_")]:
        st.session_state.pop(k, None)
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()  # older Streamlit fallback

with st.container(border=True):
    st.write(f"🔐 Logged in as **{st.session_state.get('auth_email','?')}** "
             f"• Role: **{st.session_state.get('auth_role','?').replace('_',' ').title()}** "
             f"• Board: **{st.session_state.get('auth_board','—')}**")
    st.button("Logout", on_click=logout)

# ------- Sidebar branding -------
EMAIL = st.session_state.get("auth_email", "contact@skillnestedu.com")
BOARD_LABEL = st.session_state.get("auth_board", "IB")
EXPIRY_TS = None  # optional; can be wired later if you want a timestamp
sidebar_branding(EMAIL, BOARD_LABEL, EXPIRY_TS)
page_watermark(EMAIL, EXPIRY_TS)

# ------- Sidebar inputs -------
subject = st.sidebar.selectbox("Subject", ["IB Economics","IB Math AA"])
level   = st.sidebar.selectbox("Level", ["SL","HL"])
board   = st.sidebar.selectbox("Board", ["IB"])
topic   = st.sidebar.text_input("Topic", value=st.session_state.get("prefill_topic","Price Elasticity of Demand"))
difficulty = st.sidebar.selectbox("Difficulty", ["Beginner","Advanced"])

# ------- UI -------
st.title("Student Mode")
if st.button("Generate Study Package", use_container_width=True):
    st.session_state["pkg"] = generate_package(subject, level, board, topic, difficulty)

pkg = st.session_state.get("pkg")
if not pkg:
    st.info("Enter a topic and click **Generate Study Package**.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Study Guide","Interactive","Case Study","Practice","Review"])

with tab1:
    st.subheader("Intro")
    st.write(pkg["study_guide"]["intro"])
    st.subheader("What is it?")
    for p in pkg["study_guide"]["concept"]: st.write("• " + p)
    st.subheader("Examples")
    for e in pkg["study_guide"]["examples"]: st.write("• " + e)
    st.subheader("Worked Example")
    for s in pkg["worked_example"]["steps"]: st.write("- " + s)
    st.success("Answer: " + pkg["worked_example"]["answer"])

with tab2:
    st.subheader("Interactive")
    st.caption("Your interactive diagrams will appear here.")

with tab3:
    st.subheader("Case Study (Sim)")
    st.caption("Registry wired; paste your sim URLs into interactive/case_studies/registry.json.")
    st.code("https://skillnestcasestudysim-d6355.web.app/sim?caseId=nike_outsourcing")

with tab4:
    st.subheader("Practice")
    st.markdown("**MCQ**")
    for i,q in enumerate(pkg["practice"]["mcq"],1):
        choice = st.radio(q["stem"], q["options"], key=f"mcq{i}")
        if st.button(f"Submit MCQ {i}"):
            st.write("✅ Correct" if choice==q["answer"] else f"❌ Answer: {q['answer']}")
    st.markdown("**SAQ (4)**")
    st.text_area("Explain …", key="saq")
    st.markdown("**LAQ (15)**")
    st.text_area("Evaluate …", key="laq")

with tab5:
    st.subheader("Revision")
    for n in pkg["revision"]["notes"]: st.write("• " + n)
    st.write("**Exam prep:**")
    for step in pkg["revision"]["exam_prep"]: st.write("- " + step)
