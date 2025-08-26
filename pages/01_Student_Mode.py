import streamlit as st
from ui_branding import sidebar_branding, page_watermark
from engine_v2 import generate_package
from utils.retrieval import search_snippets
from utils.storage import log_attempt
from utils.feedback import grade_mcq, grade_saq, grade_laq

st.set_page_config(page_title="SkillNestEdu — Student Mode", layout="wide")

# ---- Auth gate ----
role = st.session_state.get("role")
if role not in (
    "admin", "student_ib", "student_cbse", "student_icse", "student_ug",
    "student_ielts", "student_pte", "student_softskills", "student_spokenenglish"
):
    st.warning("Please login first (Pages → Login).")
    st.stop()

def logout():
    for k in [k for k in st.session_state.keys() if k.startswith("auth_")]:
        st.session_state.pop(k, None)
    try:
        st.rerun()
    except Exception:
        st.experimental_rerun()

with st.container(border=True):
    st.write(
        f"🔐 Logged in as **{st.session_state.get('email','?')}** • "
        f"Role: **{st.session_state.get('role','?').replace('_',' ').title()}** • "
        f"Board: **{st.session_state.get('auth_board','—')}**"
    )
    st.button("Logout", on_click=logout)

# ---- Branding ----
EMAIL = st.session_state.get("email", "student@skillnestedu.com")
BOARD_LABEL = st.session_state.get("auth_board", "IB")
sidebar_branding(EMAIL, BOARD_LABEL, None)
page_watermark(EMAIL, None)

# ---- Sidebar controls ----
subject = st.sidebar.selectbox(
    "Subject", ["IB Economics", "IB Math AA", "IELTS", "PTE", "Soft Skills", "Spoken English"]
)
level = st.sidebar.selectbox("Level", ["SL", "HL", "General"])
board = st.sidebar.selectbox(
    "Board", ["IB", "CBSE", "ICSE", "UG", "IELTS", "PTE", "Soft Skills", "Spoken English"]
)
topic = st.sidebar.text_input(
    "Topic", value=st.session_state.get("prefill_topic", "Price Elasticity of Demand")
)
difficulty = st.sidebar.selectbox("Difficulty", ["Beginner", "Advanced"])

# ---- Generate ----
st.title("Student Mode")
if st.button("Generate Study Package", use_container_width=True):
    st.session_state["pkg"] = generate_package(subject, level, board, topic, difficulty)

pkg = st.session_state.get("pkg")
if not pkg:
    st.info("Enter a topic and click **Generate Study Package**.")
    st.stop()

# ---- Tabs ----
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Study Guide", "Interactive", "Case Study", "Practice", "Review"])

with tab1:
    st.subheader("Intro")
    st.write(pkg["study_guide"]["intro"])

    st.subheader("What is it?")
    for p in pkg["study_guide"]["concept"]:
        st.write("• " + p)

    st.subheader("Citations")
    hits = search_snippets(topic)
    if not hits:
        st.caption(
            "No matching sources yet. Use **Admin Tools → Ingest PDFs** and **Build Index**, then try again."
        )
    else:
        for h in hits:
            st.write(f"• {h['source']} — p.{h['page']}")

    st.subheader("Examples")
    for e in pkg["study_guide"]["examples"]:
        st.write("• " + e)

    st.subheader("Worked Example")
    for s in pkg["worked_example"]["steps"]:
        st.write("- " + s)
    st.success("Answer: " + pkg["worked_example"]["answer"])

with tab2:
    st.subheader("Interactive")
    st.caption("Interactive diagrams coming soon (deterministic).")

with tab3:
    st.subheader("Case Study (Sim)")
    st.caption("Registry wired; paste your sim URLs into interactive/case_studies/registry.json.")
    st.code("https://skillnestcasestudysim-d6355.web.app/sim?caseId=nike_outsourcing")

with tab4:
    st.subheader("Practice")

    st.markdown("**MCQ**")
    for i, q in enumerate(pkg["practice"]["mcq"], 1):
        choice = st.radio(q["stem"], q["options"], key=f"mcq{i}")
        if st.button(f"Submit MCQ {i}"):
            res = grade_mcq(choice, q["answer"])
            st.write("✅ Correct" if res["score"] == 1 else f"❌ {res['feedback']}")
            log_attempt(role, EMAIL, subject, topic, "MCQ", f"Q{i}:{q['stem']}|{choice}",
                        score=res["score"], feedback=res["feedback"])

    st.markdown("**SAQ (4 marks)**")
    saq_text = st.text_area("Your answer", key="saq")
    if st.button("Submit SAQ"):
        res = grade_saq(saq_text, ["definition", "formula", "interpretation", "real-world"])
        st.info(f"Score {res['score']}/4 — {res['feedback']}")
        log_attempt(role, EMAIL, subject, topic, "SAQ", saq_text,
                    score=res["score"], feedback=res["feedback"])

    st.markdown("**LAQ (15 marks)**")
    laq_text = st.text_area("Your essay", key="laq")
    if st.button("Submit LAQ"):
        res = grade_laq(laq_text, ["explain", "analyse", "evaluate", "diagram", "real world", "conclusion"])
        st.info(f"Score {res['score']}/15 — {res['feedback']}")
        log_attempt(role, EMAIL, subject, topic, "LAQ", laq_text,
                    score=res["score"], feedback=res["feedback"])

with tab5:
    st.subheader("Revision")
    for n in pkg["revision"]["notes"]:
        st.write("• " + n)
    st.write("**Exam prep:**")
    for step in pkg["revision"]["exam_prep"]:
        st.write("- " + step)
