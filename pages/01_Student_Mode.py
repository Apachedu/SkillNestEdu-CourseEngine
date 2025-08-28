import sys, os
def logout_fix():
    import streamlit as st
    for k in [k for k in list(st.session_state.keys()) if k.startswith("auth_") or k in ("role","email","auth_board")]:
        st.session_state.pop(k, None)
    st.rerun()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
from ui_branding import sidebar_branding, page_watermark
from engine_v2 import generate_package
from utils.retrieval import search_snippets
from utils.storage import log_attempt
from utils.feedback import grade_mcq, grade_saq, grade_laq
from utils.catalog import load_finance_index
from utils.diagrams import ppc_diagram, elasticity_diagram

st.set_page_config(page_title="SkillNestEdu — Student Mode", layout="wide")

role = st.session_state.get("role")
if role not in ("admin", "student_ib", "student_cbse", "student_icse", "student_ug", "student_ielts", "student_pte", "student_softskills", "student_spokenenglish"):
    st.warning("Please login first (Pages → Login).")
    st.stop()

def logout():
    # Clear all authentication-related keys
    for k in [k for k in list(st.session_state.keys()) if k.startswith("auth_") or k in ("role", "email", "auth_board")]:
        st.session_state.pop(k, None)
    st.rerun()

with st.container(border=True):
    st.write(
        f"🔐 Logged in as **{st.session_state.get('email','?')}** • "
        f"Role: **{st.session_state.get('role','?').replace('_',' ').title()}** • "
        f"Board: **{st.session_state.get('auth_board','—')}**"
    )
    st.button("Logout", key="logout_01_student_mode", on_click=logout_fix)
EMAIL = st.session_state.get("email", "student@skillnestedu.com")
BOARD_LABEL = st.session_state.get("auth_board", "IB")
sidebar_branding(EMAIL, BOARD_LABEL, None)
page_watermark(EMAIL, None)

board = st.sidebar.selectbox("Board", ["IB", "CBSE", "ICSE", "UG", "IELTS", "PTE", "Soft Skills", "Spoken English"])

if board == "UG":
    subject = st.sidebar.selectbox("Subject", ["Finance", "(Other UG soon)"], index=0)
    level = st.sidebar.selectbox("Level", ["Sem-wise", "General"], index=0)
    fin = load_finance_index()
    sem_keys = sorted(fin.get("semesters", {}).keys(), key=lambda s: (s == "0", int(s) if s.isdigit() else 9999))
    sem_label = st.sidebar.selectbox("Semester", ["All"] + sem_keys)
    if sem_label == "All":
        course_options = ["(no courses)"]
    else:
        course_options = fin["semesters"].get(sem_label, [])
        if not course_options:
            course_options = ["(no courses)"]
    course = st.sidebar.selectbox("Course", course_options)
    topic = st.sidebar.text_input("Topic", value="")
    difficulty = st.sidebar.selectbox("Difficulty", ["Beginner", "Advanced"])
else:
    subject = st.sidebar.selectbox("Subject", ["IB Economics", "IB Math AA", "IELTS", "PTE", "Soft Skills", "Spoken English"])
    level = st.sidebar.selectbox("Level", ["SL", "HL", "General"])
    topic = st.sidebar.text_input("Topic", value=st.session_state.get("prefill_topic", "Price Elasticity of Demand"))
    difficulty = st.sidebar.selectbox("Difficulty", ["Beginner", "Advanced"])

st.title("Student Mode")

if st.button("Generate Study Package", use_container_width=True):
    st.session_state["pkg"] = generate_package(subject, level, board, topic, difficulty)

pkg = st.session_state.get("pkg")
if not pkg:
    st.info("Enter your selections and click Generate Study Package to see your unit.")
    st.stop()

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
        st.caption("No matching sources yet. Use Admin Tools → Ingest PDFs and Build Index, then try again.")
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
    shown = False
    if board == "IB" and subject.startswith("IB Economics"):
        if "possibility" in topic.lower() or "ppc" in topic.lower():
            ppc_diagram()
            shown = True
        elif "elasticity" in topic.lower():
            elasticity_diagram()
            shown = True
    if not shown:
        st.caption("Interactive diagram not available for this topic yet.")

with tab3:
    st.subheader("Case Study (Sim)")
    st.caption("Registry wired; paste your sim URLs into interactive/case_studies/registry.json.")
    st.code("https://skillnestcasestudysim-d6355.web.app/sim?caseId=nike_outsourcing")

with tab4:
    st.subheader("Practice")
    st.markdown("MCQ")
    for i, q in enumerate(pkg["practice"]["mcq"], 1):
        choice = st.radio(q["stem"], q["options"], key=f"mcq{i}")
        if st.button(f"Submit MCQ {i}", key=f"mcq_submit_{i}"):
            res = grade_mcq(choice, q["answer"])
            st.write("✅ Correct" if res["score"] == 1 else f"❌ {res['feedback']}")
            log_attempt(role, EMAIL, subject, topic, "MCQ", f"Q{i}:{q['stem']}|{choice}", score=res["score"], feedback=res["feedback"])
    st.markdown("SAQ (4 marks)")
    saq_text = st.text_area("Your answer", key="saq")
    if st.button("Submit SAQ", key="saq_submit"):
        res = grade_saq(saq_text, ["definition", "formula", "interpretation", "real-world"])
        st.info(f"Score {res['score']}/4 — {res['feedback']}")
        log_attempt(role, EMAIL, subject, topic, "SAQ", saq_text, score=res["score"], feedback=res["feedback"])
    st.markdown("LAQ (15 marks)")
    laq_text = st.text_area("Your essay", key="laq")
    if st.button("Submit LAQ", key="laq_submit"):
        res = grade_laq(laq_text, ["explain", "analyse", "evaluate", "diagram", "real world", "conclusion"])
        st.info(f"Score {res['score']}/15 — {res['feedback']}")
        log_attempt(role, EMAIL, subject, topic, "LAQ", laq_text, score=res["score"], feedback=res["feedback"])

with tab5:
    st.subheader("Revision")
    for n in pkg["revision"]["notes"]:
        st.write("• " + n)
    st.write("Exam prep:")
    for step in pkg["revision"]["exam_prep"]:
        st.write("- " + step)
