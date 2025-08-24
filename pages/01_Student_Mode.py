import streamlit as st, pandas as pd, time, json
from ui_branding import sidebar_branding, page_watermark
from engine_v2 import generate_package
from utils.auth import get_license_info
from utils.diagnostics import diagnose

st.set_page_config(page_title="SkillNestEdu — Student Mode", layout="wide")

# Auth gate
if "auth_email" not in st.session_state:
    st.warning("Please login first (Pages ➜ Login)."); st.stop()

auth_email = st.session_state["auth_email"]
auth_board = st.session_state.get("auth_board", "IB")
lic_email, boards, expiry = get_license_info()
sidebar_branding(lic_email, auth_board, expiry); page_watermark(lic_email, expiry)

# Sidebar controls
subject = st.sidebar.selectbox("Subject", ["IB Economics","IB Math AA"], index=["IB Economics","IB Math AA"].index(st.session_state.get('prefill_subject','IB Economics')))
level   = st.sidebar.selectbox("Level", ["SL","HL"], index=["SL","HL"].index(st.session_state.get('prefill_level','SL')))
board   = st.sidebar.selectbox("Board", ["IB"], index=0)
topic   = st.sidebar.text_input("Topic", value=st.session_state.get('prefill_topic','Price Elasticity of Demand'))
difficulty = st.sidebar.selectbox("Difficulty", ["Beginner","Advanced"])

st.title("Student Mode")
if st.button("Generate Study Package", use_container_width=True):
    st.session_state["pkg"] = generate_package(subject, level, board, topic, difficulty)

pkg = st.session_state.get("pkg")
if not pkg:
    st.info("Enter a topic and click **Generate Study Package**."); st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Study Guide","Interactive","Case Study","Practice","Review"])

with tab1:
    st.subheader("Intro"); st.write(pkg["study_guide"]["intro"])
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
    st.caption("Loaded from interactive/case_studies/registry.json")
    try:
        registry = json.load(open("interactive/case_studies/registry.json"))
        subjects = list(registry.keys())
        sel_sub = st.selectbox("Select Subject", subjects)
        cases = list(registry.get(sel_sub, {}).keys())
        if cases:
            sel_case = st.selectbox("Select Case", cases)
            meta = registry[sel_sub][sel_case]
            if meta.get("type") == "iframe":
                st.components.v1.iframe(meta.get("url",""), height=600)
            else:
                st.write("Unsupported case type.")
        else:
            st.info("No cases registered for this subject.")
    except Exception as e:
        st.error(f"Failed to load registry: {e}")

with tab4:
    st.subheader("Practice")

    # ---------- MCQ ----------
    st.markdown("**MCQ**")
    for i,q in enumerate(pkg["practice"]["mcq"],1):
        choice = st.radio(q["stem"], q["options"], key=f"mcq{i}")
        if st.button(f"Submit MCQ {i}"):
            st.write("✅ Correct" if choice==q["answer"] else f"❌ Answer: {q['answer']}")

    # ---------- SAQ ----------
    st.markdown("**SAQ (4 marks)**")
    saq_text = st.text_area("Write your SAQ answer:", key="saq")
    if st.button("Submit SAQ"):
        diag = diagnose("SAQ", question=pkg["practice"]["saq"][0]["stem"], student_answer=saq_text, expected_solution=None)
        # scale score_raw (0-100) to 0-4 if available
        score = min(pkg["practice"]["saq"][0]["max"], int(round((diag.get("score_raw",0)/100.0)*pkg["practice"]["saq"][0]["max"]))) if "score_raw" in diag else diag.get("score",0)
        st.info(f"Score: {score}/{pkg['practice']['saq'][0]['max']}  •  Confidence: {diag.get('confidence',0):.2f}  •  Source: {diag.get('source')}")
        if diag.get("steps"): st.markdown("**What to fix next:**"); [st.write(f"• {s}") for s in diag["steps"]]
        st.write("**Feedback:**", diag.get("feedback",""))
        # Log attempts + accuracy
        attempts = pd.read_csv("data/attempts.csv")
        attempts.loc[len(attempts)] = [int(time.time()), auth_email, subject, level, board, topic, "SAQ", pkg["practice"]["saq"][0]["max"], score, diag.get("feedback","")]
        attempts.to_csv("data/attempts.csv", index=False)
        acc = pd.read_csv("data/accuracy_log.csv")
        import json as _json
        acc.loc[len(acc)] = [int(time.time()), auth_email, subject, level, board, topic, "SAQ", diag.get("confidence",0.0), score, diag.get("source","rules"), diag.get("feedback",""), _json.dumps(diag.get("steps",[]))]
        acc.to_csv("data/accuracy_log.csv", index=False)
        st.success("Saved to My Attempts + Accuracy Log.")

    # ---------- LAQ ----------
    st.markdown("**LAQ (15 marks)**")
    laq_text = st.text_area("Write your LAQ answer:", key="laq", height=220)
    if st.button("Submit LAQ"):
        diag = diagnose("LAQ", question=pkg["practice"]["laq"][0]["stem"], student_answer=laq_text, expected_solution=None)
        score = min(pkg["practice"]["laq"][0]["max"], int(round((diag.get("score_raw",0)/100.0)*pkg["practice"]["laq"][0]["max"]))) if "score_raw" in diag else diag.get("score",0)
        st.info(f"Score: {score}/{pkg['practice']['laq'][0]['max']}  •  Confidence: {diag.get('confidence',0):.2f}  •  Source: {diag.get('source')}")
        if diag.get("steps"): st.markdown("**What to fix next:**"); [st.write(f"• {s}") for s in diag["steps"]]
        st.write("**Feedback:**", diag.get("feedback",""))
        attempts = pd.read_csv("data/attempts.csv")
        attempts.loc[len(attempts)] = [int(time.time()), auth_email, subject, level, board, topic, "LAQ", pkg["practice"]["laq"][0]["max"], score, diag.get("feedback","")]
        attempts.to_csv("data/attempts.csv", index=False)
        acc = pd.read_csv("data/accuracy_log.csv")
        import json as _json
        acc.loc[len(acc)] = [int(time.time()), auth_email, subject, level, board, topic, "LAQ", diag.get("confidence",0.0), score, diag.get("source","rules"), diag.get("feedback",""), _json.dumps(diag.get("steps",[]))]
        acc.to_csv("data/accuracy_log.csv", index=False)
        st.success("Saved to My Attempts + Accuracy Log.")

    # ---------- Numericals (optional) ----------
    st.markdown("**Numerical Check (optional)**")
    coln1, coln2 = st.columns(2)
    with coln1:
        expected_num = st.text_input("Expected numeric (teacher/answer key)", value="")
    with coln2:
        student_num = st.text_area("Your working + final number", height=120, key="numanswer")
    if st.button("Submit Numerical"):
        diag = diagnose("NUM", question=f"Numerical check for {topic}", student_answer=student_num, expected_solution=expected_num or None)
        # For numericals, rule score is out of 4
        score = min(4, int(diag.get("score",0)))
        st.info(f"Score: {score}/4  •  Confidence: {diag.get('confidence',0):.2f}  •  Source: {diag.get('source')}")
        if diag.get("steps"): st.markdown("**What to fix next:**"); [st.write(f"• {s}") for s in diag["steps"]]
        st.write("**Feedback:**", diag.get("feedback",""))
        # Log
        attempts = pd.read_csv("data/attempts.csv")
        attempts.loc[len(attempts)] = [int(time.time()), auth_email, subject, level, board, topic, "NUM", 4, score, diag.get("feedback","")]
        attempts.to_csv("data/attempts.csv", index=False)
        acc = pd.read_csv("data/accuracy_log.csv")
        import json as _json
        acc.loc[len(acc)] = [int(time.time()), auth_email, subject, level, board, topic, "NUM", diag.get("confidence",0.0), score, diag.get("source","rules"), diag.get("feedback",""), _json.dumps(diag.get("steps",[]))]
        acc.to_csv("data/accuracy_log.csv", index=False)
        st.success("Saved to My Attempts + Accuracy Log.")

with tab5:
    st.subheader("Revision")
    for n in pkg["revision"]["notes"]: st.write("• " + n)
    st.write("**Exam prep:**")
    for step in pkg["revision"]["exam_prep"]: st.write("- " + step)
    st.divider()
    st.subheader("My Attempts")
    try:
        df = pd.read_csv("data/attempts.csv")
        st.dataframe(df[df["email"]==auth_email].sort_values("timestamp", ascending=False), use_container_width=True)
        st.download_button("Download My Attempts (CSV)", df[df["email"]==auth_email].to_csv(index=False), "my_attempts.csv", "text/csv")
    except Exception:
        st.info("No attempts yet.")
