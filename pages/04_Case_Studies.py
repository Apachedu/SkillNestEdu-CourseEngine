import streamlit as st, json, pathlib

st.set_page_config(page_title="SkillNestEdu — Case Studies", layout="wide")

role = st.session_state.get("role")
if role not in ("admin", "student_ib", "student_cbse", "student_icse", "student_ug",
                "student_ielts", "student_pte", "student_softskills", "student_spokenenglish"):
    st.warning("Please login first (Pages → Login)."); st.stop()

st.title("Case Studies")
regpath = pathlib.Path("interactive/case_studies/registry.json")
if not regpath.exists():
    st.error("registry.json not found at interactive/case_studies/"); st.stop()

reg = json.loads(regpath.read_text())
subject = st.selectbox("Subject", list(reg.keys()))
case_titles = list(reg.get(subject, {}).keys())
if not case_titles:
    st.info("No cases for this subject yet."); st.stop()

case = st.selectbox("Case", case_titles)
entry = reg[subject][case]
st.write(f"**{subject} — {case}**")
if entry.get("type")=="iframe":
    st.components.v1.iframe(entry.get("url"), height=560)
else:
    st.code(json.dumps(entry, indent=2))
