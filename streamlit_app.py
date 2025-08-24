import streamlit as st
st.set_page_config(page_title="SkillNestEdu", layout="wide")
st.title("SkillNestEdu — Content Engine")

st.markdown("""
Use **Pages** (left sidebar):

- **Login** (first-time gate)
- **Student Mode** (study package + strict SAQ/LAQ grading + attempts log)
- **Creator Mode** (content generation helper)
- **Booking** (₹500 / 45 min slots; admin can add slots)

Case studies: Pages ➜ Student Mode ➜ **Case Study** tab (reads `interactive/case_studies/registry.json`).
""")
