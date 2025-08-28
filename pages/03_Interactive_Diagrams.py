import streamlit as st
from utils.diagrams import ppc_diagram, elasticity_diagram
import sys, os
def logout_fix():
    import streamlit as st
    for k in [k for k in list(st.session_state.keys()) if k.startswith("auth_") or k in ("role","email","auth_board")]:
        st.session_state.pop(k, None)
    st.rerun()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


st.set_page_config(page_title="Interactive Diagrams", layout="wide")
st.title("Interactive Diagrams")

role = st.session_state.get("role")
if role not in ("admin","student_ib","student_cbse","student_icse","student_ug",
                "student_ielts","student_pte","student_softskills","student_spokenenglish"):
    st.warning("Please login first (Pages → Login)."); st.stop()

choice = st.selectbox(
    "Choose a diagram",
    ["PPC — Production Possibility Curve", "Elasticity — Elastic vs Inelastic Demand"],
    index=0
)

if choice.startswith("PPC"):
    ppc_diagram()
else:
    elasticity_diagram()
