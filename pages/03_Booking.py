import streamlit as st
from utils.storage import log_booking
import sys, os
def logout_fix():
    import streamlit as st
    for k in [k for k in list(st.session_state.keys()) if k.startswith("auth_") or k in ("role","email","auth_board")]:
        st.session_state.pop(k, None)
    st.rerun()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


st.set_page_config(page_title="SkillNestEdu — Booking", layout="wide")

role = st.session_state.get("role")
if role not in ("admin", "student_ib", "student_cbse", "student_icse", "student_ug",
                "student_ielts", "student_pte", "student_softskills", "student_spokenenglish"):
    st.warning("Please login first (Pages → Login)."); st.stop()

st.title("Book a 45-minute session (₹500)")
name = st.text_input("Your name")
email = st.text_input("Your email")
slot = st.text_input("Preferred slot (ISO 8601, e.g., 2025-08-25T16:00:00+05:30)")
notes = st.text_area("Notes (topic, goals)")
if st.button("Request Booking", use_container_width=True):
    if not (name and email and slot):
        st.error("Please fill name, email, and slot.")
    else:
        log_booking(name, email, slot, notes)
        st.success("Booking request recorded. You’ll receive a confirmation email shortly.")

st.divider()
if role=="admin":
    import pandas as pd, pathlib
    st.subheader("Admin — Booking Requests")
    p = pathlib.Path("data/bookings.csv")
    if p.exists():
        df = pd.read_csv(p)
        st.dataframe(df, use_container_width=True)
    else:
        st.caption("No bookings yet.")
