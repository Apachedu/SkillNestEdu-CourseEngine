import streamlit as st, pandas as pd, time
from ui_branding import sidebar_branding, page_watermark
from utils.auth import get_license_info

PRICE = 500
DUR = 45

st.set_page_config(page_title="SkillNestEdu — Booking", layout="wide")

if "auth_email" not in st.session_state:
    st.warning("Please login first (Pages ➜ Login)."); st.stop()

auth_email = st.session_state["auth_email"]
auth_board = st.session_state.get("auth_board", "IB")
lic_email, boards, expiry = get_license_info()
sidebar_branding(lic_email, auth_board, expiry); page_watermark(lic_email, expiry)

st.title("Book a Doubt Session")
st.caption(f"₹{PRICE} / {DUR} minutes • Economics / Business Management / Finance")

# Load slots
slots = pd.read_csv("data/slots.csv")
available = slots[slots["status"]=="available"].copy()

col1, col2 = st.columns([1,2])
with col1:
    st.subheader("Available Slots")
    st.dataframe(available, use_container_width=True, height=280)
with col2:
    st.subheader("Book Now")
    slot_ids = list(available["slot_id"]) if not available.empty else []
    if slot_ids:
        sel = st.selectbox("Choose Slot ID", slot_ids)
        notes = st.text_area("Any notes?")
        if st.button("Confirm Booking", use_container_width=True):
            slots.loc[slots["slot_id"]==sel, ["status","booked_by_email"]] = ["booked", auth_email]
            slots.to_csv("data/slots.csv", index=False)
            bk = pd.read_csv("data/bookings.csv")
            new = {"timestamp": int(time.time()), "slot_id": sel, "email": auth_email,
                   "subject": str(slots.loc[slots["slot_id"]==sel, "subject"].values[0]), "notes": notes}
            bk.loc[len(bk)] = new
            bk.to_csv("data/bookings.csv", index=False)
            st.success("Slot booked. You'll receive confirmation by email (manual for now).")
    else:
        st.info("No available slots. Please check later.")

st.divider()
st.subheader("Admin — Create Slots (licensed email only)")
if auth_email == lic_email:
    with st.form("newslot"):
        date = st.date_input("Date")
        time_ = st.time_input("Time")
        subject = st.selectbox("Subject", ["Economics","Business Management","Finance"])
        dur = st.number_input("Duration (min)", min_value=15, max_value=120, value=45)
        price = st.number_input("Price (₹)", min_value=0, value=500)
        submitted = st.form_submit_button("Add Slot")
        if submitted:
            df = pd.read_csv("data/slots.csv")
            next_id = (df["slot_id"].max() if not df.empty else 0) + 1
            df.loc[len(df)] = [next_id, str(date), time_.strftime("%H:%M"), subject, dur, price, "available", ""]
            df.to_csv("data/slots.csv", index=False)
            st.success(f"Slot {next_id} added.")
    st.download_button("Download Bookings (CSV)", open("data/bookings.csv","rb"), "bookings.csv", "text/csv")
else:
    st.info("Admin actions only available to the licensed email.")
