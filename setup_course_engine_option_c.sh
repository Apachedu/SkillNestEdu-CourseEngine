#!/bin/bash
set -e

echo "🚀 Setting up SkillNestEdu — Option C (Secure + Feedback + Sims + Booking)"

# =========================
# 0) CONFIG — EDIT IF NEEDED
# =========================
LICENSED_EMAIL="contact@skillnestedu.com"
LICENSED_PASSWORD="skillnest2025"     # change after first run
ALLOWED_BOARDS='["IB"]'
EXPIRY_DATE="2026-05-31"
BOOKING_PRICE="500"
BOOKING_DURATION_MIN="45"

# =============
# 1) ENV & TREE
# =============
if [ ! -d ".venv" ]; then
  python3 -m venv .venv
  echo "✅ .venv created"
fi
source .venv/bin/activate
echo "✅ venv activated"

mkdir -p .streamlit assets boards/ib interactive/case_studies pages prompts/ib_econ prompts/ib_math_aa utils data .license

# ============
# 2) DOTFILES
# ============
cat > .gitignore <<'EOF'
__pycache__/
*.pyc
*.sqlite
.env
.streamlit/secrets.toml
.data/
.license/*.json
EOF

cat > .streamlit/config.toml <<'EOF'
[theme]
primaryColor = "#9CE00B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F2F2F2"
textColor = "#333333"
font = "sans serif"
EOF

# ==================
# 3) PY DEPENDENCIES
# ==================
cat > requirements.txt <<'EOF'
streamlit
plotly
sympy
pint
numpy
pandas
pillow
EOF

pip install --upgrade pip >/dev/null
pip install -r requirements.txt >/dev/null
echo "✅ Dependencies installed"

# ============
# 4) LICENSING
# ============
cat > .license/license.json <<EOF
{
  "email": "${LICENSED_EMAIL}",
  "password": "${LICENSED_PASSWORD}",
  "boards": ${ALLOWED_BOARDS},
  "expiry": "${EXPIRY_DATE}"
}
EOF
echo "✅ License created for ${LICENSED_EMAIL}"

# ====================
# 5) CORE UI & ENGINE
# ====================
cat > ui_branding.py <<'EOF'
import time, streamlit as st

def _fmt(datestr):
    return datestr or "—"

def sidebar_branding(email="contact@skillnestedu.com", board="IB", expiry_str=None):
    st.sidebar.image("assets/skillnestlogo.png", use_container_width=True)
    st.sidebar.markdown(f"""
<div style="padding:10px;border-radius:12px;background:#F2F2F2">
  <div style="font-weight:700;color:#333;font-size:18px">SkillNestEdu</div>
  <div style="color:#333;font-size:13px;margin-top:6px;line-height:1.3">
    Licensed to <b>{email}</b><br/>Board: <b>{board}</b><br/>Valid till <b>{_fmt(expiry_str)}</b>
  </div>
</div>""", unsafe_allow_html=True)

def page_watermark(email="contact@skillnestedu.com", expiry_str=None):
    ex = _fmt(expiry_str)
    st.markdown(f"""
<style>
.block-container::before {{
  content: "© SkillNestEdu • {email} • Valid till {ex}";
  position: fixed; top:28%; left:-12%; transform: rotate(-30deg);
  color: rgba(0,0,0,0.07); font-size:4.2rem; font-weight:700;
  z-index:0; pointer-events:none;
}}
</style>""", unsafe_allow_html=True)
EOF

cat > engine_v2.py <<'EOF'
import datetime

def generate_package(subject:str, level:str, board:str, topic:str, difficulty:str="Beginner"):
    """Stub generator—wire to Gamma API later."""
    return {
        "meta": {
            "subject": subject, "level": level, "board": board, "topic": topic,
            "generated_at": datetime.datetime.utcnow().isoformat()+"Z"
        },
        "study_guide": {
            "intro": f"{topic}: why it matters (2–3 lines).",
            "concept": [f"Definition of {topic}.", "Key idea #1", "Key idea #2"],
            "diagram_spec": {"hint": "axes/labels; add interactive later"},
            "examples": ["Real-world example 1", "Real-world example 2"],
            "key_points": ["Point A","Point B","Point C"]
        },
        "worked_example": {
            "steps": ["Step 1","Step 2","Step 3"], "answer": "Final answer"
        },
        "practice": {
            "mcq": [{"stem":"Sample MCQ?","options":["A","B","C","D"],"answer":"A"} for _ in range(5)],
            "saq": [{"stem":"Explain … (4 marks)","max":4,"criteria":["accuracy","clarity","use_of_terms","examples"]}],
            "laq": [{"stem":"Evaluate … (15 marks)","max":15,"criteria":["knowledge","analysis","evaluation","structure","terms"]}]
        },
        "revision": {
            "notes": ["Rev note 1","Rev note 2","Rev note 3"],
            "exam_prep": ["Plan: 10 min outline → 25 min write → 5 min check"]
        }
    }
EOF

# =========
# 6) AUTH
# =========
cat > utils/auth.py <<'EOF'
import json, os, datetime
import streamlit as st

LICENSE_PATH = ".license/license.json"

def load_license():
    if not os.path.exists(LICENSE_PATH):
        st.stop()
    with open(LICENSE_PATH, "r") as f:
        return json.load(f)

def validate(email:str, password:str, board:str):
    lic = load_license()
    if email != lic.get("email") or password != lic.get("password"):
        return False, "Invalid credentials."
    if board not in lic.get("boards", []):
        return False, f"Board '{board}' not permitted."
    expiry = lic.get("expiry")
    if expiry:
        try:
            if datetime.date.today() > datetime.date.fromisoformat(expiry):
                return False, "License expired."
        except Exception:
            return False, "License expiry invalid."
    return True, "OK"

def get_license_info():
    lic = load_license()
    return lic.get("email"), lic.get("boards", []), lic.get("expiry")
EOF

# ===============
# 7) GRADING PACK
# ===============
cat > utils/grading.py <<'EOF'
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class Rubric:
    criteria: List[str]
    weights: List[float]  # sum to 1

def simple_keyword_score(answer:str, keywords:List[str]) -> float:
    if not answer or not keywords: return 0.0
    found = sum(1 for k in keywords if k.lower() in answer.lower())
    return found / len(keywords)

def grade_saq(answer:str, max_marks:int, criteria:List[str]) -> Tuple[int,str]:
    # Strict but deterministic: expects at least 3 econ terms/evidence
    keywords = ["define","explain","because","so","therefore","demand","supply","elasticity","price","income","equilibrium"]
    base = simple_keyword_score(answer, keywords)
    # length pressure: 40–120 words ideal
    length = len(answer.split())
    length_factor = 1.0 if 40 <= length <= 120 else 0.7
    raw = base * length_factor
    score = int(round(raw * max_marks))
    # feedback
    fb = []
    if length < 40: fb.append("Too short; add definition and one example.")
    if "diagram" not in answer.lower(): fb.append("Mention the diagram/axes if relevant.")
    if "evaluate" in " ".join(criteria).lower() and "however" not in answer.lower(): fb.append("Add a counterpoint for evaluation.")
    if not fb: fb.append("Clear and concise. Good use of terms.")
    return max(0,min(score,max_marks)), " ".join(fb)

def grade_laq(answer:str, max_marks:int, criteria:List[str]) -> Tuple[int,str]:
    # Banded approach: knowledge/analysis/eval/structure/terms
    kw_econ = ["demand","supply","price","elasticity","tax","subsidy","equilibrium","welfare","efficiency","market failure","externality","opportunity cost","PED","PES","YED","XED"]
    knowledge = simple_keyword_score(answer, kw_econ)
    analysis = 1.0 if any(w in answer.lower() for w in ["because","thus","therefore","hence"]) else 0.6
    evaln = 1.0 if any(w in answer.lower() for w in ["however","on the other hand","depends","limitations"]) else 0.5
    structure = 1.0 if any(h in answer.lower() for h in ["introduction","conclusion"]) else 0.7
    terms = 1.0 if any(t in answer.lower() for t in ["define","diagram","axes","curve","shifts","interpret"]) else 0.6
    raw = (knowledge*0.3 + analysis*0.25 + evaln*0.25 + structure*0.1 + terms*0.1)
    # word target 350–600
    words = len(answer.split())
    word_factor = 1.0 if 350 <= words <= 600 else (0.85 if 250 <= words < 350 else 0.7)
    score = int(round(raw * word_factor * max_marks))
    fb = []
    if words < 350: fb.append("Increase depth (target 350–600 words).")
    if evaln < 1.0: fb.append("Stronger evaluation with 'however/depends' and real-world evidence.")
    if structure < 1.0: fb.append("Add Introduction and Conclusion for structure.")
    if "diagram" not in answer.lower(): fb.append("Refer to a diagram and interpret movements/shifts.")
    if not fb: fb.append("Well structured, analytical, and evaluative.")
    return max(0,min(score,max_marks)), " ".join(fb)
EOF

# =======================
# 8) CASE STUDY REGISTRY
# =======================
cat > interactive/case_studies/registry.json <<'EOF'
{
  "IB Economics": {
    "Nike Outsourcing": {
      "type": "iframe",
      "url": "https://skillnestcasestudysim-d6355.web.app/sim?caseId=nike_outsourcing"
    }
  },
  "IB Business Management": {
  }
}
EOF

# =======================
# 9) IB BOARD STANDARDS
# =======================
cat > boards/ib/standards.json <<'EOF'
{
  "paper_map": { "short_answer_marks": [2,4], "long_answer_marks": [10,15,20], "numericals": false },
  "command_terms": ["define","explain","analyse","evaluate","discuss"],
  "expiry": "2026-05-31"
}
EOF

# =================
# 10) PROMPT TEMPL
# =================
mkdir -p prompts/ib_econ prompts/ib_math_aa

cat > prompts/ib_econ/self_study_package.txt <<'EOF'
[Role] IB Economics tutor. Produce a COMPLETE self-study unit for <Topic>.
[Sections] Intro; What is it?; Diagram guide; Interpretation; Key Points; Worked Example; Practice (10 MCQ, 5 SAQ, 3 LAQ); Revision Notes (10 bullets); Exam Prep (timed plan).
[Style] Student-friendly, exam-ready, UK spelling.
EOF

cat > prompts/ib_math_aa/self_study_package.txt <<'EOF'
[Role] IB Mathematics AA tutor. Deep worked steps and many numericals for <Topic>.
Include 10 practice numericals with full solutions, plus 10 MCQ and 5 SAQ.
EOF

# ==============
# 11) DATA FILES
# ==============
cat > data/attempts.csv <<'EOF'
timestamp,email,subject,level,board,topic,qtype,max_marks,score,feedback
EOF

cat > data/slots.csv <<EOF
slot_id,date,time,subject,duration_min,price,status,booked_by_email
1,$(date -v+1d +"%Y-%m-%d"),17:00,Economics,${BOOKING_DURATION_MIN},${BOOKING_PRICE},available,
2,$(date -v+2d +"%Y-%m-%d"),18:00,Business Management,${BOOKING_DURATION_MIN},${BOOKING_PRICE},available,
EOF

cat > data/bookings.csv <<'EOF'
timestamp,slot_id,email,subject,notes
EOF

# =========================
# 12) PAGES — LOGIN GATE
# =========================
cat > pages/00_Login.py <<'EOF'
import streamlit as st
from utils.auth import validate, get_license_info
from ui_branding import sidebar_branding, page_watermark

st.set_page_config(page_title="SkillNestEdu — Login", layout="wide")

lic_email, boards, expiry = get_license_info()
sidebar_branding(lic_email, boards[0] if boards else "—", expiry)
page_watermark(lic_email, expiry)

st.title("Login")
email = st.text_input("Licensed Email")
password = st.text_input("Password", type="password")
board = st.selectbox("Board", boards if boards else ["IB"])

if st.button("Login", use_container_width=True):
    ok, msg = validate(email, password, board)
    if ok:
        st.session_state["auth_email"] = email
        st.session_state["auth_board"] = board
        st.success("Logged in. Use the sidebar ➜ Pages.")
    else:
        st.error(msg)
EOF

# =========================
# 13) STUDENT MODE (AUTH)
# =========================
cat > pages/01_Student_Mode.py <<'EOF'
import streamlit as st, pandas as pd, time, json
from ui_branding import sidebar_branding, page_watermark
from engine_v2 import generate_package
from utils.auth import get_license_info
from utils.grading import grade_saq, grade_laq

st.set_page_config(page_title="SkillNestEdu — Student Mode", layout="wide")

# Auth gate
if "auth_email" not in st.session_state:
    st.warning("Please login first (Pages ➜ Login).")
    st.stop()

auth_email = st.session_state["auth_email"]
auth_board = st.session_state.get("auth_board", "IB")
lic_email, boards, expiry = get_license_info()
sidebar_branding(lic_email, auth_board, expiry); page_watermark(lic_email, expiry)

# Sidebar controls
subject = st.sidebar.selectbox("Subject", ["IB Economics","IB Math AA"])
level   = st.sidebar.selectbox("Level", ["SL","HL"])
board   = st.sidebar.selectbox("Board", ["IB"])
topic   = st.sidebar.text_input("Topic", value="Price Elasticity of Demand")
difficulty = st.sidebar.selectbox("Difficulty", ["Beginner","Advanced"])

st.title("Student Mode")
if st.button("Generate Study Package", use_container_width=True):
    st.session_state["pkg"] = generate_package(subject, level, board, topic, difficulty)

pkg = st.session_state.get("pkg")
if not pkg:
    st.info("Enter a topic and click **Generate Study Package**."); st.stop()

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
    # MCQ demo
    st.markdown("**MCQ**")
    for i,q in enumerate(pkg["practice"]["mcq"],1):
        choice = st.radio(q["stem"], q["options"], key=f"mcq{i}")
        if st.button(f"Submit MCQ {i}"):
            st.write("✅ Correct" if choice==q["answer"] else f"❌ Answer: {q['answer']}")
    # SAQ
    st.markdown("**SAQ (4 marks)**")
    saq_text = st.text_area("Write your SAQ answer:", key="saq")
    if st.button("Submit SAQ"):
        score, fb = grade_saq(saq_text, pkg["practice"]["saq"][0]["max"], pkg["practice"]["saq"][0]["criteria"])
        st.info(f"Score: {score}/{pkg['practice']['saq'][0]['max']}")
        st.write("Feedback:", fb)
        df = pd.read_csv("data/attempts.csv")
        new = {
            "timestamp": int(time.time()), "email": auth_email, "subject": subject,
            "level": level, "board": board, "topic": topic,
            "qtype": "SAQ", "max_marks": pkg["practice"]["saq"][0]["max"],
            "score": score, "feedback": fb
        }
        df.loc[len(df)] = new
        df.to_csv("data/attempts.csv", index=False)
        st.success("Saved to My Attempts.")
    # LAQ
    st.markdown("**LAQ (15 marks)**")
    laq_text = st.text_area("Write your LAQ answer:", key="laq", height=220)
    if st.button("Submit LAQ"):
        score, fb = grade_laq(laq_text, pkg["practice"]["laq"][0]["max"], pkg["practice"]["laq"][0]["criteria"])
        st.info(f"Score: {score}/{pkg['practice']['laq'][0]['max']}")
        st.write("Feedback:", fb)
        df = pd.read_csv("data/attempts.csv")
        new = {
            "timestamp": int(time.time()), "email": auth_email, "subject": subject,
            "level": level, "board": board, "topic": topic,
            "qtype": "LAQ", "max_marks": pkg["practice"]["laq"][0]["max"],
            "score": score, "feedback": fb
        }
        df.loc[len(df)] = new
        df.to_csv("data/attempts.csv", index=False)
        st.success("Saved to My Attempts.")

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
    except Exception as e:
        st.info("No attempts yet.")
EOF

# =========================
# 14) CREATOR MODE (AUTH)
# =========================
cat > pages/02_Creator_Mode.py <<'EOF'
import streamlit as st
from ui_branding import sidebar_branding, page_watermark
from engine_v2 import generate_package
from utils.auth import get_license_info

st.set_page_config(page_title="SkillNestEdu — Creator Mode", layout="wide")

if "auth_email" not in st.session_state:
    st.warning("Please login first (Pages ➜ Login)."); st.stop()

auth_email = st.session_state["auth_email"]
auth_board = st.session_state.get("auth_board", "IB")
lic_email, boards, expiry = get_license_info()
sidebar_branding(lic_email, auth_board, expiry); page_watermark(lic_email, expiry)

st.title("Creator Mode")
subject = st.selectbox("Subject", ["IB Economics","IB Math AA"])
level = st.selectbox("Level", ["SL","HL"])
topic = st.text_input("Topic", value="Price Elasticity of Demand")

if st.button("Generate"):
    pkg = generate_package(subject, level, "IB", topic)
    st.success("Package generated")
    st.subheader("Carousel Points")
    for s in pkg["study_guide"]["key_points"]: st.write("• " + s)
    st.subheader("Reel Script")
    st.write(f"Hook → 3 steps → CTA about {topic}")
    st.subheader("Blog Outline")
    st.write("Intro → Concept → Examples → Practice → Summary")
EOF

# =========================
# 15) BOOKING PAGE (AUTH)
# =========================
cat > pages/03_Booking.py <<EOF
import streamlit as st, pandas as pd, time
from ui_branding import sidebar_branding, page_watermark
from utils.auth import get_license_info

PRICE = ${BOOKING_PRICE}
DUR = ${BOOKING_DURATION_MIN}

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
        dur = st.number_input("Duration (min)", min_value=15, max_value=120, value=${BOOKING_DURATION_MIN})
        price = st.number_input("Price (₹)", min_value=0, value=${BOOKING_PRICE})
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
EOF

# ======================
# 16) ROOT APP LAUNCHER
# ======================
cat > streamlit_app.py <<'EOF'
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
EOF

# =====================
# 17) FINAL INSTRUCTIONS
# =====================
echo "✅ Files ready."
echo "👉 Place your logo at: assets/skillnestlogo.png"
echo "🔐 License: .license/license.json (update password/expiry if needed)"

echo "🚀 Launching Streamlit..."
exec streamlit run streamlit_app.py

