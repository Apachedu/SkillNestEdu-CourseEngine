import os, json, csv, textwrap, pathlib, datetime

ROOT = pathlib.Path(".").resolve()

FILES = {
# ---------------- utils/storage.py ----------------
"utils/storage.py": r'''import os, csv, datetime, pathlib
DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
ATTEMPTS_CSV = DATA_DIR / "attempts.csv"
BOOKINGS_CSV = DATA_DIR / "bookings.csv"

def _append_row(path, header, rowdict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists: w.writeheader()
        w.writerow(rowdict)

def log_attempt(role, email, subject, topic, qtype, payload, score=None, feedback=None):
    row = {
        "ts": datetime.datetime.utcnow().isoformat()+"Z",
        "role": role, "email": email, "subject": subject, "topic": topic,
        "qtype": qtype, "payload": payload, "score": "" if score is None else score,
        "feedback": "" if feedback is None else feedback
    }
    _append_row(ATTEMPTS_CSV, list(row.keys()), row)

def log_booking(name, email, slot_iso, notes, price="500", minutes="45"):
    row = {
        "ts": datetime.datetime.utcnow().isoformat()+"Z",
        "name": name, "email": email, "slot_iso": slot_iso, "minutes": minutes, "price": price, "notes": notes
    }
    _append_row(BOOKINGS_CSV, list(row.keys()), row)
''',

# ---------------- utils/feedback.py ----------------
"utils/feedback.py": r'''import re, json

def _norm(s): return re.sub(r"\s+", " ", (s or "")).strip().lower()

def grade_mcq(user_choice, correct):
    ok = (user_choice == correct)
    return {"score": 1 if ok else 0, "feedback": ("Correct." if ok else f"Answer is {correct}"), "confidence": 0.95}

def grade_saq(text, keywords:list):
    t = _norm(text); found = [k for k in keywords if k.lower() in t]
    score = round(len(found)/max(1,len(keywords))*4, 1)  # out of 4
    fb = f"Covered: {', '.join(found)}. Missing: {', '.join([k for k in keywords if k not in found])}."
    return {"score": score, "feedback": fb, "confidence": 0.7 if score<2 else 0.85}

def grade_laq(text, rubric_points:list):
    t = _norm(text); pts = []
    for rp in rubric_points:
        ok = all(word.lower() in t for word in rp.split())
        pts.append(1 if ok else 0)
    score15 = sum(pts)* (15/len(rubric_points))
    fb = f"Met {sum(pts)}/{len(pts)} rubric items."
    return {"score": round(score15,1), "feedback": fb, "confidence": 0.75}

def diagnose_numeric(steps:list, expected_answer:str):
    hints=[]
    for s in steps:
        if "%" in s and ("elasticity" in s.lower() or "ped" in s.lower()):
            if "sign" not in s.lower(): hints.append("Include the sign (elasticity often negative).")
        if "units" in s.lower() and "price" in s.lower():
            hints.append("Check units — keep currency/quantity consistent.")
    if not hints: hints=["Re-check formula, substitution, and rounding."]
    ok = (expected_answer.strip() in "".join(steps))
    return {"score": 1 if ok else 0, "feedback": " | ".join(hints), "confidence": 0.6}

# Optional Ollama integration (if installed) — safe fallback to rules when not available
def llm_feedback(prompt:str, model:str="llama3"):
    try:
        import subprocess, json
        proc = subprocess.run(["ollama","run",model], input=prompt.encode(), stdout=subprocess.PIPE, timeout=20)
        out = proc.stdout.decode()
        return {"score": None, "feedback": out.strip()[:1000], "confidence": 0.6}
    except Exception:
        return {"score": None, "feedback": "(Local model not available; used rule-based feedback.)", "confidence": 0.5}
''',

# ---------------- pages/01_Student_Mode.py ----------------
"pages/01_Student_Mode.py": r'''import streamlit as st
from ui_branding import sidebar_branding, page_watermark
from engine_v2 import generate_package
from utils.storage import log_attempt
from utils.feedback import grade_mcq, grade_saq, grade_laq

st.set_page_config(page_title="SkillNestEdu — Student Mode", layout="wide")

role = st.session_state.get("auth_role")
if role not in ("admin", "student_ib", "student_cbse", "student_icse", "student_ug",
                "student_ielts", "student_pte", "student_softskills", "student_spokenenglish"):
    st.warning("Please login first (Pages → Login)."); st.stop()

def logout():
    for k in [k for k in st.session_state.keys() if k.startswith("auth_")]: st.session_state.pop(k, None)
    try: st.rerun()
    except Exception: st.experimental_rerun()

with st.container(border=True):
    st.write(f"🔐 Logged in as **{st.session_state.get('auth_email','?')}** • Role: **{st.session_state.get('auth_role','?').replace('_',' ').title()}** • Board: **{st.session_state.get('auth_board','—')}**")
    st.button("Logout", on_click=logout)

EMAIL = st.session_state.get("auth_email", "student@skillnestedu.com")
BOARD_LABEL = st.session_state.get("auth_board", "IB")
sidebar_branding(EMAIL, BOARD_LABEL, None); page_watermark(EMAIL, None)

subject = st.sidebar.selectbox("Subject", ["IB Economics","IB Math AA","IELTS","PTE","Soft Skills","Spoken English"])
level   = st.sidebar.selectbox("Level", ["SL","HL","General"])
board   = st.sidebar.selectbox("Board", ["IB","CBSE","ICSE","UG","IELTS","PTE","Soft Skills","Spoken English"])
topic   = st.sidebar.text_input("Topic", value=st.session_state.get("prefill_topic","Price Elasticity of Demand"))
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
    st.caption("Interactive diagrams coming soon (deterministic).")

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
            res = grade_mcq(choice, q["answer"])
            st.write("✅ Correct" if res["score"]==1 else f"❌ {res['feedback']}")
            log_attempt(role, EMAIL, subject, topic, "MCQ", f"Q{i}:{q['stem']}|{choice}", score=res["score"], feedback=res["feedback"])

    st.markdown("**SAQ (4 marks)**")
    saq_text = st.text_area("Your answer", key="saq")
    if st.button("Submit SAQ"):
        res = grade_saq(saq_text, ["definition","formula","interpretation","real-world"])
        st.info(f"Score {res['score']}/4 — {res['feedback']}")
        log_attempt(role, EMAIL, subject, topic, "SAQ", saq_text, score=res["score"], feedback=res["feedback"])

    st.markdown("**LAQ (15 marks)**")
    laq_text = st.text_area("Your essay", key="laq")
    if st.button("Submit LAQ"):
        res = grade_laq(laq_text, ["explain", "analyse", "evaluate", "diagram", "real world", "conclusion"])
        st.info(f"Score {res['score']}/15 — {res['feedback']}")
        log_attempt(role, EMAIL, subject, topic, "LAQ", laq_text, score=res["score"], feedback=res["feedback"])

with tab5:
    st.subheader("Revision")
    for n in pkg["revision"]["notes"]: st.write("• " + n)
    st.write("**Exam prep:**")
    for step in pkg["revision"]["exam_prep"]: st.write("- " + step)
''',

# ---------------- pages/02_Creator_Mode.py ----------------
"pages/02_Creator_Mode.py": r'''import streamlit as st
from ui_branding import sidebar_branding, page_watermark
from engine_v2 import generate_package

st.set_page_config(page_title="SkillNestEdu — Creator Mode", layout="wide")

role = st.session_state.get("auth_role")
if role != "admin":
    st.error("Admins only. Please login as Admin (Pages → Login)."); st.stop()

def logout():
    for k in [k for k in st.session_state.keys() if k.startswith("auth_")]: st.session_state.pop(k, None)
    try: st.rerun()
    except Exception: st.experimental_rerun()

with st.container(border=True):
    st.write(f"🔐 Logged in as **{st.session_state.get('auth_email','?')}** • Role: **{st.session_state.get('auth_role','?').replace('_',' ').title()}**")
    st.button("Logout", on_click=logout)

EMAIL = st.session_state.get("auth_email", "admin@skillnestedu.com")
sidebar_branding(EMAIL, "All Boards", None); page_watermark(EMAIL, None)

st.title("Creator Mode")
subject = st.selectbox("Subject", ["IB Economics","IB Math AA","IELTS","PTE","Soft Skills","Spoken English"])
level = st.selectbox("Level", ["SL","HL","General"])
topic = st.text_input("Topic")
if st.button("Generate"):
    pkg = generate_package(subject, level, "Any", topic)
    st.success("Package generated")
    st.subheader("Carousel Points")
    for s in pkg["study_guide"]["key_points"]: st.write("• " + s)
    st.subheader("Reel Script"); st.write(f"Hook → 3 steps → CTA about {topic}")
    st.subheader("Blog Outline"); st.write("Intro → Concept → Examples → Practice → Summary")
''',

# ---------------- pages/03_Booking.py ----------------
"pages/03_Booking.py": r'''import streamlit as st
from utils.storage import log_booking

st.set_page_config(page_title="SkillNestEdu — Booking", layout="wide")

role = st.session_state.get("auth_role")
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
''',

# ---------------- pages/04_Case_Studies.py ----------------
"pages/04_Case_Studies.py": r'''import streamlit as st, json, pathlib

st.set_page_config(page_title="SkillNestEdu — Case Studies", layout="wide")

role = st.session_state.get("auth_role")
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
''',
}

# write files
for rel, content in FILES.items():
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print("Wrote", rel)

print("\nPhase 2 drop-ins installed. Next:\n- Run: git add .\n- Run: git commit -m \"phase2: content+feedback+booking+cases\"\n- Run: git push\n- Reboot Streamlit Cloud\n")

