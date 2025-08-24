#!/bin/bash
set -e
echo "🔧 Installing Option C — Hybrid Diagnostics (rules + Ollama)"

# Ensure folders
mkdir -p utils data

# Create/ensure accuracy log
if [ ! -f data/accuracy_log.csv ]; then
  echo "timestamp,email,subject,level,board,topic,qtype,confidence,score,source,feedback,steps_json" > data/accuracy_log.csv
fi

# ----- utils/diagnostics.py -----
cat > utils/diagnostics.py <<'PY'
import os, json, math, time, subprocess, shlex
from typing import Dict, Any, Optional, List

# ENV toggles
USE_OLLAMA = os.getenv("OLLAMA", "1") == "1"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

def _clean_text(s: Optional[str]) -> str:
    return (s or "").strip()

def _base_rules(qtype: str, question: str, student: str, expected: Optional[str]) -> Dict[str, Any]:
    """Lightweight rule checks for SAQ/LAQ + numericals, returns dict with score/feedback/steps."""
    student_l = student.lower()
    feedback: List[str] = []
    steps: List[str] = []

    # Common emptiness check
    if not student or len(student.split()) < 3:
      return {"score": 0, "feedback": "Answer too short or empty.", "steps": ["Add a complete response."], "confidence": 0.4, "source": "rules"}

    # SAQ rules
    if qtype.upper() == "SAQ":
        expect_terms = ["define","explain","because","therefore","diagram","axes"]
        hits = sum(1 for t in expect_terms if t in student_l)
        if hits < 2:
            feedback.append("Add explicit definition/explanation and at least one reason ('because/therefore').")
        if "diagram" not in student_l:
            feedback.append("Mention the relevant diagram and axes if applicable.")
        length = len(student.split())
        if length < 40:
            feedback.append("Write at least ~60–100 words for a complete SAQ.")
        score = max(1, min(4, int(round((hits/4.0)*4))))  # rough banding
        steps = [
            "State a clear definition using correct terms.",
            "Explain causality using 'because/therefore'.",
            "Reference the diagram/axes if relevant.",
            "End with a one-line insight or implication."
        ]
        return {"score": score, "feedback": " ".join(feedback) if feedback else "Clear, concise use of terms.", "steps": steps, "confidence": 0.65, "source": "rules"}

    # LAQ rules
    if qtype.upper() == "LAQ":
        bands = {
            "knowledge": any(w in student_l for w in ["demand","supply","price","elasticity","equilibrium","diagram","curve"]),
            "analysis": any(w in student_l for w in ["because","thus","therefore","hence"]),
            "evaluation": any(w in student_l for w in ["however","on the other hand","depends","limitations"]),
            "structure": any(w in student_l for w in ["introduction","conclusion"]),
            "terms": any(w in student_l for w in ["define","diagram","axes","shift","interpret"])
        }
        raw = sum(1 if v else 0 for v in bands.values()) / 5.0
        score = int(round(raw * 15))
        fb_bits = []
        if not bands["evaluation"]: fb_bits.append("Add evaluation using 'however/depends' with evidence.")
        if not bands["structure"]: fb_bits.append("Add Introduction and Conclusion for clear structure.")
        if "diagram" not in student_l: fb_bits.append("Refer to a diagram and interpret shifts/movements.")
        steps = [
            "Open with Introduction (define key terms, outline argument).",
            "Develop Analysis paragraphs (because/therefore chains).",
            "Insert Evaluation (counterpoints, depends-on, real evidence).",
            "Close with Conclusion (answer the question, conditions)."
        ]
        return {"score": score, "feedback": " ".join(fb_bits) if fb_bits else "Strong analysis and evaluation.", "steps": steps, "confidence": 0.6 + 0.3*raw, "source": "rules"}

    # Numerical rules
    if qtype.upper() == "NUM":
        fb = []
        steps = ["Write the correct formula.", "Substitute values with consistent units.", "Compute carefully.", "Interpret result with units/context."]
        score = 0
        conf = 0.55
        if expected is not None:
            try:
                # tolerant numeric compare if expected looks numeric
                exp = float(str(expected).replace("%","").strip())
                # try extracting a number from student
                import re
                nums = re.findall(r"-?\d+\.?\d*", student)
                stu_val = float(nums[0]) if nums else None
                if stu_val is None:
                    fb.append("No numeric value detected; show your calculation and final number.")
                else:
                    # tolerance 1–5%
                    tol = max(0.01*abs(exp), 0.05)
                    if abs(stu_val - exp) <= tol:
                        score = 4
                        conf = 0.8
                        fb.append("Correct value within tolerance.")
                    else:
                        fb.append(f"Calculation mismatch: expected ≈ {exp}, you wrote {stu_val}. Recheck division/multiplication and unit conversions.")
                        score = 2
                        conf = 0.6
            except Exception:
                fb.append("Could not verify number; ensure you provide a final numeric result.")
        else:
            fb.append("No expected result provided; rule-check limited.")
        return {"score": score, "feedback": " ".join(fb) if fb else "Looks correct.", "steps": steps, "confidence": conf, "source": "rules"}

    # default
    return {"score": 0, "feedback": "Unsupported question type.", "steps": ["Specify SAQ/LAQ/NUM."], "confidence": 0.4, "source": "rules"}

def _ollama_available() -> bool:
    if not USE_OLLAMA: return False
    try:
        subprocess.run(["ollama","--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def _ollama_json(prompt: str) -> Optional[Dict[str, Any]]:
    """Call ollama run <model> with a JSON-instruction prompt and try to parse a JSON object from stdout."""
    cmd = f"ollama run {shlex.quote(OLLAMA_MODEL)}"
    try:
        proc = subprocess.run(cmd, input=prompt.encode("utf-8"), shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        out = proc.stdout.decode("utf-8", errors="ignore").strip()
        # Try to locate a JSON object in output
        import re
        m = re.search(r"\{[\s\S]*\}", out)
        if m:
            return json.loads(m.group(0))
        # fallback: plain text
        return {"feedback": out}
    except Exception as e:
        return None

SCHEMA = {
  "type":"object",
  "properties":{
    "score":{"type":"number"},
    "feedback":{"type":"string"},
    "steps":{"type":"array","items":{"type":"string"}},
    "confidence":{"type":"number"}
  },
  "required":["feedback"]
}

def _apply_schema(d: Dict[str, Any]) -> Dict[str, Any]:
    # fill defaults
    d = dict(d or {})
    d.setdefault("score", 0)
    d.setdefault("steps", [])
    d.setdefault("confidence", 0.5)
    return d

def diagnose(qtype: str, question: str, student_answer: str, expected_solution: Optional[str]=None, context: Optional[str]=None) -> Dict[str, Any]:
    """
    Hybrid diagnostics:
      1) Rule-based baseline (always)
      2) If Ollama available: ask for step-specific diagnostics and a JSON response
      3) Merge with conservative policy (never overrule clear rule failures unless model confidence >= 0.7)
    """
    base = _base_rules(qtype, question, student_answer, expected_solution)

    if not _ollama_available():
        base["source"] = "rules"
        return _apply_schema(base)

    # Build prompt for JSON feedback
    schema_hint = """Respond ONLY as JSON with keys: score (0-100), feedback (string), steps (array of strings), confidence (0.0-1.0)."""
    task = f"""You are an IB-style examiner. Provide step-by-step diagnostics aligned to the rubric.

QUESTION TYPE: {qtype}
QUESTION: {question}
EXPECTED (if any): {expected_solution or "n/a"}
STUDENT ANSWER:
{student_answer}

{schema_hint}
Avoid generic praise. Pinpoint what is missing (definitions, diagrams, calculations, evaluation)."""
    model = _ollama_json(task)
    if not model or not isinstance(model, dict):
        return _apply_schema(base)

    model = _apply_schema(model)
    # Merge: conservative — if model confidence low, prefer base; else combine
    conf = float(model.get("confidence", 0.5))
    merged: Dict[str, Any] = dict(base)
    if conf >= 0.7:
        # Use model score scaled into our max bands where appropriate
        merged["feedback"] = (base.get("feedback","") + " ").strip() + model.get("feedback","")
        merged["steps"] = list(dict.fromkeys(base.get("steps",[]) + model.get("steps",[])))
        merged["confidence"] = max(conf, merged.get("confidence",0.6))
        merged["source"] = "rules+ollama"
        # If SAQ/LAQ, cap scores by max marks from UI; here keep raw 0-100 to be scaled by caller
        merged["score_raw"] = model.get("score", 0)
    else:
        merged["source"] = "rules"
    return _apply_schema(merged)
PY

# ----- pages/01_Student_Mode.py (full replacement) -----
cat > pages/01_Student_Mode.py <<'PY'
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
PY

echo "✅ Option C diagnostics installed. Restart Streamlit to take effect."
