#!/bin/bash
set -e
echo "🌱 Seeding sample content"

mkdir -p content/ib_econ content/ib_math_aa pages

# ------ IB Econ: PED ------
cat > content/ib_econ/ped.json <<'JSON'
{
  "subject": "IB Economics",
  "level": "SL",
  "topic": "Price Elasticity of Demand",
  "difficulty": "Beginner",
  "study_guide": {
    "intro": "Price Elasticity of Demand (PED) measures how responsive quantity demanded is to a change in price. It helps firms with pricing and governments with tax policy.",
    "concept": [
      "Definition: PED = %ΔQd / %ΔP (usually negative).",
      "Elastic vs inelastic: |PED| > 1 elastic; |PED| < 1 inelastic.",
      "Determinants: substitutes, necessity, time, income share.",
      "Total revenue link: moving along a demand curve."
    ],
    "examples": [
      "Luxury fashion vs. salt (elastic vs. inelastic).",
      "Fuel taxes and government revenue planning."
    ],
    "key_points": [
      "Sign convention: PED is negative; use absolute value for magnitude.",
      "Elasticity varies along a linear demand curve.",
      "Total revenue moves opposite to price when demand is elastic."
    ]
  },
  "worked_example": {
    "steps": [
      "Identify initial and new price/quantity.",
      "Compute %ΔQ and %ΔP using midpoint formula.",
      "PED = %ΔQ / %ΔP → interpret magnitude and sign."
    ],
    "answer": "PED magnitude 0.4 → demand is inelastic; price ↑ raises total revenue."
  },
  "practice": {
    "mcq": [
      {"stem":"If |PED|=1.8, a price increase will likely…","options":["raise TR","lower TR","not change TR","cannot say"],"answer":"lower TR"},
      {"stem":"Which raises elasticity?","options":["Fewer substitutes","Shorter time","Necessity","More substitutes"],"answer":"More substitutes"},
      {"stem":"PED is usually negative because…","options":["law of demand","Giffen goods","Veblen effects","income effect only"],"answer":"law of demand"}
    ],
    "saq": [{"stem":"Explain why PED tends to be higher in the long run.","max":4}],
    "laq": [{"stem":"‘Indirect taxes on inelastic goods raise revenue but may be regressive.’ Evaluate.","max":15}],
    "numericals":[
      {"prompt":"P: 10→12, Q: 100→92 (midpoint). Compute PED (magnitude).","expected":"0.4","unit":""}
    ]
  },
  "revision": {
    "notes":[
      "Midpoint formula reduces base effect.",
      "Elasticity changes along linear demand.",
      "Total revenue test: elastic ↔ TR and P move opposite."
    ],
    "exam_prep":["Plan: 2m define → 2m diagram → 4m application → 6m evaluation."]
  }
}
JSON

# ------ IB Math AA: Quadratics ------
cat > content/ib_math_aa/quadratics.json <<'JSON'
{
  "subject":"IB Math AA",
  "level":"SL",
  "topic":"Quadratic Functions — Basics",
  "study_guide":{
    "intro":"Quadratics model parabolic relationships: y=ax^2+bx+c.",
    "concept":[
      "Forms: standard (ax^2+bx+c), vertex a(x-h)^2+k, factored a(x-r1)(x-r2).",
      "Vertex: (-b/2a, f(-b/2a)); axis of symmetry x=-b/2a.",
      "Discriminant Δ=b^2-4ac → roots behaviour."
    ],
    "examples":[
      "Maximum area problems.",
      "Projectile motion (ignoring air resistance)."
    ],
    "key_points":[
      "Sign of a controls opening and concavity.",
      "Δ>0 two roots; Δ=0 one root; Δ<0 complex roots."
    ]
  },
  "worked_example":{
    "steps":[
      "Given y=2x^2-8x+3, compute vertex x=-b/2a=8/4=2.",
      "Vertex y=2(2)^2-8(2)+3=8-16+3=-5 → (2,-5).",
      "Discriminant Δ=b^2-4ac=64-24=40 → two real roots.",
      "Roots via quadratic formula x=(8±√40)/4 = (4±√10)/2."
    ],
    "answer":"Vertex (2,-5); two real roots (4±√10)/2."
  },
  "practice":{
    "mcq":[
      {"stem":"For y=-x^2+6x-5, the parabola opens…","options":["up","down","sideways","cannot say"],"answer":"down"}
    ],
    "saq":[{"stem":"Find the axis of symmetry and vertex for y=3x^2+6x+1.","max":4}],
    "laq":[{"stem":"Explain how the discriminant informs the nature of roots and sketch implications.","max":10}],
    "numericals":[
      {"prompt":"For y=x^2-4x+3, compute Δ and the roots.","expected":"1 and roots 1,3","unit":""}
    ]
  },
  "revision":{
    "notes":[
      "Complete the square to get vertex form.",
      "Δ<0 → no real x-intercepts."
    ],
    "exam_prep":["Sketch → features → solve → interpret."]
  }
}
JSON

# ------ Samples page ------
cat > pages/04_Sample_Units.py <<'PY'
import streamlit as st, json, glob, os
from ui_branding import sidebar_branding, page_watermark
from utils.auth import get_license_info

st.set_page_config(page_title="SkillNestEdu — Sample Units", layout="wide")

if "auth_email" not in st.session_state:
    st.warning("Please login first (Pages ➜ Login)."); st.stop()

lic_email, boards, expiry = get_license_info()
sidebar_branding(lic_email, boards[0] if boards else "IB", expiry); page_watermark(lic_email, expiry)

st.title("Sample Units")
cols = st.columns(2)

def render_card(path):
    data = json.load(open(path))
    with st.container(border=True):
        st.subheader(f"{data['subject']} — {data['topic']}")
        st.caption(f"Level: {data.get('level','')} • Difficulty: {data.get('difficulty','—')}")
        st.write(data["study_guide"]["intro"])
        if st.button(f"Prefill Student Mode with '{data['topic']}'", key=path):
            st.session_state["prefill_subject"] = data["subject"]
            st.session_state["prefill_level"] = data.get("level","SL")
            st.session_state["prefill_board"] = "IB"
            st.session_state["prefill_topic"] = data["topic"]
            st.success("Open Pages ➜ Student Mode and click Generate.")
        with st.expander("Open details"):
            st.write("**Key Points**")
            for k in data["study_guide"]["key_points"]:
                st.write("• " + k)
            st.write("**Sample MCQ**")
            for q in data["practice"].get("mcq", [])[:2]:
                st.write("• " + q["stem"])

paths = sorted(glob.glob("content/ib_econ/*.json")) + sorted(glob.glob("content/ib_math_aa/*.json"))
for i,p in enumerate(paths):
    with cols[i%2]:
        render_card(p)
PY

# ------ Student Mode prefill tweak ------
python - <<'PY'
from pathlib import Path
p = Path("pages/01_Student_Mode.py")
txt = p.read_text()
marker = "topic   = st.sidebar.text_input(\"Topic\", value=\"Price Elasticity of Demand\")"
if marker in txt and "prefill_topic" not in txt:
    txt = txt.replace(
        marker,
        "topic   = st.sidebar.text_input(\"Topic\", value=st.session_state.get('prefill_topic','Price Elasticity of Demand'))"
    ).replace(
        "subject = st.sidebar.selectbox(\"Subject\", [\"IB Economics\",\"IB Math AA\"])",
        "subject = st.sidebar.selectbox(\"Subject\", [\"IB Economics\",\"IB Math AA\"], index=[\"IB Economics\",\"IB Math AA\"].index(st.session_state.get('prefill_subject','IB Economics')))"
    ).replace(
        "level   = st.sidebar.selectbox(\"Level\", [\"SL\",\"HL\"])",
        "level   = st.sidebar.selectbox(\"Level\", [\"SL\",\"HL\"], index=[\"SL\",\"HL\"].index(st.session_state.get('prefill_level','SL')))"
    ).replace(
        "board   = st.sidebar.selectbox(\"Board\", [\"IB\"])",
        "board   = st.sidebar.selectbox(\"Board\", [\"IB\"], index=0)"
    )
    p.write_text(txt)
    print("Patched Student Mode for prefill.")
else:
    print("Student Mode already patched or marker not found.")
PY

echo "✅ Content seeded. Open Pages ➜ Sample Units."
