import os
import json
from typing import Dict, Any, List

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from google.oauth2 import service_account
from googleapiclient.discovery import build

from content_loader import load_content, ordered_keys

# =============== Page configuration ===============
st.set_page_config(
    page_title="SkillNestEdu Course Engine",
    page_icon="📘",
    layout="centered",
    initial_sidebar_state="expanded",
)

# =============== Google auth (env/secret-based) ===============
SCOPES = [
    "https://www.googleapis.com/auth/documents.readonly",
]


def _get_google_creds():
    raw = None
    if "GOOGLE_SERVICE_ACCOUNT" in st.secrets:
        raw = st.secrets["GOOGLE_SERVICE_ACCOUNT"]
    elif os.environ.get("GOOGLE_SERVICE_ACCOUNT"):
        raw = os.environ["GOOGLE_SERVICE_ACCOUNT"]
    else:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT secret not set in Streamlit secrets or env.")
    info = raw if isinstance(raw, dict) else json.loads(raw)
    return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)


# =============== Diagram functions ===============

def ppc_diagram():
    st.markdown("Use the slider to see shifts in production possibilities.")
    slider = st.slider("Resource / Technology Index (1–10)", 1, 10, 5, key="ppc_slider")

    x = np.linspace(0, 10, 200)
    y = slider - 0.5 * (x ** 2 / 10)
    y = np.maximum(y, 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="PPC", line=dict(color="#2563eb")))

    def y_at(val):
        return slider - 0.5 * (val ** 2 / 10)

    pts_x = [2, 6, 8]
    pts_y = [y_at(2), y_at(6), y_at(8)]
    fig.add_trace(
        go.Scatter(
            x=pts_x,
            y=pts_y,
            mode="markers+text",
            text=["A", "B", "C"],
            textposition="top center",
            marker=dict(size=10, color="#f59e0b"),
            name="Choices",
        )
    )

    fig.update_layout(title="Production Possibility Curve (PPC)", xaxis_title="Good A", yaxis_title="Good B")
    st.plotly_chart(fig, use_container_width=True)

    baseline = 5
    x_intercept = np.sqrt(20 * slider)
    y_intercept = slider
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Max Good A (x‑intercept)", f"{x_intercept:.1f}", delta=f"{x_intercept - np.sqrt(20*baseline):.1f}")
    with c2:
        st.metric("Max Good B (y‑intercept)", f"{y_intercept:.1f}", delta=f"{y_intercept - baseline:.1f}")
    with c3:
        if slider > baseline:
            st.success("Outward shift — more resources/productivity → larger attainable set.")
        elif slider < baseline:
            st.warning("Inward shift — resource loss/shock → smaller attainable set.")
        else:
            st.info("Baseline frontier.")

    st.markdown("**Opportunity Cost (choose a movement):**")
    move = st.radio("", ["B → C (gain A)", "A → B (gain B)"], horizontal=True, key="ppc_move")
    scale_label = st.radio("Units scale", ["per 1 unit", "per 10 units", "per 100 units"], horizontal=True, key="ppc_scale")
    scale = {"per 1 unit": 1, "per 10 units": 10, "per 100 units": 100}[scale_label]

    yA, yB, yC = y_at(2), y_at(6), y_at(8)
    if move.startswith("B"):
        dA, dB = 8 - 6, yC - yB
        oc = abs(dB / dA) if dA != 0 else float("nan")  # B per 1 A
        oc_scaled = oc * scale
        gain_A, loss_B = scale, oc_scaled
        c1, c2, c3 = st.columns(3)
        c1.metric("Gain in A", f"+{gain_A:.0f} units"); c2.metric("Loss in B", f"{loss_B:.1f} units")
        c3.metric(f"OC (B per {scale} A)", f"{oc_scaled:.1f}" if scale > 1 else f"{oc:.2f}")
        st.caption(f"Base movement on the chart is B→C (ΔA={dA}, ΔB={abs(dB):.1f}); scale shows an equivalent ratio for {scale} A.")
    else:
        dB, dA = (yB - yA), 6 - 2
        oc = abs(dA / dB) if dB != 0 else float("nan")  # A per 1 B
        oc_scaled = oc * scale
        gain_B, loss_A = scale, oc_scaled
        c1, c2, c3 = st.columns(3)
        c1.metric("Gain in B", f"+{gain_B:.1f} units"); c2.metric("Loss in A", f"{loss_A:.1f} units")
        c3.metric(f"OC (A per {scale} B)", f"{oc_scaled:.1f}" if scale > 1 else f"{oc:.2f}")
        st.caption(f"Base movement on the chart is A→B (ΔB={abs(dB):.1f}, ΔA={dA}); scale shows an equivalent ratio for {scale} B.")

    st.markdown("_Tip: On a concave PPC, OC rises as you move right—resources are specialized._")


def elasticity_diagram():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[10, 8, 6, 4, 2], y=[2, 4, 6, 8, 10], mode="lines+markers", name="Elastic Demand", line=dict(color="green")))
    fig.add_trace(go.Scatter(x=[10, 9, 8, 7, 6], y=[2, 3, 4, 5, 6], mode="lines+markers", name="Inelastic Demand", line=dict(color="red")))
    fig.update_layout(title="Elastic vs Inelastic Demand", xaxis_title="Price", yaxis_title="Quantity", legend=dict(x=0.7, y=0.95))
    st.plotly_chart(fig, use_container_width=True)


# =============== Learn Mode orchestrator ===============
def learn_mode(subject: str, level: str, topic: str, subtopic: str | None, teacher_mode: bool = False):
    entry = CONFIG[subject][topic][subtopic or "Overview"]

    # Header
    st.markdown(f"## 📘 {level} - {topic}{': ' + subtopic if subtopic else ''}")

    # Step flow (diagram is inserted after Explanation)
    steps = [
        ("Objectives", lambda: render_objectives(entry.get("learning_objectives", []), entry.get("time_estimate"))),
        ("Recap", lambda: render_prereq(entry.get("prereq_recap", []))),
        ("Explain", lambda: render_explanation(entry.get("explanation", []))),
        ("Textbook pointers", lambda: render_textbook_pointers(entry.get("textbook_pointers"))),
        ("Examples", lambda: render_worked_examples(entry.get("worked_examples", []))),
        ("Misconceptions", lambda: render_misconceptions(entry.get("misconceptions", []))),
        ("Command Terms", lambda: render_command_terms(entry.get("command_terms", []))),
        ("Glossary", lambda: render_glossary(entry.get("glossary", {}))),
        ("TOK", lambda: render_tok(entry.get("tok_insight", ""))),
        ("Practice (MCQ)", lambda: render_mcqs(entry.get("mcqs", []))),
        ("Short Questions", lambda: render_short_questions(entry.get("short_questions", []), teacher_mode)),
        ("Extended", lambda: render_extended_questions(entry.get("extended_questions", []), teacher_mode)),
        ("Exit Ticket", lambda: render_exit_ticket(entry.get("exit_ticket", []))),
        ("IA/EE", lambda: render_ia_ee(entry)),
        ("Teacher Notes", lambda: render_teacher_notes(entry.get("teacher_notes", {})) if teacher_mode else None),
    ]
    # Insert diagram after the Explanation step (index 3)
    steps.insert(3, ("Interactive Diagram", lambda: render_diagram_step(entry)))

    # Session state for step navigation
    key = f"step_{subject}_{topic}_{subtopic or 'Overview'}"
    if key not in st.session_state:
        st.session_state[key] = 0
    idx = st.session_state[key]

    # Progress + render current step
    st.progress((idx + 1) / max(1, len(steps)))
    st.caption(f"Step {idx + 1} of {len(steps)} — {steps[idx][0]}")
    steps[idx][1]()

    # Nav buttons
    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("⬅️ Back", key=f"back_{key}_{idx}", disabled=idx == 0):
            st.session_state[key] = max(0, idx - 1)
            st.rerun()
    with col_next:
        if st.button("Next ➡️", key=f"next_{key}_{idx}", disabled=idx == len(steps) - 1):
            st.session_state[key] = min(len(steps) - 1, idx + 1)
            st.rerun()


# =============== UI ===============
st.title("📘 SkillNestEdu Course Engine")
st.subheader("Your AI-powered self-study lesson builder")

subjects = list(CONFIG.keys())
subject = st.selectbox("Choose Subject", subjects)
level = st.selectbox("Choose IB Level", ["IB1", "IB2"])

topics = list(CONFIG[subject].keys())
topic = st.selectbox("Choose a Topic", topics)

subs = list(CONFIG[subject][topic].keys())
subtopic = st.selectbox("Choose Subtopic", [s for s in subs])
sub = None if subtopic == "Overview" else subtopic

# 🔒 Password-gated Teacher View
teacher_toggle = st.checkbox("🔒 Teacher View (password required)")
teacher_mode = False
if teacher_toggle:
    if not st.session_state.get("teacher_ok", False):
        pw = st.text_input("Enter teacher password", type="password", key="teacher_pw")
        if st.button("Unlock", type="primary"):
            expected = _get_teacher_pass()
            if not expected:
                st.warning("Teacher password is not set. Ask admin to set TEACHER_PASS in Streamlit Secrets.")
            elif pw == expected:
                st.session_state["teacher_ok"] = True
                st.success("Teacher View unlocked for this session.")
            else:
                st.error("Incorrect password.")
    teacher_mode = st.session_state.get("teacher_ok", False)
else:
    st.session_state["teacher_ok"] = False

# Start Learn Mode
start_clicked = st.button("Start Learn Mode ▶️", key="start_learn")
signature = f"{subject}|{topic}|{subtopic or 'Overview'}"
if start_clicked:
    st.session_state["learn_active"] = True
    st.session_state["active_signature"] = signature
    st.session_state[f"step_{subject}_{topic}_{subtopic or 'Overview'}"] = 0

if st.session_state.get("learn_active") and st.session_state.get("active_signature") == signature:
    learn_mode(subject, level, topic, sub, teacher_mode)
