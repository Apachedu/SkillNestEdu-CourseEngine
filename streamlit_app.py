# ================= SkillNestEdu Course Engine (FULL) =================
# Streamlit app: self‑study lesson builder with teacher view
# - Loads content from /content/*.yml via content_loader.py
# - Password‑gated teacher sidebar
# - Interactive PPC + Elasticity diagrams
# ====================================================================

from __future__ import annotations

import os
import json
from typing import Dict, Any, List, Tuple

import streamlit as st
import plotly.graph_objects as go
import numpy as np

from content_loader import load_content, ordered_keys

# =============== Page configuration ===============
st.set_page_config(
    page_title="SkillNestEdu Course Engine",
    page_icon="📘",
    layout="centered",
    initial_sidebar_state="expanded",
)

# =============== Utilities ===============

def section_title(text: str):
    st.markdown(f"### {text}")


def _get_teacher_pass() -> str | None:
    """Read teacher password from Streamlit secrets or env (TEACHER_PASS)."""
    try:
        val = st.secrets.get("TEACHER_PASS")
    except Exception:
        val = None
    return val or os.environ.get("TEACHER_PASS")


# =============== Diagrams ===============

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


# =============== Render helpers ===============

def render_objectives(items: List[str], time_estimate: str | None):
    section_title("Learning Objectives")
    if time_estimate:
        st.caption(f"⏱️ {time_estimate}")
    for it in items or []:
        st.markdown(f"- {it}")


def render_prereq(items: List[str]):
    if not items:
        st.info("No quick recap for this step.")
        return
    with st.expander("Prerequisite Recap (2-minute refresher)", True):
        for it in items:
            st.markdown(f"- {it}")


def render_explanation(lines: List[str]):
    for ln in lines or []:
        st.markdown(f"- {ln}")


def render_textbook_pointers(items: List[str] | None):
    if not items:
        return
    with st.expander("Textbook pointers", True):
        for it in items:
            st.markdown(f"- {it}")


def render_worked_examples(examples: List[Dict[str, Any]]):
    if not examples:
        return
    for ex in examples:
        with st.expander(f"Example: {ex.get('title','')}"):
            for s in ex.get("steps", []):
                st.markdown(f"- {s}")
            if ex.get("notes"):
                st.caption("**Notes:**")
                for n in ex["notes"]:
                    st.markdown(f"- {n}")
            if ex.get("solution"):
                st.success(ex["solution"])


def render_misconceptions(items: List[str]):
    if not items:
        return
    st.markdown("**Common misconceptions**")
    for it in items:
        st.markdown(f"- {it}")


def render_command_terms(items: List[Dict[str, Any]]):
    if not items:
        return
    with st.expander("Command terms"):
        for x in items:
            st.markdown(f"- **{x.get('term','')}** — {x.get('definition','')}")
            if x.get("mini_frame"):
                st.caption(x["mini_frame"])


def render_glossary(items: Dict[str, str]):
    if not items:
        return
    with st.expander("Glossary"):
        for k, v in items.items():
            st.markdown(f"- **{k}:** {v}")


def render_tok(text: str):
    if not text:
        return
    section_title("Theory of Knowledge (TOK)")
    st.info(text)


def render_mcqs(questions: List[Dict[str, Any]]):
    if not questions:
        return
    score = 0
    for i, q in enumerate(questions, 1):
        st.markdown(f"**Q{i}. {q['question']}**")
        choice = st.radio("", list(q["options"].keys()), key=f"mcq_{i}")
        if choice == q["answer"]:
            st.success("Correct!")
            score += 1
        else:
            st.error(f"Incorrect. Correct answer: {q['answer']}")
        if q.get("rationales"):
            st.caption(q["rationales"][choice])
    st.info(f"Score: {score}/{len(questions)}")


def render_short_questions(qs: List[Dict[str, Any]], teacher_mode: bool):
    if not qs:
        return
    for i, q in enumerate(qs, 1):
        with st.expander(f"Short Question {i}"):
            st.markdown(q["question"])
            if q.get("hint_chain"):
                with st.expander("Hints"):
                    for h in q["hint_chain"]:
                        st.markdown(f"- {h}")
            st.text_area("Your answer", key=f"sq_{i}")
            if teacher_mode and q.get("model_answer"):
                st.success(f"Model answer: {q['model_answer']}")


def render_extended_questions(qs: List[Dict[str, Any]], teacher_mode: bool):
    if not qs:
        return
    for i, q in enumerate(qs, 1):
        with st.expander(f"Extended Response {i}"):
            st.markdown(q["question"])
            if q.get("planning_scaffold"):
                st.caption("Planning scaffold:")
                for p in q["planning_scaffold"]:
                    st.markdown(f"- {p}")
            st.text_area("Outline / Answer", key=f"er_{i}")
            if teacher_mode and q.get("model_answer"):
                st.success(f"Model answer: {q['model_answer']}")


def render_exit_ticket(items: List[str]):
    if not items:
        return
    with st.expander("Exit Ticket"):
        for it in items:
            st.markdown(f"- {it}")


def render_ia_ee(entry: Dict[str, Any]):
    ia = entry.get("ia_scaffold") or []
    ee = entry.get("ee_scaffold") or []
    if not (ia or ee):
        return
    with st.expander("IA / EE Scaffolding"):
        if ia:
            st.markdown("**IA ideas**")
            for it in ia:
                st.markdown(f"- {it}")
        if ee:
            st.markdown("**EE ideas**")
            for it in ee:
                st.markdown(f"- {it}")


def render_teacher_notes(notes: Dict[str, Any]):
    if not notes:
        st.info("No additional teacher notes for this section.")
        return
    with st.expander("Detailed teacher notes"):
        for k in ["talking_points", "sample_answers"]:
            if notes.get(k):
                st.markdown(f"**{k.replace('_',' ').title()}**")
                for it in notes[k]:
                    st.markdown(f"- {it}")


# =============== Teacher sidebar (per subtopic) ===============

def render_teacher_panel(entry: Dict[str, Any]):
    with st.sidebar:
        st.markdown("### 🧑‍🏫 Teacher Panel")
        tn = entry.get("teacher_notes", {})
        if tn.get("pacing"):
            st.caption(f"⏱️ {tn['pacing']}")
        if tn.get("lecture_script"):
            with st.expander("Lecture script", True):
                for x in tn["lecture_script"]:
                    st.markdown(f"- {x}")
        if tn.get("socratic_prompts"):
            with st.expander("Socratic prompts"):
                for x in tn["socratic_prompts"]:
                    st.markdown(f"- {x}")
        if tn.get("board_plan"):
            with st.expander("Board plan"):
                for x in tn["board_plan"]:
                    st.markdown(f"- {x}")
        if tn.get("live_checks"):
            with st.expander("Live checks"):
                for x in tn["live_checks"]:
                    st.markdown(f"- {x}")
        # Helpful always
        if entry.get("textbook_pointers"):
            with st.expander("Textbook pointers"):
                for x in entry["textbook_pointers"]:
                    st.markdown(f"- {x}")
        if entry.get("misconceptions"):
            with st.expander("Common misconceptions"):
                for x in entry["misconceptions"]:
                    st.markdown(f"- {x}")


# =============== Learn Mode orchestrator ===============

def render_diagram_step(entry: Dict[str, Any]):
    func_name = entry.get("diagram")
    if not func_name:
        return
    section_title("Interactive Diagram")
    pre = entry.get("diagram_preamble") or (entry.get("explanation") or [None])[0]
    if pre:
        st.info(pre)
    func = globals().get(func_name)
    if callable(func):
        func()
        if entry.get("alt_text"):
            st.caption(f"Alt-text: {entry['alt_text']}")


def learn_mode(CONFIG: dict, subject: str, level: str, topic: str, subtopic: str | None, teacher_mode: bool = False):
    entry = CONFIG[subject][topic][subtopic or "Overview"]
    st.markdown(f"## 📘 {level} - {topic}{': ' + subtopic if subtopic else ''}")

    # Show teacher sidebar per subtopic
    if teacher_mode:
        render_teacher_panel(entry)

    # Steps (auto-hide if no content)
    steps: List[Tuple[str, Any]] = [
        ("Objectives", lambda: render_objectives(entry.get("learning_objectives", []), entry.get("time_estimate"))),
        entry.get("prereq_recap") and ("Recap", lambda: render_prereq(entry.get("prereq_recap", []))),
        ("Explain", lambda: render_explanation(entry.get("explanation", []))),
        entry.get("textbook_pointers") and ("Textbook pointers", lambda: render_textbook_pointers(entry.get("textbook_pointers"))),
        entry.get("diagram") and ("Interactive Diagram", lambda: render_diagram_step(entry)),
        entry.get("worked_examples") and ("Examples", lambda: render_worked_examples(entry.get("worked_examples", []))),
        entry.get("misconceptions") and ("Misconceptions", lambda: render_misconceptions(entry.get("misconceptions", []))),
        entry.get("command_terms") and ("Command Terms", lambda: render_command_terms(entry.get("command_terms", []))),
        entry.get("glossary") and ("Glossary", lambda: render_glossary(entry.get("glossary", {}))),
        entry.get("tok_insight") and ("TOK", lambda: render_tok(entry.get("tok_insight", ""))),
        entry.get("mcqs") and ("Practice (MCQ)", lambda: render_mcqs(entry.get("mcqs", []))),
        entry.get("short_questions") and ("Short Questions", lambda: render_short_questions(entry.get("short_questions", []), teacher_mode)),
        entry.get("extended_questions") and ("Extended", lambda: render_extended_questions(entry.get("extended_questions", []), teacher_mode)),
        entry.get("exit_ticket") and ("Exit Ticket", lambda: render_exit_ticket(entry.get("exit_ticket", []))),
        (entry.get("ia_scaffold") or entry.get("ee_scaffold")) and ("IA/EE", lambda: render_ia_ee(entry)),
        teacher_mode and ("Teacher Notes", lambda: render_teacher_notes(entry.get("teacher_notes", {}))),
    ]
    steps = [s for s in steps if s]

    key = f"step_{subject}_{topic}_{subtopic or 'Overview'}"
    if key not in st.session_state:
        st.session_state[key] = 0
    idx = st.session_state[key]

    st.progress((idx + 1) / max(1, len(steps)))
    st.caption(f"Step {idx + 1} of {len(steps)} — {steps[idx][0]}")
    steps[idx][1]()

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

CONFIG = load_content("content")
if not CONFIG:
    st.error("No content found. Create the /content folder with YAML files as in the Split Pack.")
else:
    subjects = ordered_keys(CONFIG)
    subject = st.selectbox("Choose Subject", subjects)
    level = st.selectbox("Choose IB Level", ["IB1", "IB2"])

    topics = ordered_keys(CONFIG[subject])
    topic = st.selectbox("Choose a Topic", topics)

    subs = ordered_keys(CONFIG[subject][topic])
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

    start_clicked = st.button("Start Learn Mode ▶️", key="start_learn")
    signature = f"{subject}|{topic}|{sub or 'Overview'}"
    if start_clicked:
        st.session_state["learn_active"] = True
        st.session_state["active_signature"] = signature
        st.session_state[f"step_{subject}_{topic}_{sub or 'Overview'}"] = 0

    if st.session_state.get("learn_active") and st.session_state.get("active_signature") == signature:
        learn_mode(CONFIG, subject, level, topic, sub, teacher_mode)
