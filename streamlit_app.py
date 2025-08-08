from __future__ import annotations

import os
from typing import Dict, Any, List

import streamlit as st
import plotly.graph_objects as go
import numpy as np

# Load YAML content (built‑in safe loader — ignores external content_loader.py)
import yaml
from copy import deepcopy

def _deep_merge(dest: dict, src: dict):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dest.get(k), dict):
            _deep_merge(dest[k], v)
        else:
            dest[k] = deepcopy(v)
    return dest


def load_content(content_dir: str = "content") -> dict:
    """Load all YAML files from /content and show a clear error (file + line) if one is invalid."""
    config: dict = {}
    if not os.path.isdir(content_dir):
        return config
    for fname in sorted(os.listdir(content_dir)):
        if not fname.lower().endswith((".yml", ".yaml")):
            continue
        path = os.path.join(content_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except Exception as e:
            st.error(f"❌ YAML error in **{fname}**

{type(e).__name__}: {e}")
            st.stop()
        _deep_merge(config, data)
    return config


def ordered_keys(d: dict, order_key: str = "_order") -> list:
    if not isinstance(d, dict):
        return []
    explicit = d.get(order_key)
    keys = [k for k in (explicit or d.keys()) if k != order_key and k in d]
    if explicit:
        keys += [k for k in d.keys() if k not in explicit and k != order_key]
    return keys

# ───────────────── Page config ─────────────────
st.set_page_config(
    page_title="SkillNestEdu Course Engine",
    page_icon="📘",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ───────────────── Teacher password (secrets/env) ─────────────────

def _get_teacher_pass() -> str | None:
    try:
        val = st.secrets.get("TEACHER_PASS")
    except Exception:
        val = None
    return val or os.environ.get("TEACHER_PASS")

# ───────────────── Diagrams ─────────────────

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

    # Live metrics
    baseline = 5
    x_intercept = np.sqrt(20 * slider)  # when B = 0
    y_intercept = slider                 # when A = 0

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

    # Opportunity cost coach
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

# ───────────────── Render helpers ─────────────────

def section_title(text: str):
    st.markdown(f"### {text}")


def render_objectives(objs: List[str], time_estimate: str | None):
    cols = st.columns([3, 1])
    with cols[0]:
        section_title("Learning Objectives")
        for o in objs or []:
            st.markdown(f"- {o}")
    with cols[1]:
        if time_estimate:
            st.caption(f"⏱️ {time_estimate}")


def render_prereq(items: List[str]):
    if not items:
        return
    with st.expander("Prerequisite Recap (2‑minute refresher)"):
        for it in items:
            st.markdown(f"- {it}")


def render_explanation(points: List[str]):
    section_title("Explanation")
    for p in points or []:
        st.markdown(f"- {p}")


def render_worked_examples(examples: List[Dict[str, Any]]):
    if not examples:
        return
    section_title("Worked Examples")
    for i, ex in enumerate(examples, 1):
        with st.expander(f"Example {i}: {ex.get('title','')}"):
            if ex.get("steps"):
                st.markdown("**Steps**")
                for s in ex["steps"]:
                    st.markdown(f"1. {s}")
            if ex.get("notes"):
                st.markdown("**Why these steps?**")
                for n in ex["notes"]:
                    st.markdown(f"- {n}")
            if ex.get("solution"):
                st.markdown("**Model Solution**")
                st.markdown(ex["solution"])


def render_misconceptions(items: List[str]):
    if not items:
        return
    with st.expander("Common Mistakes"):
        for m in items:
            st.markdown(f"- {m}")


def render_command_terms(terms: List[Dict[str, str]]):
    if not terms:
        return
    section_title("Command Terms")
    for ct in terms:
        st.markdown(f"- **{ct['term']}** — {ct['definition']}")
        if ct.get("mini_frame"):
            st.caption(f"Response frame: {ct['mini_frame']}")


def render_tok(tok: str):
    if tok:
        section_title("TOK Insight")
        st.markdown(tok, unsafe_allow_html=True)


def render_short_questions(items: List[Dict[str, Any]], teacher_mode: bool):
    if not items:
        return
    section_title("Short Questions")
    for i, q in enumerate(items, 1):
        with st.expander(f"Q{i}. {q['question']}"):
            for j, hint in enumerate(q.get("hint_chain", []), 1):
                st.markdown(f"*Hint {j}:* {hint}")
            if teacher_mode and q.get("model_answer"):
                st.markdown("**Model Answer**")
                st.markdown(q["model_answer"])


def render_extended_questions(items: List[Dict[str, Any]], teacher_mode: bool):
    if not items:
        return
    section_title("Extended Response")
    for i, q in enumerate(items, 1):
        with st.expander(f"Q{i}. {q['question']}"):
            if q.get("planning_scaffold"):
                st.markdown("**Planning Scaffold**")
                for s in q["planning_scaffold"]:
                    st.markdown(f"- {s}")
            if q.get("criteria"):
                st.caption(q["criteria"])
            if teacher_mode and q.get("model_answer"):
                st.markdown("**Model Answer**")
                st.markdown(q["model_answer"])


def render_mcqs(items: List[Dict[str, Any]], *, subject: str, topic: str, subtopic: str | None):
    if not items:
        return
    section_title("MCQs (check your understanding)")
    for i, mc in enumerate(items, 1):
        key = f"mcq_{subject}_{topic}_{subtopic or 'Overview'}_{i}"
        order = ["A", "B", "C", "D"]
        labels = [f"{k}. {mc['options'][k]}" for k in order if k in mc["options"]]
        choice_label = st.radio(mc["question"], labels, key=key)
        if choice_label:
            choice_key = choice_label.split(".")[0]
            if choice_key == mc["answer"]:
                st.success(f"Correct! {mc['rationales'][choice_key]}")
            else:
                st.error(f"Not quite. {mc['rationales'][choice_key]}")
                st.info(f"✅ Correct answer: {mc['answer']}. {mc['rationales'][mc['answer']]}")


def render_exit_ticket(items: List[str]):
    if not items:
        return
    section_title("Exit Ticket")
    for i, et in enumerate(items, 1):
        st.markdown(f"{i}. {et}")


def render_ia_ee(entry: Dict[str, Any]):
    if not entry.get("ia_scaffold") and not entry.get("ee_scaffold"):
        return
    section_title("IA/EE Scaffolding")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**IA Scaffold**")
        ia = entry.get("ia_scaffold")
        if isinstance(ia, list):
            for it in ia:
                st.markdown(f"- {it}")
        elif ia:
            st.markdown(ia)
    with cols[1]:
        st.markdown("**EE Scaffold**")
        ee = entry.get("ee_scaffold")
        if isinstance(ee, list):
            for it in ee:
                st.markdown(f"- {it}")
        elif ee:
            st.markdown(ee)


def render_teacher_notes(notes: Dict[str, Any]):
    if not notes:
        return
    section_title("Teacher's Notes")
    tps = notes.get("talking_points", [])
    sas = notes.get("sample_answers", [])
    if tps:
        st.markdown("**Talking Points**")
        for tp in tps:
            st.markdown(f"- {tp}")
    if sas:
        st.markdown("**Sample Answers**")
        for sa in sas:
            st.markdown(f"- {sa}")


def render_teacher_panel(entry: Dict[str, Any]):
    """Compact teacher-only dashboard shown in the sidebar."""
    notes = entry.get("teacher_notes", {}) or {}
    st.markdown("### 👩‍🏫 Teacher Panel")

    # Lecture plan / pacing
    if notes.get("lecture_script") or notes.get("pacing"):
        st.markdown("**Lecture plan & pacing**")
        for b in notes.get("lecture_script", []):
            st.markdown(f"- {b}")
        if notes.get("pacing"):
            st.caption(f"⏱️ Pacing: {notes['pacing']}")

    # Socratic prompts
    if notes.get("socratic_prompts"):
        st.markdown("**Socratic prompts**")
        for q in notes["socratic_prompts"]:
            st.markdown(f"- {q}")

    # Misconceptions (from entry)
    if entry.get("misconceptions"):
        st.markdown("**Common misconceptions**")
        for m in entry["misconceptions"]:
            st.markdown(f"- {m}")

    # Board / slide plan
    if notes.get("board_plan"):
        st.markdown("**Board/Slide plan**")
        for b in notes["board_plan"]:
            st.markdown(f"- {b}")

    # Live checks
    if notes.get("live_checks"):
        st.markdown("**Live checks (CFU)**")
        for c in notes["live_checks"]:
            st.markdown(f"- {c}")

    # Textbook pointers
    if entry.get("textbook_pointers"):
        st.markdown("**Textbook pointers (Tragakes)**")
        tps = entry["textbook_pointers"]
        for it in (tps if isinstance(tps, list) else [tps]):
            st.markdown(f"- {it}")

    # Model answers quick access
    if notes.get("sample_answers"):
        with st.expander("Model answers (quick)"):
            for sa in notes["sample_answers"]:
                st.markdown(f"- {sa}")


def render_visuals(items: List[Any]):
    if not items:
        return
    section_title("Visuals")
    n = min(3, len(items))
    cols = st.columns(n)
    for i, itm in enumerate(items):
        with cols[i % n]:
            src, cap = None, ""
            if isinstance(itm, dict):
                src = itm.get("src") or itm.get("url")
                cap = itm.get("caption", "")
            elif isinstance(itm, str):
                src = itm
            if not src:
                continue
            lower = src.lower()
            if lower.endswith((".png", ".jpg", ".jpeg", ".webp", ".gif")):
                st.image(src, caption=cap, use_column_width=True)
            elif lower.endswith(".svg"):
                st.markdown(f'<img src="{src}" style="max-width:100%; height:auto;"/>', unsafe_allow_html=True)
                if cap:
                    st.caption(cap)
            else:
                st.warning(f"Unsupported visual format: {src}. Use PNG/JPG/SVG.")


def render_textbook_pointers(ptrs):
    if not ptrs:
        return
    section_title("Textbook pointers (Tragakes 3rd ed.)")
    if isinstance(ptrs, list):
        for it in ptrs:
            st.markdown(f"- {it}")
    else:
        st.markdown(f"- {ptrs}")


def render_glossary(gloss: Dict[str, str]):
    if not gloss:
        return
    section_title("Glossary")
    for term, definition in gloss.items():
        with st.expander(term):
            st.markdown(definition)


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

# ───────────────── Learn Mode ─────────────────

def learn_mode(CONFIG: dict, subject: str, level: str, topic: str, subtopic: str | None, teacher_mode: bool = False):
    entry = CONFIG[subject][topic][subtopic or "Overview"]
    st.markdown(f"## 📘 {level} - {topic}{': ' + subtopic if subtopic else ''}")

    # Build the step list cleanly (no duplicates). Include a step only if it has content.
    steps: List[tuple[str, callable]] = []

    steps.append(("Objectives", lambda: render_objectives(entry.get("learning_objectives", []), entry.get("time_estimate"))))
    if entry.get("prereq_recap"):
        steps.append(("Recap", lambda: render_prereq(entry.get("prereq_recap", []))))
    if entry.get("explanation"):
        steps.append(("Explain", lambda: render_explanation(entry.get("explanation", []))))
    if entry.get("textbook_pointers"):
        steps.append(("Textbook pointers", lambda: render_textbook_pointers(entry.get("textbook_pointers"))))
    if entry.get("visuals"):
        steps.append(("Visuals", lambda: render_visuals(entry.get("visuals"))))
    if entry.get("diagram"):
        steps.append(("Interactive Diagram", lambda: render_diagram_step(entry)))
    if entry.get("worked_examples"):
        steps.append(("Examples", lambda: render_worked_examples(entry.get("worked_examples", []))))
    if entry.get("misconceptions"):
        steps.append(("Misconceptions", lambda: render_misconceptions(entry.get("misconceptions", []))))
    if entry.get("command_terms"):
        steps.append(("Command Terms", lambda: render_command_terms(entry.get("command_terms", []))))
    if entry.get("glossary"):
        steps.append(("Glossary", lambda: render_glossary(entry.get("glossary", {}))))
    if entry.get("tok_insight"):
        steps.append(("TOK", lambda: render_tok(entry.get("tok_insight", ""))))
    if entry.get("mcqs"):
        steps.append(("Practice (MCQ)", lambda: render_mcqs(entry.get("mcqs", []), subject=subject, topic=topic, subtopic=subtopic)))
    if entry.get("short_questions"):
        steps.append(("Short Questions", lambda: render_short_questions(entry.get("short_questions", []), teacher_mode)))
    if entry.get("extended_questions"):
        steps.append(("Extended", lambda: render_extended_questions(entry.get("extended_questions", []), teacher_mode)))
    if entry.get("exit_ticket"):
        steps.append(("Exit Ticket", lambda: render_exit_ticket(entry.get("exit_ticket", []))))
    if entry.get("ia_scaffold") or entry.get("ee_scaffold"):
        steps.append(("IA/EE", lambda: render_ia_ee(entry)))
    if teacher_mode and entry.get("teacher_notes"):
        steps.append(("Teacher Notes", lambda: render_teacher_notes(entry.get("teacher_notes", {}))))

    # Navigation state
    key = f"step_{subject}_{topic}_{subtopic or 'Overview'}"
    if key not in st.session_state:
        st.session_state[key] = 0
    idx = st.session_state[key]
    idx = max(0, min(idx, len(steps) - 1))

    # Progress + render
    if steps:
        st.progress((idx + 1) / len(steps))
        st.caption(f"Step {idx + 1} of {len(steps)} — {steps[idx][0]}")
        steps[idx][1]()
    else:
        st.info("No content for this subtopic yet.")

    # Nav buttons
    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("⬅️ Back", key=f"back_{key}_{idx}", disabled=idx == 0):
            st.session_state[key] = max(0, idx - 1)
            st.rerun()
    with col_next:
        if st.button("Next ➡️", key=f"next_{key}_{idx}", disabled=idx >= len(steps) - 1):
            st.session_state[key] = min(len(steps) - 1, idx + 1)
            st.rerun()

# ───────────────── UI ─────────────────

st.title("📘 SkillNestEdu Course Engine")
st.subheader("Your AI-powered self-study lesson builder")

CONFIG = load_content("content")
if not CONFIG:
    st.error("No content found. Create the /content folder with YAML files.")
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

    # Start / Resume Learn Mode (reset when dropdowns change)
    selection_signature = f"{subject}|{topic}|{sub or 'Overview'}"
    if st.session_state.get("active_signature") and st.session_state["active_signature"] != selection_signature:
        st.session_state["learn_active"] = False
        st.session_state["active_signature"] = selection_signature

    # 👩‍🏫 Sidebar teacher panel (visible only to teachers)
    if teacher_mode:
        entry_preview = CONFIG[subject][topic][sub or "Overview"]
        with st.sidebar:
            render_teacher_panel(entry_preview)

    start_clicked = st.button("Start Learn Mode ▶️", key="start_learn")
    if start_clicked:
        st.session_state["learn_active"] = True
        st.session_state["active_signature"] = selection_signature
        st.session_state[f"step_{subject}_{topic}_{sub or 'Overview'}"] = 0

    if st.session_state.get("learn_active") and st.session_state.get("active_signature") == selection_signature:
        learn_mode(CONFIG, subject, level, topic, sub, teacher_mode)

