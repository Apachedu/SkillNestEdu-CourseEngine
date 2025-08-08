import os
import json
from typing import Dict, Any, List

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import yaml
from google.oauth2 import service_account
from googleapiclient.discovery import build

_APP_README = """
SkillNestEdu Course Engine (Streamlit)
- Self-study Learn Mode with teacher view
- Secret-based Google auth (no credentials file committed)

Run locally:
  pip install -r requirements.txt
  streamlit run streamlit_app.py

Required packages (requirements.txt):
  streamlit
  plotly
  numpy
  google-auth
  google-api-python-client
  PyYAML
"""

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
    """Return Google Credentials from Streamlit secrets or env var GOOGLE_SERVICE_ACCOUNT.
    You only need this if you call the Google Docs/Drive APIs below.
    """
    raw = None
    if "GOOGLE_SERVICE_ACCOUNT" in st.secrets:
        raw = st.secrets["GOOGLE_SERVICE_ACCOUNT"]
    elif os.environ.get("GOOGLE_SERVICE_ACCOUNT"):
        raw = os.environ["GOOGLE_SERVICE_ACCOUNT"]
    else:
        raise RuntimeError("GOOGLE_SERVICE_ACCOUNT secret not set in Streamlit secrets or env.")
    info = raw if isinstance(raw, dict) else json.loads(raw)
    return service_account.Credentials.from_service_account_info(info, scopes=SCOPES)


# Helper: Teacher password from secrets/env

def _get_teacher_pass() -> str | None:
    """Return teacher password from Streamlit secrets or env var TEACHER_PASS.
    If not set, returns None.
    """
    try:
        val = st.secrets.get("TEACHER_PASS")  # None if missing
    except Exception:
        val = None
    return val or os.environ.get("TEACHER_PASS")

# Optional: Google Docs helpers (not required for current UI)
DOCUMENT_ID = "1NOykWSpT31a2vbwlPQRIjAAexqpeOi1UQcBAP6t-pgU"

@st.cache_data(show_spinner=False, ttl=3600)
def fetch_doc_text(doc_id: str) -> str:
    creds = _get_google_creds()
    service = build("docs", "v1", credentials=creds)
    doc = service.documents().get(documentId=doc_id).execute()
    body = doc.get("body", {}).get("content", [])
    chunks = []
    for el in body:
        p = el.get("paragraph")
        if p:
            for run in p.get("elements", []):
                chunks.append(run.get("textRun", {}).get("content", ""))
    return "".join(chunks)


# =============== Diagram functions ===============

def ppc_diagram():
    st.markdown("Use the slider to see shifts in production possibilities.")
    slider = st.slider("Resource / Technology Index (1–10)", 1, 10, 5, key="ppc_slider")

    # Curve
    x = np.linspace(0, 10, 200)
    y = slider - 0.5 * (x ** 2 / 10)
    y = np.maximum(y, 0)  # keep chart tidy at extremes

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="PPC", line=dict(color="#2563eb")))

    # Three labelled points we talk about below
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

    # —— Dynamic explanation cards ——
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

    # —— Live opportunity cost coach ——
    st.markdown("**Opportunity Cost (choose a movement):**")
    move = st.radio(
        "",
        ["B → C (gain A)", "A → B (gain B)"],
        horizontal=True,
        key="ppc_move",
    )
    scale_label = st.radio(
        "Units scale",
        ["per 1 unit", "per 10 units", "per 100 units"],
        horizontal=True,
        key="ppc_scale",
    )
    scale = {"per 1 unit": 1, "per 10 units": 10, "per 100 units": 100}[scale_label]

    yA, yB, yC = y_at(2), y_at(6), y_at(8)

    if move.startswith("B"):
        dA, dB = 8 - 6, yC - yB
        oc = abs(dB / dA) if dA != 0 else float("nan")  # B per 1 A
        oc_scaled = oc * scale
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Gain in A", f"+{dA:.0f} units")
        with c2:
            st.metric("Loss in B", f"{abs(dB):.1f} units")
        with c3:
            st.metric(f"OC (B per {scale} A)", f"{oc_scaled:.0f}" if scale > 1 else f"{oc:.2f}")
        st.caption(f"Per unit view: **{oc:.2f} B per 1 A**.")
    else:
        dB, dA = (yB - yA), 6 - 2
        oc = abs(dA / dB) if dB != 0 else float("nan")  # A per 1 B
        oc_scaled = oc * scale
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Gain in B", f"+{dB:.1f} units")
        with c2:
            st.metric("Loss in A", f"{abs(dA):.0f} units")
        with c3:
            st.metric(f"OC (A per {scale} B)", f"{oc_scaled:.0f}" if scale > 1 else f"{oc:.2f}")
        st.caption(f"Per unit view: **{oc:.2f} A per 1 B**.")

    st.markdown("_Tip: On a concave PPC, OC rises as you move right—resources are specialized._")


def elasticity_diagram():
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[10, 8, 6, 4, 2],
            y=[2, 4, 6, 8, 10],
            mode="lines+markers",
            name="Elastic Demand",
            line=dict(color="green"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[10, 9, 8, 7, 6],
            y=[2, 3, 4, 5, 6],
            mode="lines+markers",
            name="Inelastic Demand",
            line=dict(color="red"),
        )
    )
    fig.update_layout(title="Elastic vs Inelastic Demand", xaxis_title="Price", yaxis_title="Quantity", legend=dict(x=0.7, y=0.95))
    st.plotly_chart(fig, use_container_width=True)


# =============== Content CONFIG (rich, self-study) ===============
# Schema per subtopic:
#   learning_objectives: [str]
#   prereq_recap: [str]
#   explanation: [str]
#   worked_examples: [{title, steps: [str], notes: [str], solution: str}]
#   misconceptions: [str]
#   command_terms: [{term, definition, mini_frame}]
#   short_questions: [{question, hint_chain:[str], model_answer:str}]
#   extended_questions: [{question, planning_scaffold:[str], criteria:str, model_answer:str}]
#   mcqs: [{question, options:{A,B,C,D}, answer:'A', rationales:{A,B,C,D}}]
#   exit_ticket: [str]
#   ia_scaffold: str | [str]
#   ee_scaffold: str | [str]
#   teacher_notes: {talking_points:[str], sample_answers:[str]}
#   time_estimate: str
#   diagram: 'ppc_diagram' | 'elasticity_diagram' | None
#   alt_text: str

CONFIG = yaml.safe_load(
    """
Economics:
  Foundations of Economics:
    Overview:
      time_estimate: "60 minutes"
      learning_objectives:
        - "Define economics, scarcity, choice, and opportunity cost."
        - "Distinguish positive vs normative statements with examples."
        - "Differentiate microeconomics and macroeconomics."
        - "Identify the four factors of production and their rewards."
        - "Explain how economic systems answer the three basic questions."
      prereq_recap:
        - "Basic graph reading; correlation vs causation."
        - "Percent & ratio calculations (for later topics)."
      explanation:
        - "Economics studies how societies allocate scarce resources among competing wants."
        - "Scarcity implies choice; every choice involves opportunity cost (next best alternative forgone)."
        - "Microeconomics focuses on individual markets; macroeconomics on the economy as a whole."
        - "Positive statements are testable; normative statements are value judgments."
        - "Factors of production: land (rent), labour (wages), capital (interest), entrepreneurship (profit)."
        - "Three basic questions: what to produce, how to produce, for whom to produce."
        - "Economic systems: market, planned (command), mixed—each uses price signals vs central planning to answer the questions."
        - "Models and ceteris paribus help simplify complexity; they are approximations, not reality."
        - "PPC links: scarcity → limited frontier; choice → movement along; growth/decline → shifts."
      worked_examples:
        - title: "Classify statements: positive or normative"
          steps:
            - "'Inflation fell to 4.8%' vs 'The central bank should cut rates now'."
            - "Ask: can we test the statement with data?"
            - "Label each as positive/normative and justify briefly."
          notes:
            - "Words like 'should' often signal normative."
          solution: "First is positive (testable with data). Second is normative (value judgement about policy)."
        - title: "Compute opportunity cost from a simple choice"
          steps:
            - "You have 2 hours to study: 1h Econ + 1h Math, or 2h Econ."
            - "If you choose 2h Econ, what is your opportunity cost?"
            - "State it explicitly."
          notes:
            - "Opportunity cost is the best alternative forgone, not all alternatives."
          solution: "1 hour of Math study is the opportunity cost of the extra hour of Econ."
        - title: "Identify factors of production in a scenario"
          steps:
            - "A bakery uses an oven, flour, bakers, and the owner's talent to design recipes."
            - "Map each item to FoP: land, labour, capital, enterprise."
          notes:
            - "Capital is man-made aid to production; land includes natural resources."
          solution: "Oven = capital; flour = input from land; bakers = labour; owner = entrepreneurship."
      misconceptions:
        - "Scarcity equals poverty (No—rich economies still face scarcity)."
        - "Opportunity cost is the money price (It may be time, convenience, etc.)."
        - "Positive = good, normative = bad (No—these are methodological categories)."
      command_terms:
        - term: Define
          definition: "Give the precise meaning of a word/phrase."
          mini_frame: "Definition → concise example (optional)."
        - term: Distinguish
          definition: "Make clear the differences between two or more concepts."
          mini_frame: "Criterion 1/2/3 → How they differ."
        - term: Explain
          definition: "Give a detailed account including reasons or causes."
          mini_frame: "Cause → Mechanism → Effect."
        - term: Evaluate
          definition: "Make an appraisal by weighing strengths and limitations."
          mini_frame: "Criteria → Pros/Cons → Judgement."
      short_questions:
        - question: "Define scarcity and explain why it necessitates choice."
          hint_chain:
            - "Link limited resources to unlimited wants."
            - "State the consequence for decision-making."
          model_answer: "Scarcity is the condition of finite resources and unlimited wants; it forces choice about resource allocation."
        - question: "Distinguish positive and normative statements with one example of each."
          hint_chain:
            - "Ask: is it testable?"
            - "Use 'should' as a clue for normative."
          model_answer: "Positive: 'Minimum wage increased by 5%' (testable). Normative: 'Minimum wage should rise' (value judgement)."
        - question: "Identify the factor of production and reward: a software developer's salary."
          hint_chain:
            - "Think labour, land, capital, enterprise."
          model_answer: "Labour; reward is wages."
      extended_questions:
        - question: "Evaluate whether a free‑market system or a planned system is better at answering the three basic economic questions."
          planning_scaffold:
            - "Define both systems briefly."
            - "Criteria: efficiency, equity, innovation, stability."
            - "Use examples; consider mixed economies."
            - "Conclude with reasoned judgement."
          criteria: "AO3 Evaluate"
          model_answer: "Markets harness price signals and incentives for efficiency and innovation, but may underprovide public goods/equity; planning coordinates at scale but faces information/incentive problems. Most adopt mixed systems; effectiveness depends on institutions and context."
        - question: "To what extent can opportunity cost be zero?"
          planning_scaffold:
            - "Clarify 'next best alternative'."
            - "Consider idle resources vs true 'free' goods."
            - "Use PPC and time‑allocation examples."
          criteria: "AO3 Evaluate"
          model_answer: "True zero OC is rare except with free goods or slack time where no alternative is forgone; generally, some alternative use exists, so OC > 0."
      mcqs:
        - question: "Which is a normative statement?"
          options: {A: "Unemployment is 7%", B: "The government should reduce unemployment", C: "Exports rose last quarter", D: "The tax rate is 18%"}
          answer: B
          rationales: {A: "Testable (positive).", B: "Contains 'should'—value judgement.", C: "Testable.", D: "Testable."}
        - question: "Microeconomics primarily studies …"
          options: {A: "National output and inflation", B: "Behaviour of individual firms and consumers", C: "Global trade balances", D: "Fiscal policy"}
          answer: B
          rationales: {A: "Macro focus.", B: "Correct.", C: "Macro/international.", D: "Macro policy."}
        - question: "The reward for the factor 'land' is …"
          options: {A: "Wages", B: "Interest", C: "Rent", D: "Profit"}
          answer: C
          rationales: {A: "Labour.", B: "Capital.", C: "Correct.", D: "Enterprise."}
        - question: "Opportunity cost refers to …"
          options: {A: "The money paid for a good", B: "The next best alternative forgone", C: "All alternatives not chosen", D: "Sunk cost"}
          answer: B
          rationales: {A: "Price not OC.", B: "Correct.", C: "Only the next best matters.", D: "Past costs are irrelevant."}
        - question: "Which system relies mainly on price signals?"
          options: {A: "Planned economy", B: "Market economy", C: "Traditional economy", D: "Autarky"}
          answer: B
          rationales: {A: "Central planning.", B: "Correct.", C: "Customs.", D: "Self‑sufficiency (not a system design)."}
        - question: "A statement 'The central bank should lower interest rates' is …"
          options: {A: "Positive", B: "Normative", C: "Descriptive", D: "Empirical"}
          answer: B
          rationales: {A: "Not testable as a fact.", B: "Correct.", C: "Too vague.", D: "Relates to data, not value."}
      exit_ticket:
        - "Write one positive and one normative statement about a current issue."
        - "State one trade‑off you faced today and its opportunity cost."
      ia_scaffold:
        - "Find a recent news article illustrating scarcity or trade‑offs."
        - "Identify relevant concepts (FoP, OC, market vs command)."
        - "Sketch a PPC or simple diagram if applicable and explain the shift/movement."
      ee_scaffold:
        - "Draft an EE RQ around 'How effectively does [policy] address [scarcity] in [country]?'"
        - "List data sources and a tentative method."
      teacher_notes:
        talking_points:
          - "Use local examples of scarcity (e.g., water restrictions) to ground concepts."
          - "Clarify 'positive vs normative' with classroom poll."
        sample_answers:
          - "Short Q: Labour/wages; Land/rent; Capital/interest; Enterprise/profit."
    Production Possibility Curve:
      time_estimate: "45 minutes"
      diagram: ppc_diagram
      alt_text: "Concave PPC showing trade‑offs between Good A and Good B."
      learning_objectives:
        - "Define scarcity and opportunity cost in the context of PPC."
        - "Interpret points inside, on, and outside the PPC."
        - "Explain outward/inward shifts and their causes."
      prereq_recap:
        - "Factors of production: land, labour, capital, enterprise."
        - "Basic graph reading (axes, intercepts)."
      explanation:
        - "The PPC models maximum feasible combinations given fixed resources and technology."
        - "Concavity arises because resources are not equally efficient across goods (increasing opportunity cost)."
        - "Movement along the curve represents reallocation; shift of the curve represents capacity change."
        - "Points: inside = inefficiency/unemployment; on = productive efficiency; outside = currently unattainable."
      worked_examples:
        - title: "Identify opportunity cost on a PPC"
          steps:
            - "Start at point B (6 of A, 4 of B)."
            - "Move to point C (8 of A, 2 of B)."
            - "Compute the loss in Good B and the gain in Good A."
          notes:
            - "Use absolute changes (ΔA, ΔB)."
            - "Phrase the opportunity cost as 'giving up ... to gain ...'."
          solution: "Moving from B to C: +2 units of Good A costs −2 units of Good B; OC of each extra A is 1 B."
        - title: "Shift due to technology"
          steps:
            - "Assume a tech improvement in Good A only."
            - "Redraw/visualize a pivot—A‑intercept increases, B‑intercept unchanged."
            - "Explain effect on attainable set and trade‑offs."
          notes:
            - "Partial (biased) outward shift indicates sector‑specific productivity."
          solution: "The PPC pivots outward around the B‑intercept; more A can be produced at any level, increasing choice set."
      misconceptions:
        - "Outward shift always benefits everyone equally (distribution may vary)."
        - "Any point outside is 'inefficient'—it is unattainable, not inefficient."
      command_terms:
        - term: Define
          definition: "Give the precise meaning of a word/phrase."
          mini_frame: "State definition in 1–2 lines; avoid examples."
        - term: Explain
          definition: "Give a detailed account including reasons or causes."
          mini_frame: "Cause → Mechanism → Effect."
        - term: Evaluate
          definition: "Make an appraisal by weighing strengths and limitations."
          mini_frame: "Criteria → Pros/Cons → Judgement."
      short_questions:
        - question: "Define opportunity cost using a PPC example."
          hint_chain:
            - "What do you give up when you choose one point over another?"
            - "Compute Δ in the other good when moving along the curve."
          model_answer: "Opportunity cost is the good forgone when choosing another combination; e.g., moving from (6A,4B) to (8A,2B) costs 2B."
        - question: "What does a point inside the PPC indicate?"
          hint_chain:
            - "Think employment of resources."
            - "Is output maximized for given inputs?"
          model_answer: "Inefficiency/unemployment—economy can increase output of at least one good without sacrificing the other."
        - question: "Why is the PPC usually concave to the origin?"
          hint_chain:
            - "Are resources equally adaptable?"
            - "Think increasing marginal opportunity cost."
          model_answer: "Because resources are specialized, reallocating them raises the OC of additional units, making the curve concave."
      extended_questions:
        - question: "Evaluate the impact of immigration on a country's PPC in the short and long run."
          planning_scaffold:
            - "Identify which FoP changes."
            - "Short‑run vs long‑run capacity and productivity."
            - "Winners/losers and distributional effects."
            - "Conclude with justified judgement."
          criteria: "AO3 Evaluate: criteria + balanced analysis + reasoned conclusion."
          model_answer: "Immigration increases labour, shifting PPC outward; short‑run pressures may limit gains; long‑run human capital can raise productivity. Distribution depends on sectoral demand/policy."
        - question: "To what extent does technological change eliminate scarcity?"
          planning_scaffold:
            - "Differentiate shift vs movement along PPC."
            - "Consider absolute vs relative scarcity."
            - "Environmental/resource constraints."
          criteria: "AO3 Evaluate."
          model_answer: "Tech shifts PPC outward but scarcity persists due to unlimited wants and finite resources; composition effects and externalities matter."
      mcqs:
        - question: "A point outside the PPC represents …"
          options: {A: "unattainable with current resources", B: "productive efficiency", C: "allocative efficiency", D: "underutilization"}
          answer: A
          rationales: {A: "Correct: beyond current capacity.", B: "Efficiency is on the curve.", C: "Allocative efficiency needs preferences; not implied.", D: "Inside, not outside."}
        - question: "If all resources become unemployed, the economy moves …"
          options: {A: "along the PPC", B: "inside the PPC", C: "outside the PPC", D: "the PPC pivots outward"}
          answer: B
          rationales: {A: "Movement along keeps efficiency.", B: "Correct: inefficiency.", C: "Still unattainable.", D: "Unemployment doesn't change capacity."}
        - question: "A biased tech improvement in Good A causes the PPC to …"
          options: {A: "shift inward", B: "pivot outward on B‑axis", C: "shift parallel outward", D: "not change"}
          answer: B
          rationales: {A: "Improvement increases capacity.", B: "Correct: A‑intercept increases.", C: "Parallel shift implies neutral tech.", D: "Tech does change it."}
        - question: "Opportunity cost from moving from X to Y is measured by …"
          options: {A: "loss of Good A", B: "gain in Good B", C: "loss of the alternative good", D: "sum of both goods"}
          answer: C
          rationales: {A: "Depends on direction.", B: "Wrong direction.", C: "Correct: what you give up.", D: "Not how OC is defined."}
        - question: "Which statement is true?"
          options: {A: "All points on PPC are allocatively efficient", B: "All points on PPC are productively efficient", C: "Points inside are unattainable", D: "Points outside are inefficient"}
          answer: B
          rationales: {A: "Allocative depends on preferences.", B: "Correct by definition.", C: "Inside are attainable but inefficient.", D: "Outside are unattainable."}
        - question: "An outward parallel shift most likely results from …"
          options: {A: "increase in labour only", B: "neutral technological progress", C: "reallocation of resources", D: "higher unemployment"}
          answer: B
          rationales: {A: "Biased to labour‑intensive good, not parallel.", B: "Correct: raises productivity in both goods.", C: "Movement along, not shift.", D: "Moves inside, not shift."}

  Elasticity:
    Price Elasticity Interactive:
      time_estimate: "50 minutes"
      diagram: elasticity_diagram
      alt_text: "Two demand curves: steep (inelastic) and flat (elastic)."
      learning_objectives:
        - "Define and compute PED."
        - "Interpret elastic vs inelastic demand with revenue implications."
      prereq_recap:
        - "Percentage change calculations."
        - "Reading demand curves."
      explanation:
        - "PED = %ΔQd / %ΔP; magnitude indicates responsiveness."
        - "Elastic (>1): small price change → larger % change in Qd; Inelastic (<1): opposite."
        - "Total revenue moves opposite price in elastic range, same direction in inelastic range."
      worked_examples:
        - title: "Compute PED using midpoint method"
          steps:
            - "Initial P=10, Q=100 → New P=12, Q=90."
            - "Compute %ΔP and %ΔQ using midpoint formula."
            - "Calculate PED and interpret."
          notes:
            - "Midpoint avoids base effects."
          solution: "%ΔP= (12-10)/((12+10)/2)=2/11 ≈ 18.18%; %ΔQ=(90-100)/95=−10.53%; PED≈0.58 (inelastic)."
      misconceptions:
        - "Elasticity equals slope (it does not)."
      command_terms:
        - term: Calculate
          definition: "Obtain a numerical answer showing steps."
          mini_frame: "State formula → Substitute → Compute → Interpret."
        - term: Discuss
          definition: "Offer a considered review of arguments."
          mini_frame: "For → Against → Conditions → Mini‑conclusion."
      short_questions:
        - question: "State two determinants of PED."
          hint_chain:
            - "Think necessity vs luxury."
            - "Consider substitutes and time."
          model_answer: "Availability of substitutes, proportion of income, necessity vs luxury, time period."
        - question: "When price falls and TR rises, what is demand elasticity in that range?"
          hint_chain:
            - "Think TR test."
          model_answer: "Elastic (>1)."
      extended_questions:
        - question: "Evaluate the usefulness of PED for a public transport authority."
          planning_scaffold:
            - "Objectives: revenue vs coverage."
            - "Short‑run vs long‑run elasticity."
            - "Equity/externalities."
          criteria: "AO3 Evaluate"
          model_answer: "PED guides pricing; SR inelastic may raise TR, LR may be elastic due to alternatives; consider social goals and externalities."
      mcqs:
        - question: "PED is defined as …"
          options: {A: "%ΔP/%ΔQd", B: "%ΔQd/%ΔP", C: "%ΔQd×%ΔP", D: "%ΔP−%ΔQd"}
          answer: B
          rationales: {A: "Inverse of PED.", B: "Correct.", C: "Not a ratio.", D: "Not a definition."}
        - question: "Demand is unit elastic when …"
          options: {A: "PED=0", B: "PED=1", C: "PED>1", D: "PED<1"}
          answer: B
          rationales: {A: "Perfectly inelastic.", B: "Correct.", C: "Elastic.", D: "Inelastic."}
        - question: "If price rises and TR falls, demand is …"
          options: {A: "elastic", B: "inelastic", C: "unit elastic", D: "perfectly inelastic"}
          answer: A
          rationales: {A: "Correct: TR moves opposite price.", B: "TR would rise.", C: "TR unchanged.", D: "TR rises with price."}
        - question: "Which increases PED?"
          options: {A: "Fewer substitutes", B: "Shorter time horizon", C: "Luxury nature", D: "Smaller income share"}
          answer: C
          rationales: {A: "Decreases PED.", B: "Less time to adjust → lower PED.", C: "Correct.", D: "Lower sensitivity."}
        - question: "A steeper demand curve at a point implies …"
          options: {A: "always lower PED", B: "always higher PED", C: "nothing—slope ≠ elasticity", D: "PED=1"}
          answer: C
          rationales: {A: "Depends on P/Q.", B: "Same issue.", C: "Correct.", D: "Not implied."}
        - question: "TR is maximized when …"
          options: {A: "price is highest", B: "PED=1", C: "PED>1", D: "PED<1"}
          answer: B
          rationales: {A: "Not necessarily.", B: "Correct: unit elasticity.", C: "TR rises when price falls.", D: "TR rises when price rises."}

Business Management:
  Business Organization:
    Overview:
      time_estimate: "40 minutes"
      learning_objectives:
        - "Differentiate between profit and non-profit aims."
        - "Compare ownership forms."
      prereq_recap:
        - "Basic stakeholder concept."
      explanation:
        - "Businesses coordinate resources to create value."
        - "Ownership forms affect liability, control, and finance."
      worked_examples: []
      misconceptions:
        - "Non-profits do not make any surplus (they can, but reinvest)."
      command_terms:
        - term: Compare
          definition: "Give an account of similarities and differences."
          mini_frame: "Similarities → Differences → Mini-conclusion."
      short_questions:
        - question: "List three ownership forms and one feature each."
          hint_chain: ["Think liability and control."]
          model_answer: "Sole trader (unlimited liability), Partnership (shared control), Corporation (limited liability)."
      extended_questions:
        - question: "Discuss whether social enterprises can be as financially sustainable as for-profits."
          planning_scaffold: ["Define sustainability", "Revenue models", "Cost structures", "Impact measurement"]
          criteria: "AO3 Discuss"
          model_answer: "Under certain models (hybrids, cross-subsidy) they can; depends on market, governance, and funding."
      mcqs:
        - question: "Which structure limits owner liability?"
          options: {A: "Sole trader", B: "Partnership", C: "Corporation", D: "General partnership"}
          answer: C
          rationales: {A: "Unlimited liability.", B: "Partners usually have unlimited liability.", C: "Correct.", D: "General partners have unlimited liability."}
      exit_ticket: ["Define stakeholder in one line."]
      ia_scaffold: ["Design stakeholder interview questions."]
      ee_scaffold: ["Build a case study framework for governance analysis."]
      teacher_notes:
        talking_points: ["Clarify liability vs control."]
        sample_answers: ["MCQ rationale examples as above."]
"""
)


# =============== Render helpers ===============

def section_title(text: str):
    st.markdown(f"### {text}")


def render_objectives(objs: List[str], time_estimate: str | None):
    cols = st.columns([3, 1])
    with cols[0]:
        section_title("Learning Objectives")
        for o in objs:
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
    for p in points:
        st.markdown(f"- {p}")


def render_worked_examples(examples: List[Dict[str, Any]]):
    if not examples:
        return
    section_title("Worked Examples")
    for i, ex in enumerate(examples, 1):
        with st.expander(f"Example {i}: {ex['title']}"):
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


def render_mcqs(items: List[Dict[str, Any]]):
    if not items:
        return
    section_title("MCQs (check your understanding)")
    for i, mc in enumerate(items, 1):
        key = f"mcq_{i}"
        # Stable option order A→D to avoid widget churn
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


# =============== Learn Mode orchestrator ===============

def learn_mode(subject: str, level: str, topic: str, subtopic: str | None, teacher_mode: bool = False):
    entry = CONFIG[subject][topic][subtopic or "Overview"]

    # Header
    st.markdown(f"## 📘 {level} - {topic}{': ' + subtopic if subtopic else ''}")

    # Diagram first if any
    func_name = entry.get("diagram")
    if func_name:
        func = globals().get(func_name)
        if callable(func):
            func()
            if entry.get("alt_text"):
                st.caption(f"Alt-text: {entry['alt_text']}")

    steps = [
        ("Objectives", lambda: render_objectives(entry.get("learning_objectives", []), entry.get("time_estimate"))),
        ("Recap", lambda: render_prereq(entry.get("prereq_recap", []))),
        ("Explain", lambda: render_explanation(entry.get("explanation", []))),
        ("Examples", lambda: render_worked_examples(entry.get("worked_examples", []))),
        ("Misconceptions", lambda: render_misconceptions(entry.get("misconceptions", []))),
        ("Command Terms", lambda: render_command_terms(entry.get("command_terms", []))),
        ("TOK", lambda: render_tok(entry.get("tok_insight", ""))),
        ("Practice (MCQ)", lambda: render_mcqs(entry.get("mcqs", []))),
        ("Short Questions", lambda: render_short_questions(entry.get("short_questions", []), teacher_mode)),
        ("Extended", lambda: render_extended_questions(entry.get("extended_questions", []), teacher_mode)),
        ("Exit Ticket", lambda: render_exit_ticket(entry.get("exit_ticket", []))),
        ("IA/EE", lambda: render_ia_ee(entry)),
        ("Teacher Notes", lambda: render_teacher_notes(entry.get("teacher_notes", {})) if teacher_mode else None),
    ]

    # Session state for step navigation
    key = f"step_{subject}_{topic}_{subtopic or 'Overview'}"
    if key not in st.session_state:
        st.session_state[key] = 0
    idx = st.session_state[key]

    # Progress and navigation
    st.progress((idx + 1) / max(1, len(steps)))
    st.caption(f"Step {idx + 1} of {len(steps)} — {steps[idx][0]}")

    # Render current step
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

subjects = list(CONFIG.keys())
subject = st.selectbox("Choose Subject", subjects)
level = st.selectbox("Choose IB Level", ["IB1", "IB2"])

topics = list(CONFIG[subject].keys())
topic = st.selectbox("Choose a Topic", topics)
subs = list(CONFIG[subject][topic].keys())
subtopic = st.selectbox("Choose Subtopic", ["Overview"] + [s for s in subs if s != "Overview"])  # Overview default
sub = subtopic if subtopic != "Overview" else None

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

# Start Learn Mode: set active signature in session state, render exactly once per run
signature = f"{subject}|{topic}|{sub or 'Overview'}"
start_clicked = st.button("Start Learn Mode ▶️", key="start_learn")
if start_clicked:
    st.session_state["learn_active"] = True
    st.session_state["active_signature"] = signature
    # reset step position when starting a new lesson
    st.session_state[f"step_{subject}_{topic}_{sub or 'Overview'}"] = 0

if st.session_state.get("learn_active") and st.session_state.get("active_signature") == signature:
    learn_mode(subject, level, topic, sub, teacher_mode)
