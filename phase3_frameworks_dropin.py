import pathlib, textwrap

ROOT = pathlib.Path(".").resolve()

FILES = {
"frameworks/__init__.py": "",
"frameworks/router.py": r'''from typing import Dict, List

def _bloom_tag(level: str) -> str:
    levels = {"R":"Remember","U":"Understand","A":"Apply","An":"Analyze","E":"Evaluate","C":"Create"}
    return levels.get(level, level)

def _gagne_skeleton(topic:str, objectives:List[str], content_blocks:List[str], practice:Dict, transfer:Dict, tok_ia_ee:Dict=None) -> Dict:
    return {
        "intro": {
            "hook": f"Real-world hook for {topic}: a quick scenario to grab attention.",
            "recall": ["MCQ warm-up 1 (R)", "MCQ warm-up 2 (R)"],
            "objectives": objectives
        },
        "content": [
            {"text": b, "diagram_spec": {"hint": "Add interactive diagram/axes labels"}, "bloom": _bloom_tag("U")}
            for b in content_blocks
        ],
        "guidance": {"teacher_notes": ["Scaffold difficult terms", "Use dual-coding: text + diagram"]},
        "practice": practice,
        "feedback": {"rubric": ["Clarity","Use of terminology","Application/analysis","Conclusion/structure"]},
        "assessment": {"past_paper_style": True, "bloom_mix": ["A","An","E"]},
        "transfer": transfer,
        "tok_ia_ee": tok_ia_ee or {}
    }

def _ib_econ_bm(topic:str) -> Dict:
    objectives = [f"Explain {topic}", f"Apply {topic} to a real market", f"Analyse welfare/efficiency effects"]
    content_blocks = [
        f"Definition(s) and core intuition for {topic}.",
        f"Diagram(s) to represent {topic} with clear axes and shifts.",
        f"Interpretation: what changes, who gains/loses, assumptions/limitations."
    ]
    practice = {
        "mcq": [{"stem": f"{topic}: identify the correct diagram shift", "options": ["A","B","C","D"], "answer": "B"} for _ in range(5)],
        "saq": [{"stem": f"Explain {topic} with reference to a labelled diagram.", "max": 6, "bloom": _bloom_tag("A")}],
        "laq": [{"stem": f"Evaluate the impact of {topic} on different stakeholders.", "max": 15, "bloom": _bloom_tag("E")}],
        "data_response": {"table_or_chart":"Add small dataset for interpretation"}
    }
    transfer = {"mini_case": "Short case with command term (Analyse/Evaluate)", "sim_pointer": "interactive/case_studies/registry.json"}
    tok_ia_ee = {
        "tok_q": f"How do we know {topic} holds across contexts?",
        "ia_seed": f"Recent article idea linking to {topic}",
        "ee_link": f"Potential EE RQ angle for {topic}"
    }
    return _gagne_skeleton(topic, objectives, content_blocks, practice, transfer, tok_ia_ee)

def _ib_math(topic:str) -> Dict:
    objectives = [f"Define and use the formulae for {topic}", f"Solve 10 numericals with steps", f"Interpret results"]
    content_blocks = [
        f"Key definitions and theorems for {topic}.",
        f"Worked example 1 with step-by-step solution.",
        f"Worked example 2 with common pitfalls and checks."
    ]
    practice = {
        "numericals": [{"stem": f"Solve a problem on {topic} (show steps)", "answer": "Final numeric", "bloom": _bloom_tag("A")} for _ in range(10)],
        "mcq": [{"stem": f"{topic}: choose the correct step/identity", "options": ["A","B","C","D"], "answer":"A"} for _ in range(5)],
        "saq": [{"stem": f"Explain why a step is valid in {topic}.", "max": 4, "bloom": _bloom_tag("An")}]
    }
    transfer = {"mini_challenge": "Apply to a modelling task / IA-style prompt"}
    return _gagne_skeleton(topic, objectives, content_blocks, practice, transfer)

def _ib_science(topic:str) -> Dict:
    objectives = [f"Explain {topic} with correct scientific vocabulary", f"Analyse data related to {topic}", f"Design/critique an IA method"]
    content_blocks = [
        f"Concept overview for {topic} with units and definitions.",
        f"Worked example with calculation/uncertainty.",
        f"Data interpretation: trends, anomalies, sources of error."
    ]
    practice = {
        "mcq": [{"stem": f"{topic}: pick correct concept/units", "options":["A","B","C","D"], "answer":"C"} for _ in range(5)],
        "saq": [{"stem": f"Justify a choice of method for {topic}.", "max": 4, "bloom": _bloom_tag("An")}],
        "laq": [{"stem": f"Evaluate an experiment design for {topic}.", "max": 10, "bloom": _bloom_tag("E")}]
    }
    transfer = {"lab_planning": "Hypothesis + variables + method sketch"}
    return _gagne_skeleton(topic, objectives, content_blocks, practice, transfer)

def _ib_cs(topic:str) -> Dict:
    objectives = [f"Explain {topic} in CS terms", f"Trace/derive algorithm steps", f"Analyse complexity and trade-offs"]
    content_blocks = [
        f"Definitions and contexts for {topic}.",
        f"Pseudocode and trace table for {topic}.",
        f"Complexity discussion and edge cases."
    ]
    practice = {
        "mcq": [{"stem": f"{topic}: pick correct output for input", "options":["A","B","C","D"], "answer":"D"} for _ in range(5)],
        "saq": [{"stem": f"Complete the trace table for {topic}.", "max": 4, "bloom": _bloom_tag("A")}],
        "laq": [{"stem": f"Evaluate implementation choices for {topic}.", "max": 10, "bloom": _bloom_tag("E")}]
    }
    transfer = {"project_snippet": "Small coding task or design question"}
    return _gagne_skeleton(topic, objectives, content_blocks, practice, transfer)

def _indian_board(topic:str) -> Dict:
    objectives = [f"Define {topic}", f"Solve typical numericals on {topic}", f"Answer board-style LAQs on {topic}"]
    content_blocks = [
        f"Theory explanation for {topic}, aligned with NCERT/University syllabus.",
        f"Worked numerical example for {topic}.",
        f"Common mistakes and exam tips."
    ]
    practice = {
        "numericals": [{"stem": f"{topic}: compute value given data", "answer":"numeric", "bloom":_bloom_tag('A')} for _ in range(5)],
        "saq": [{"stem": f"Define/Explain {topic}.", "max": 3, "bloom": _bloom_tag("R")}],
        "laq": [{"stem": f"Discuss/Evaluate {topic} in detail.", "max": 8, "bloom": _bloom_tag("E")}]
    }
    transfer = {"india_case": "Small India-context caselet (GST/RBI/startups)"}
    return _gagne_skeleton(topic, objectives, content_blocks, practice, transfer)

def _language_skill(topic:str, course:str) -> Dict:
    objectives = [f"Understand band descriptors for {course}", f"Apply strategy for {topic}", f"Produce and self-assess a response"]
    content_blocks = [
        f"{course} strategy for {topic}: structure, timing, common errors.",
        f"Model response snippets: do vs don't.",
        f"Self-check list mapped to band descriptors."
    ]
    practice = {
        "prompt": {"stem": f"{course} prompt for {topic}", "timer": True},
        "saq": [{"stem": "Rewrite sentence for clarity/grammar", "max": 2, "bloom": _bloom_tag("U")}],
        "laq": [{"stem": "Write full response; reflect with band checklist", "max": 9, "bloom": _bloom_tag("E")}]
    }
    transfer = {"role_play": "Mock interview or debate scenario"}
    return _gagne_skeleton(topic, objectives, content_blocks, practice, transfer)

def _route(board:str, subject:str) -> str:
    b = (board or "").strip().lower()
    s = (subject or "").strip().lower()
    if b == "ib":
        if s in ["economics","ib economics","business management","ib business management","bm"]:
            return "ib_econ_bm"
        if s in ["math aa","math ai","mathematics","ib math aa","ib math ai"]:
            return "ib_math"
        if s in ["physics","chemistry","biology","ib physics","ib chemistry","ib biology"]:
            return "ib_science"
        if s in ["computer science","cs","ib computer science"]:
            return "ib_cs"
        return "ib_generic"
    if b in ["cbse","icse","ug"]:
        return "indian_board"
    if b in ["ielts","pte","soft skills","spoken english","spoken"]:
        return "language_skill"
    return "generic"

def generate_auto_package(subject:str, level:str, board:str, topic:str, difficulty:str="Beginner") -> Dict:
    route = _route(board, subject)
    if route == "ib_econ_bm":
        pkg = _ib_econ_bm(topic)
    elif route == "ib_math":
        pkg = _ib_math(topic)
    elif route == "ib_science":
        pkg = _ib_science(topic)
    elif route == "ib_cs":
        pkg = _ib_cs(topic)
    elif route == "indian_board":
        pkg = _indian_board(topic)
    elif route == "language_skill":
        course = subject.upper() if subject else "Language"
        pkg = _language_skill(topic, course)
    else:
        pkg = _indian_board(topic)

    return {
        "meta": {"subject": subject, "level": level, "board": board, "topic": topic, "difficulty": difficulty},
        "lesson": pkg
    }
''',
"engine_v2.py": r'''import datetime
from frameworks.router import generate_auto_package

def generate_package(subject:str, level:str, board:str, topic:str, difficulty:str="Beginner"):
    pkg = generate_auto_package(subject, level, board, topic, difficulty)
    pkg["meta"]["generated_at"] = datetime.datetime.utcnow().isoformat()+"Z"
    # Back-compat fields used by pages:
    return {
        "meta": pkg["meta"],
        "study_guide": {
            "intro": f"{topic}: why it matters (auto).",
            "concept": [b.get("text","") for b in pkg["lesson"].get("content",[])],
            "diagram_spec": {"hint": "interactive diagrams coming here"},
            "examples": ["Auto example 1","Auto example 2"],
            "key_points": ["Key A","Key B","Key C"]
        },
        "worked_example": {"steps": ["Step 1","Step 2","Step 3"], "answer": "Final answer"},
        "practice": pkg["lesson"].get("practice", {"mcq": [], "saq": [], "laq": []}),
        "revision": {"notes": ["Rev note 1","Rev note 2"], "exam_prep": ["10 min outline → 25 min write → 5 min check"]}
    }
''',
}

for rel, content in FILES.items():
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    print("Wrote", rel)

print("\nFramework router installed. Next:\n- git add frameworks engine_v2.py\n- git commit -m \"frameworks: auto-router + ID skeletons\"\n- git push\n- Reboot Streamlit Cloud\n")

