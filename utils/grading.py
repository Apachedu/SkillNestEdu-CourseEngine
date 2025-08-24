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
