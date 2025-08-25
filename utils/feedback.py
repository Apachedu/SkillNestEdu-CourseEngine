import re, json

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
