import datetime
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
