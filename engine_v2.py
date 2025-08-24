import datetime

def generate_package(subject:str, level:str, board:str, topic:str, difficulty:str="Beginner"):
    """Stub generator—wire to Gamma API later."""
    return {
        "meta": {
            "subject": subject, "level": level, "board": board, "topic": topic,
            "generated_at": datetime.datetime.utcnow().isoformat()+"Z"
        },
        "study_guide": {
            "intro": f"{topic}: why it matters (2–3 lines).",
            "concept": [f"Definition of {topic}.", "Key idea #1", "Key idea #2"],
            "diagram_spec": {"hint": "axes/labels; add interactive later"},
            "examples": ["Real-world example 1", "Real-world example 2"],
            "key_points": ["Point A","Point B","Point C"]
        },
        "worked_example": {
            "steps": ["Step 1","Step 2","Step 3"], "answer": "Final answer"
        },
        "practice": {
            "mcq": [{"stem":"Sample MCQ?","options":["A","B","C","D"],"answer":"A"} for _ in range(5)],
            "saq": [{"stem":"Explain … (4 marks)","max":4,"criteria":["accuracy","clarity","use_of_terms","examples"]}],
            "laq": [{"stem":"Evaluate … (15 marks)","max":15,"criteria":["knowledge","analysis","evaluation","structure","terms"]}]
        },
        "revision": {
            "notes": ["Rev note 1","Rev note 2","Rev note 3"],
            "exam_prep": ["Plan: 10 min outline → 25 min write → 5 min check"]
        }
    }
