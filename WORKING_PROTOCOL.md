# SkillNestEdu — Working Protocol (Jeeves Mode)

## Environment Legend (Where to Run)
- **V** → VS Code (editor itself, file browsing/editing)
- **VT** → VS Code Terminal (inside VS Code, project-level commands, python/streamlit)
- **M** → Mac Terminal (system-wide commands, installing tools like Ollama, Homebrew)
- **S** → Streamlit (the app UI in your browser after `streamlit run`)
- **G** → Gamma API (exporting polished decks/learning modules)

---

## Prime Directive
**Always think ahead** so everything is:
- **More automated** (one-run scripts, reusable templates, idempotent)
- **Enhanced** (scalable, smart defaults, config-driven)
- **Visually appealing** (clean UI, consistent branding, accessible)
- **Content-rich** (exam-ready resources, markbands, teacher notes)
- **Easy UI** (zero-ambiguity copy, friendly flows)

---

## Zero-Hallucination & Accuracy (Non-SME Safety)
- **Deterministic generation**: temperature=0, top_p=0.1 (or strict defaults).
- **Must-cite rule** (when applicable): outputs include sources or say “Not enough info” — **never invent**.
- **Refusal over fiction**: if uncertain, the system surfaces gaps, not guesses.
- **Validation layer**: post-gen checks (keywords, equations, unit checks, rubric mapping).
- **Audit trail**: store prompts + responses for replay.
- **Red-flag phrases**: block/flag claims with “always/never/guarantee” unless verified.

---

[rest of your protocol unchanged…]
