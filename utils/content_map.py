from __future__ import annotations
import os, functools
import yaml
import streamlit as st
from typing import Dict, Any, Optional

# Import diagram functions here so we can resolve by name from YAML
from utils.diagrams import ppc_diagram, elasticity_diagram

# Map of function names in YAML -> actual callables
DIAGRAM_REGISTRY = {
    "ppc_diagram": ppc_diagram,
    "elasticity_diagram": elasticity_diagram,
}

@functools.lru_cache(maxsize=1)
def _load_yaml() -> Dict[str, Any]:
    path = os.path.join("content", "economics.yaml")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def _normalize_subject(subject: str) -> str:
    # Map UI subjects to YAML subjects
    s = (subject or "").strip().lower()
    if s in {"ib economics", "economics"}:
        return "Economics"
    return subject  # fallback

def get_topic_tree(subject: str) -> Dict[str, Dict[str, Any]]:
    data = _load_yaml()
    return data.get(_normalize_subject(subject), {}) if isinstance(data, dict) else {}

def get_blocks(subject: str, topic: str) -> Dict[str, Dict[str, Any]]:
    tree = get_topic_tree(subject)
    return tree.get(topic, {}) if isinstance(tree, dict) else {}

def render_block(block: Dict[str, Any]) -> None:
    btype = (block.get("type") or "").lower()
    if btype == "markdown":
        st.markdown(block.get("content", ""))
    elif btype == "diagram":
        func_name = block.get("function", "")
        fn = DIAGRAM_REGISTRY.get(func_name)
        if fn:
            fn()
        else:
            st.warning(f"Diagram function '{func_name}' not found.")
    else:
        st.info("No renderer for this block type yet.")

def render_auto_ui(subject: str, topic: str) -> None:
    """Renders a selector of available blocks for the chosen subject/topic."""
    blocks = get_blocks(subject, topic)
    if not blocks:
        st.caption("No mapped interactive/markdown content for this topic yet.")
        return
    names = list(blocks.keys())
    choice = st.selectbox("Interactive Unit", names, index=0)
    st.divider()
    render_block(blocks[choice])
