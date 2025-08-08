import os
from copy import deepcopy
import yaml
import streamlit as st

def _deep_merge(dest: dict, src: dict):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dest.get(k), dict):
            _deep_merge(dest[k], v)
        else:
            dest[k] = deepcopy(v)
    return dest

def ordered_keys(d: dict, order_key: str = "_order") -> list:
    if not isinstance(d, dict):
        return []
    explicit = d.get(order_key)
    keys = [k for k in (explicit or d.keys()) if k != order_key and k in d]
    if explicit:
        keys += [k for k in d.keys() if k not in explicit and k != order_key]
    return keys

def load_content(content_dir: str = "content") -> dict:
    """Load all YAML files and show a clear error with file + line if one is invalid."""
    config: dict = {}
    if not os.path.isdir(content_dir):
        st.error("Missing /content folder.")
        return config

    for fname in sorted(os.listdir(content_dir)):
        if not fname.lower().endswith((".yml", ".yaml")):
            continue
        path = os.path.join(content_dir, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            st.error(f"❌ YAML error in **{fname}**. Fix the indentation/quotes near the line shown below.\n\n**Details:** {e}")
            st.stop()
        _deep_merge(config, data)
    return config
