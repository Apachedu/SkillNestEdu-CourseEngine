import os
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
    """Load and deep-merge all .yml/.yaml files under content_dir into one CONFIG dict."""
    config: dict = {}
    if not os.path.isdir(content_dir):
        return config
    for fname in sorted(os.listdir(content_dir)):
        if not fname.lower().endswith((".yml", ".yaml")):
            continue
        path = os.path.join(content_dir, fname)
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        _deep_merge(config, data)
    return config

def ordered_keys(d: dict, order_key: str = "_order") -> list:
    """Return keys of a mapping honoring an optional _order list, hiding the order key itself."""
    if not isinstance(d, dict):
        return []
    explicit = d.get(order_key)
    keys = [k for k in (explicit or d.keys()) if k != order_key and k in d]
    if explicit:
        keys += [k for k in d.keys() if k not in explicit and k != order_key]
    return keys
