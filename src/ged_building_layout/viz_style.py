# src/ged_building_layout/viz_style.py
from __future__ import annotations
from typing import Dict

# -----------------------------
# Node category color map
# -----------------------------
CATEGORY_COLOR_MAP: Dict[str, str] = {
    "0": "#a6cee3", "1": "#1f78b4", "2": "#b2df8a", "3": "#33a02c",
    "4": "#fb9a99", "5": "#e31a1c", "6": "#fdbf6f", "7": "#ff7f00",
    "8": "#cab2d6", "9": "#6a3d9a",
    "10": "gray", "11": "gray",
    "12": "#ffff99", "13": "#b15928",
    "17": "#66c2a5",
    "18": "white",   # unknown / fallback
}

# -----------------------------
# Edge style map (shared)
# -----------------------------
EDGE_STYLE_MAP: Dict[str, Dict[str, float]] = {
    "corridor": {"width": 1.0, "alpha": 0.3},
    "door": {"width": 2.0, "alpha": 0.6},
    "adjacent": {"width": 3.0, "alpha": 1.0},
}
