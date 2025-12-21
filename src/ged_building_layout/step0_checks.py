# src/ged_building_layout/step0_checks.py
from __future__ import annotations

import os
import json
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
from dataclasses import dataclass
import networkx as nx
import matplotlib.pyplot as plt

from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.validation import explain_validity
from .viz_style import CATEGORY_COLOR_MAP

# optional: only needed for overlay on jpg
try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class Step0Config:
    valid_region_labels: Tuple[str, ...] = tuple(str(i) for i in range(0, 14))
    corridor_label: str = "12"
    door_labels: Tuple[str, ...] = ("14", "16")
    adjacent_label: str = "15"
    adjacent_threshold: float = 0.1

    # colors for function label check
    category_color_map: Optional[Dict[str, str]] = None


def _ensure_cfg(cfg: Optional[Step0Config]) -> Step0Config:
    cfg = cfg or Step0Config()
    if cfg.category_color_map is None:
        object.__setattr__(cfg, "category_color_map", CATEGORY_COLOR_MAP)
    return cfg


# -----------------------------
# Utilities
# -----------------------------
def _convert_rectangle_to_polygon(points: Sequence[Sequence[float]]) -> Polygon:
    (x1, y1), (x2, y2) = points
    return Polygon([[x1, y1], (x2, y1), (x2, y2), (x1, y2)])


def _shape_to_polygon(shape: dict, *, allow_invalid: bool = False) -> Optional[Polygon]:
    """
    Construct a Polygon from a LabelMe shape.
    - If allow_invalid=False: returns None for invalid polygons.
    - If allow_invalid=True : keeps invalid polygons (still drops empty).
    """
    pts = shape.get("points")
    if not pts:
        return None

    if len(pts) >= 3:
        poly = Polygon(pts)
    elif len(pts) == 2:
        poly = _convert_rectangle_to_polygon(pts)
    else:
        return None

    if poly.is_empty:
        return None

    if (not allow_invalid) and (not poly.is_valid):
        return None

    return poly


def _match_jpg(json_path: str, jpg_folder: str) -> Optional[str]:
    stem = os.path.splitext(os.path.basename(json_path))[0]
    jpg_path = os.path.join(jpg_folder, f"{stem}.jpg")
    if os.path.exists(jpg_path):
        return jpg_path
    # sometimes jpeg/png
    for ext in (".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        p = os.path.join(jpg_folder, f"{stem}{ext}")
        if os.path.exists(p):
            return p
    return None


def _load_image_rgb(path: str):
    if cv2 is None:
        raise RuntimeError("opencv-python is required for Step0 overlays. Please install opencv-python.")
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _mkdir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


# -----------------------------
# Check 1: invalid polygons + empty group_id
# -----------------------------
def run_check_invalid_polygons(
    json_folder: str,
    jpg_folder: str,
    out_root: str,
    *,
    cfg: Optional[Step0Config] = None,
) -> str:
    """
    Output: out_root/invalid_polygons/*.png + invalid_polygons_report.csv
    """
    cfg = _ensure_cfg(cfg)
    out_dir = _mkdir(os.path.join(out_root, "invalid_polygons"))
    report_csv = os.path.join(out_dir, "invalid_polygons_report.csv")

    rows = []
    for fn in sorted(os.listdir(json_folder)):
        if not fn.endswith(".json"):
            continue

        json_path = os.path.join(json_folder, fn)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_h = data.get("imageHeight")
        img_w = data.get("imageWidth")

        invalid_polys = []     # (poly, label, group_id, reason)
        empty_gid_polys = []   # (poly, label)

        for shape in data.get("shapes", []):
            label = str(shape.get("label"))
            group_id = shape.get("group_id")

            # IMPORTANT: allow_invalid=True so we can actually detect invalid polygons
            poly = _shape_to_polygon(shape, allow_invalid=True)
            if poly is None:
                continue

            if group_id is None:
                empty_gid_polys.append((poly, label))
                rows.append({"file": fn, "issue": "empty_group_id", "label": label, "group_id": "", "detail": ""})

            if not poly.is_valid:
                reason = explain_validity(poly)
                invalid_polys.append((poly, label, group_id, reason))
                rows.append({"file": fn, "issue": "invalid_polygon", "label": label, "group_id": group_id, "detail": reason})

        if (not invalid_polys) and (not empty_gid_polys):
            continue

        jpg_path = _match_jpg(json_path, jpg_folder)
        fig, ax = plt.subplots(figsize=(12, 10))
        if jpg_path:
            img = _load_image_rgb(jpg_path)
            ax.imshow(img, extent=[0, img_w, img_h, 0])
        else:
            ax.set_title(f"{fn} (NO JPG FOUND)")
            ax.set_xlim(0, img_w)
            ax.set_ylim(img_h, 0)

        # invalid -> red
        for poly, label, gid, _reason in invalid_polys:
            try:
                x, y = poly.exterior.xy
                ax.fill(x, y, facecolor="red", edgecolor="black", alpha=0.6)
                ax.text(poly.centroid.x, poly.centroid.y, f"{label}({gid})", fontsize=8, color="white")
            except Exception:
                # if geometry is too broken to draw, still keep it in report
                pass

        # empty gid -> green
        for poly, label in empty_gid_polys:
            try:
                x, y = poly.exterior.xy
                ax.fill(x, y, facecolor="green", edgecolor="black", alpha=0.6)
                ax.text(poly.centroid.x, poly.centroid.y, f"{label}", fontsize=8, color="white")
            except Exception:
                pass

        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout()

        out_png = os.path.join(out_dir, f"{os.path.splitext(fn)[0]}.png")
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()

    if rows:
        with open(report_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["file", "issue", "label", "group_id", "detail"])
            w.writeheader()
            w.writerows(rows)

    return out_dir


# -----------------------------
# Check 2: corridor polygons + group_id labels
# -----------------------------
def run_check_corridors(
    json_folder: str,
    jpg_folder: str,
    out_root: str,
    *,
    cfg: Optional[Step0Config] = None,
) -> str:
    """
    Output: out_root/corridors/*.png
    """
    import matplotlib.cm as cm

    cfg = _ensure_cfg(cfg)
    out_dir = _mkdir(os.path.join(out_root, "corridors"))

    for fn in sorted(os.listdir(json_folder)):
        if not fn.endswith(".json"):
            continue

        json_path = os.path.join(json_folder, fn)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_h = data.get("imageHeight")
        img_w = data.get("imageWidth")

        corridors: List[Tuple[Polygon, object]] = []
        for shape in data.get("shapes", []):
            label = str(shape.get("label"))
            if label != cfg.corridor_label:
                continue
            gid = shape.get("group_id")
            poly = _shape_to_polygon(shape, allow_invalid=False)
            if poly is None:
                continue
            corridors.append((poly, gid))

        if not corridors:
            continue

        jpg_path = _match_jpg(json_path, jpg_folder)
        fig, ax = plt.subplots(figsize=(12, 10))
        if jpg_path:
            img = _load_image_rgb(jpg_path)
            ax.imshow(img, extent=[0, img_w, img_h, 0])

        unique_ids = sorted(set(gid for _, gid in corridors))
        id_to_color = {gid: cm.tab20(i / max(1, len(unique_ids))) for i, gid in enumerate(unique_ids)}

        for poly, gid in corridors:
            x, y = poly.exterior.xy
            ax.fill(x, y, facecolor=id_to_color.get(gid, "gray"), edgecolor="black", alpha=0.6)
            ax.text(poly.centroid.x, poly.centroid.y, f"{gid}", fontsize=14, color="black")

        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout()

        out_png = os.path.join(out_dir, f"{os.path.splitext(fn)[0]}.png")
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()

    return out_dir


# -----------------------------
# Check 3: function label coloring overlay
# -----------------------------
def run_check_function_labels(
    json_folder: str,
    jpg_folder: str,
    out_root: str,
    *,
    cfg: Optional[Step0Config] = None,
) -> str:
    """
    Output: out_root/function_labels/*.png
    """
    cfg = _ensure_cfg(cfg)
    out_dir = _mkdir(os.path.join(out_root, "function_labels"))

    for fn in sorted(os.listdir(json_folder)):
        if not fn.endswith(".json"):
            continue

        json_path = os.path.join(json_folder, fn)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        img_h = data.get("imageHeight")
        img_w = data.get("imageWidth")

        polys: List[Tuple[Polygon, str]] = []
        for shape in data.get("shapes", []):
            poly = _shape_to_polygon(shape, allow_invalid=False)
            if poly is None:
                continue
            label = str(shape.get("label"))
            polys.append((poly, label))

        jpg_path = _match_jpg(json_path, jpg_folder)
        if not jpg_path:
            continue

        img = _load_image_rgb(jpg_path)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(img, extent=[0, img_w, img_h, 0])

        for poly, label in polys:
            color = cfg.category_color_map.get(label, "gray")  # type: ignore[union-attr]
            x, y = poly.exterior.xy
            ax.fill(x, y, facecolor=color, alpha=0.6)

        ax.set_aspect("equal")
        ax.axis("off")
        plt.tight_layout()

        out_png = os.path.join(out_dir, f"{os.path.splitext(fn)[0]}.png")
        plt.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close()

    return out_dir


# -----------------------------
# Check 4: isolates + connectivity (based on BG logic)
# -----------------------------
def _build_bg_for_connectivity(data: dict, cfg: Step0Config) -> nx.Graph:
    shapes = data.get("shapes", [])

    grouped_polygons: Dict[Tuple[object, str], List[Polygon]] = {}

    for shape in shapes:
        label = str(shape.get("label"))
        gid = shape.get("group_id")
        if gid is None:
            continue
        if label not in set(cfg.valid_region_labels):
            continue

        poly = _shape_to_polygon(shape, allow_invalid=False)
        if poly is None:
            continue

        grouped_polygons.setdefault((gid, label), []).append(poly)

    merged_region_polygons = {(gid, label): unary_union(polys) for (gid, label), polys in grouped_polygons.items()}

    door_polygons: List[Polygon] = []
    adjacent_polygons: List[Polygon] = []

    for shape in shapes:
        label = str(shape.get("label"))
        if label not in (*cfg.door_labels, cfg.adjacent_label):
            continue
        poly = _shape_to_polygon(shape, allow_invalid=False)
        if poly is None:
            continue

        if label in cfg.door_labels:
            door_polygons.append(poly)
        else:
            adjacent_polygons.append(poly)

    G = nx.Graph()
    for (gid, label), poly in merged_region_polygons.items():
        G.add_node((gid, label), pos=poly.centroid.coords[0], functiontype=label)

    for id1, poly1 in merged_region_polygons.items():
        for id2, poly2 in merged_region_polygons.items():
            if id1 >= id2:
                continue

            for door in door_polygons:
                if poly1.intersects(door) and poly2.intersects(door):
                    G.add_edge(id1, id2, type="door")

            for adj in adjacent_polygons:
                if adj.area <= 0:
                    continue
                overlap1 = poly1.intersection(adj).area / adj.area
                overlap2 = poly2.intersection(adj).area / adj.area
                if overlap1 > cfg.adjacent_threshold and overlap2 > cfg.adjacent_threshold:
                    G.add_edge(id1, id2, type="adjacent")

    return G


def run_check_isolates_and_connectivity(
    json_folder: str,
    out_root: str,
    *,
    cfg: Optional[Step0Config] = None,
) -> str:
    """
    Output: out_root/connectivity/connectivity_report.csv
    (This check does not require jpg.)
    """
    cfg = _ensure_cfg(cfg)
    out_dir = _mkdir(os.path.join(out_root, "connectivity"))
    out_csv = os.path.join(out_dir, "connectivity_report.csv")

    rows = []
    for fn in sorted(os.listdir(json_folder)):
        if not fn.endswith(".json"):
            continue
        json_path = os.path.join(json_folder, fn)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        G = _build_bg_for_connectivity(data, cfg)

        isolates = list(nx.isolates(G))
        n_cc = nx.number_connected_components(G) if G.number_of_nodes() > 0 else 0
        lcc_size = 0
        if G.number_of_nodes() > 0:
            lcc_size = max((len(c) for c in nx.connected_components(G)), default=0)

        rows.append({
            "file": fn,
            "n_nodes": G.number_of_nodes(),
            "n_edges": G.number_of_edges(),
            "n_isolates": len(isolates),
            "isolates": ";".join([f"{lab}({gid})" for (gid, lab) in isolates]) if isolates else "",
            "n_connected_components": n_cc,
            "lcc_size": lcc_size,
        })

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["file", "n_nodes", "n_edges", "n_isolates", "isolates", "n_connected_components", "lcc_size"],
        )
        w.writeheader()
        w.writerows(rows)

    return out_dir


# -----------------------------
# Step0 orchestrator: run any subset, write to subfolders
# -----------------------------
def run_step0_checks(
    json_folder: str,
    jpg_folder: str,
    out_root: str,
    *,
    checks: Sequence[str] = ("invalid", "corridor", "labels", "connectivity"),
    cfg: Optional[Step0Config] = None,
) -> Dict[str, str]:
    """
    checks:
      - "invalid"      -> invalid polygons + empty group_id overlay
      - "corridor"     -> corridor overlay (group_id labeled)
      - "labels"       -> function label coloring overlay
      - "connectivity" -> isolates + connectivity csv report
    """
    cfg = _ensure_cfg(cfg)
    out: Dict[str, str] = {}
    _mkdir(out_root)

    for c in checks:
        c = c.lower().strip()
        if c == "invalid":
            out[c] = run_check_invalid_polygons(json_folder, jpg_folder, out_root, cfg=cfg)
        elif c == "corridor":
            out[c] = run_check_corridors(json_folder, jpg_folder, out_root, cfg=cfg)
        elif c == "labels":
            out[c] = run_check_function_labels(json_folder, jpg_folder, out_root, cfg=cfg)
        elif c == "connectivity":
            out[c] = run_check_isolates_and_connectivity(json_folder, out_root, cfg=cfg)
        else:
            raise ValueError(f"Unknown check: {c}")

    return out
