from __future__ import annotations

import os
import json
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import networkx as nx
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
from .viz_style import CATEGORY_COLOR_MAP, EDGE_STYLE_MAP



@dataclass(frozen=True)
class BaseGraphConfig:
    adjacent_threshold: float = 0.1  # overlap ratio on adjacent polygon area
    valid_region_labels: Tuple[str, ...] = tuple(str(i) for i in range(0, 14))
    door_labels: Tuple[str, ...] = ("14", "16")  # treat 16 as door too
    adjacent_label: str = "15"

    # node attrs
    pos_key: str = "pos"
    function_key: str = "functiontype"
    area_key: str = "area"

    # edge attrs
    edge_type_key: str = "type"

    # behavior
    drop_isolates: bool = True
    skip_empty_graph: bool = False   # if True: do not save pkls with 0 nodes or 0 edges


def _convert_rectangle_to_polygon(points: np.ndarray) -> Polygon:
    (x1, y1), (x2, y2) = points
    return Polygon([[x1, y1], (x2, y1), (x2, y2), (x1, y2)])


def _shape_to_polygon(shape: dict) -> Optional[Polygon]:
    pts = np.array(shape.get("points", []), dtype=np.float32)
    if len(pts) >= 3:
        poly = Polygon(pts)
    elif len(pts) == 2:
        poly = _convert_rectangle_to_polygon(pts)
    else:
        return None
    if (not poly.is_valid) or poly.is_empty:
        return None
    return poly


def build_base_graph_from_labelme_json(data: dict, cfg: BaseGraphConfig) -> nx.Graph:
    """
    Build base connectivity graph from one LabelMe JSON.
    Nodes: merged polygons per (group_id, label), with pos=centroid and area=polygon.area.
    Edges:
      - door: if two regions both intersect the same door polygon
      - adjacent: if two regions overlap the same adjacent polygon above threshold
    """
    shapes = data.get("shapes", [])
    if not shapes:
        return nx.Graph()

    valid_labels = set(cfg.valid_region_labels)

    # 1) merge region polygons by (group_id, label)
    grouped_polygons: Dict[Tuple[object, str], List[Polygon]] = defaultdict(list)

    for shape in shapes:
        label = str(shape.get("label"))
        group_id = shape.get("group_id")

        if group_id is None or label not in valid_labels:
            continue

        poly = _shape_to_polygon(shape)
        if poly is None:
            continue

        grouped_polygons[(group_id, label)].append(poly)

    if not grouped_polygons:
        return nx.Graph()

    merged_region_polygons: Dict[Tuple[object, str], Polygon] = {
        key: unary_union(polys) for key, polys in grouped_polygons.items()
    }

    # 2) collect door polygons + adjacent polygons
    door_polygons: List[Polygon] = []
    adjacent_polygons: List[Polygon] = []

    for shape in shapes:
        label = str(shape.get("label"))
        if label not in (*cfg.door_labels, cfg.adjacent_label):
            continue

        poly = _shape_to_polygon(shape)
        if poly is None:
            continue

        if label in cfg.door_labels:
            door_polygons.append(poly)
        elif label == cfg.adjacent_label:
            adjacent_polygons.append(poly)

    # 3) build graph + nodes
    G = nx.Graph()

    for node_id, region_poly in merged_region_polygons.items():
        # Note: centroid may fail for some geometries; guard it
        try:
            centroid = region_poly.centroid.coords[0]
        except Exception:
            # fallback: use bounds center
            minx, miny, maxx, maxy = region_poly.bounds
            centroid = ((minx + maxx) / 2.0, (miny + maxy) / 2.0)

        gid, label = node_id
        G.add_node(
            node_id,
            **{
                cfg.pos_key: centroid,
                cfg.function_key: str(label),
                cfg.area_key: float(region_poly.area),
            },
        )

    # ---- Performance improvement ----
    # Instead of double looping all pairs and then checking each door/adj polygon,
    # we build a spatial index for regions and only test candidates that intersect each door/adj.
    region_keys = list(merged_region_polygons.keys())
    region_polys = [merged_region_polygons[k] for k in region_keys]
    tree = STRtree(region_polys)
    poly_to_key = {id(p): k for p, k in zip(region_polys, region_keys)}

    # 4a) door edges: for each door polygon, find intersecting regions; connect all pairs
    for door in door_polygons:
        candidates = tree.query(door)
        hit_keys = []
        for p in candidates:
            if p.intersects(door):
                hit_keys.append(poly_to_key[id(p)])
        if len(hit_keys) >= 2:
            # connect all pairs
            for i in range(len(hit_keys)):
                for j in range(i + 1, len(hit_keys)):
                    u, v = hit_keys[i], hit_keys[j]
                    G.add_edge(u, v, **{cfg.edge_type_key: "door"})

    # 4b) adjacent edges: for each adjacent polygon, find regions that overlap enough
    for adj in adjacent_polygons:
        if adj.area <= 0:
            continue
        candidates = tree.query(adj)

        hit_keys = []
        for p in candidates:
            # overlap ratio: region ∩ adj / adj.area
            inter_area = p.intersection(adj).area
            overlap = inter_area / adj.area if adj.area > 0 else 0.0
            if overlap > cfg.adjacent_threshold:
                hit_keys.append(poly_to_key[id(p)])

        if len(hit_keys) >= 2:
            for i in range(len(hit_keys)):
                for j in range(i + 1, len(hit_keys)):
                    u, v = hit_keys[i], hit_keys[j]
                    # if already door edge exists, we keep it (door usually "stronger");
                    # but your original code allows both; here we avoid overwriting.
                    if not G.has_edge(u, v):
                        G.add_edge(u, v, **{cfg.edge_type_key: "adjacent"})

    # 5) remove isolates if needed
    if cfg.drop_isolates:
        isolates = [n for n in G.nodes if G.degree(n) == 0]
        if isolates:
            G.remove_nodes_from(isolates)

    return G


def run_step2_build_basegraphs(
    json_folder: str,
    output_folder: str,
    *,
    cfg: Optional[BaseGraphConfig] = None,
    save_png: bool = True,
    png_subdir: str = "png",
    save_index_csv: bool = True,
    index_csv_name: str = "step2_basegraphs_index.csv",
) -> pd.DataFrame:
    """
    Batch: read all .json -> save .pkl base graphs (with area) to output_folder.

    Returns a dataframe with:
      file, stem, n_nodes, n_edges, pkl_path, saved, reason
    """
    cfg = cfg or BaseGraphConfig()
    os.makedirs(output_folder, exist_ok=True)
    png_dir = os.path.join(output_folder, png_subdir)
    if save_png:
        os.makedirs(png_dir, exist_ok=True)

    rows: List[dict] = []
    def _plot_base_graph_png(G: nx.Graph, json_data: dict, out_path: str):
        image_height = json_data["imageHeight"]
        raw_pos = nx.get_node_attributes(G, "pos")
        pos = {n: (x, image_height - y) for n, (x, y) in raw_pos.items()}

        node_colors = [
            CATEGORY_COLOR_MAP.get(G.nodes[n].get(cfg.function_key, "18"), "#ffffff")
            for n in G.nodes
        ]


        plt.figure(figsize=(10, 8))
        nx.draw_networkx_nodes(
            G, pos,
            node_color=node_colors,
            node_size=300,
            edgecolors="black",
            linewidths=0.8,
        )

        for u, v, d in G.edges(data=True):
            etype = d.get(cfg.edge_type_key, "adjacent")
            style = EDGE_STYLE_MAP.get(etype, {"width": 1.0, "alpha": 0.6})
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                width=style["width"],
                alpha=style["alpha"],
                edge_color="black",
            )

        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()

    for fn in sorted(os.listdir(json_folder)):
        if not fn.endswith(".json"):
            continue

        json_path = os.path.join(json_folder, fn)
        stem = os.path.splitext(fn)[0]

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            rows.append({
                "file": fn,
                "stem": stem,
                "n_nodes": 0,
                "n_edges": 0,
                "pkl_path": "",
                "saved": False,
                "reason": f"json_read_error: {e}",
            })
            continue

        G = build_base_graph_from_labelme_json(data, cfg)

        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        out_pkl = os.path.join(output_folder, stem + ".pkl")

        saved = True
        reason = "ok"
        if cfg.skip_empty_graph and (n_nodes == 0 or n_edges == 0):
            saved = False
            reason = "skip_empty_graph"
        else:
            try:
                with open(out_pkl, "wb") as pf:
                    pickle.dump(G, pf)
            except Exception as e:
                saved = False
                reason = f"pkl_write_error: {e}"
        if save_png and saved:
            try:
                _plot_base_graph_png(
                    G,
                    json_data=data,
                    out_path=os.path.join(png_dir, stem + ".png"),
                )
            except Exception as e:
                # 不影响主流程，只记录
                reason = f"{reason}; png_error: {e}"

        rows.append({
            "file": fn,
            "stem": stem,
            "n_nodes": n_nodes,
            "n_edges": n_edges,
            "pkl_path": out_pkl if saved else "",
            "saved": saved,
            "reason": reason,
        })

    df = pd.DataFrame(rows)

    if save_index_csv:
        csv_path = os.path.join(output_folder, index_csv_name)
        df.to_csv(csv_path, index=False)

    return df
