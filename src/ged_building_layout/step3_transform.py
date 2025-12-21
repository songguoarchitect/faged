from __future__ import annotations

import os
import json
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
from .viz_style import CATEGORY_COLOR_MAP, EDGE_STYLE_MAP

def _plot_graph_png(
    G: nx.Graph,
    *,
    out_path: str,
    pos_key: str = "pos",
    function_key: str = "functiontype",
    edge_type_key: str = "type",
    category_color_map: Optional[Dict[str, str]] = None,
    edge_style_map: Optional[Dict[str, Dict[str, float]]] = None,
    fig_size: Tuple[float, float] = (12, 9),
    pad_ratio: float = 0.06,
    margin_ratio: float = 0.18,  
) -> None:
    import os
    import matplotlib.pyplot as plt

    pos = nx.get_node_attributes(G, pos_key)
    if not pos:
        return

    # image y-down -> plot y-up
    pos = {n: (float(x), -float(y)) for n, (x, y) in pos.items()}

    category_color_map = category_color_map or CATEGORY_COLOR_MAP
    edge_style_map = edge_style_map or EDGE_STYLE_MAP

    def _set_limits_with_padding(pos_dict, pad_ratio=0.06, margin_ratio=0.18):
        xs = [p[0] for p in pos_dict.values()]
        ys = [p[1] for p in pos_dict.values()]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        w = max(xmax - xmin, 1e-6)
        h = max(ymax - ymin, 1e-6)
        padx = w * pad_ratio
        pady = h * pad_ratio

        ax = plt.gca()
        ax.set_xlim(xmin - padx, xmax + padx)
        ax.set_ylim(ymin - pady, ymax + pady)
        ax.set_aspect("equal", adjustable="box")
        ax.margins(margin_ratio)  

    nodelist = list(G.nodes())
    node_colors = [
        category_color_map.get(str(G.nodes[n].get(function_key, "18")), "white")
        for n in nodelist
    ]

    plt.figure(figsize=fig_size)

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=nodelist,
        node_size=200,
        node_color=node_colors,
        edgecolors="black",
        linewidths=0.8,
    )

    type_to_edges: Dict[str, List[Tuple[object, object]]] = {}
    for u, v, d in G.edges(data=True):
        et = str(d.get(edge_type_key, "adjacent"))
        type_to_edges.setdefault(et, []).append((u, v))

    for et, edgelist in type_to_edges.items():
        style = edge_style_map.get(et, {"width": 1.0, "alpha": 0.6})
        nx.draw_networkx_edges(
            G, pos,
            edgelist=edgelist,
            width=style.get("width", 1.0),
            alpha=style.get("alpha", 0.6),
            edge_color="black",
        )

    _set_limits_with_padding(pos, pad_ratio=pad_ratio, margin_ratio=margin_ratio)
    plt.axis("off")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300)  # ✅ no bbox_inches="tight"
    plt.close()


# ---------- corridor removal: relative threshold ----------
def remove_corridor_and_reconnect_relative_threshold(
    graph: nx.Graph,
    pos: Dict[object, Tuple[float, float]],
    *,
    corridor_label: str = "12",
    threshold_ratio: float = 0.2,
    edge_type_key: str = "type",
) -> nx.Graph:
    """
    Remove corridor nodes (label == corridor_label) and reconnect neighbors if
    their Euclidean distance < (spatial_scale * threshold_ratio).

    Edge type inference:
      adjacent+adjacent -> adjacent
      door+door         -> corridor
      door+adjacent     -> door
    """
    if not pos:
        return graph.copy()

    x_vals = [coord[0] for coord in pos.values()]
    y_vals = [coord[1] for coord in pos.values()]
    spatial_scale = max(max(x_vals) - min(x_vals), max(y_vals) - min(y_vals))
    distance_threshold = spatial_scale * float(threshold_ratio)

    new_graph = graph.copy()
    corridor_label = str(corridor_label)
    to_remove = [n for n in graph.nodes if str(n[1]) == corridor_label]

    for corridor_node in to_remove:
        neighbors = list(graph.neighbors(corridor_node))
        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                n1, n2 = neighbors[i], neighbors[j]
                if str(n1[1]) == corridor_label or str(n2[1]) == corridor_label:
                    continue

                edge1_type = graph.get_edge_data(corridor_node, n1, {}).get(edge_type_key)
                edge2_type = graph.get_edge_data(corridor_node, n2, {}).get(edge_type_key)

                if n1 in pos and n2 in pos:
                    dist = Point(pos[n1]).distance(Point(pos[n2]))
                    if dist < distance_threshold and not new_graph.has_edge(n1, n2):
                        if edge1_type == "adjacent" and edge2_type == "adjacent":
                            new_graph.add_edge(n1, n2, **{edge_type_key: "adjacent"})
                        elif edge1_type == "door" and edge2_type == "door":
                            new_graph.add_edge(n1, n2, **{edge_type_key: "corridor"})
                        elif (
                            (edge1_type == "adjacent" and edge2_type == "door") or
                            (edge1_type == "door" and edge2_type == "adjacent")
                        ):
                            new_graph.add_edge(n1, n2, **{edge_type_key: "door"})

    new_graph.remove_nodes_from(to_remove)

    # restore pos attribute
    for node in new_graph.nodes:
        if node in pos:
            new_graph.nodes[node]["pos"] = pos[node]

    return new_graph


# ---------- corridor removal: absolute threshold (meters) ----------
def _convert_rectangle_to_polygon(points) -> Polygon:
    (x1, y1), (x2, y2) = points
    return Polygon([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])


def _get_mode_from_continuous(data: List[float], bin_width: float = 1.0) -> Tuple[float, int]:
    counts, bin_edges = np.histogram(
        data,
        bins=np.arange(min(data), max(data) + bin_width, bin_width),
    )
    max_bin_index = int(np.argmax(counts))
    mode_estimate = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
    return float(mode_estimate), int(counts[max_bin_index])


def estimate_scale_from_doors(
    shapes: list,
    *,
    door_labels: Tuple[str, ...] = ("14",),
    assume_length_m: float = 1.5,
) -> Optional[Tuple[float, float, List[float]]]:
    """
    Estimate pixel->meter scale using door short edge mode.
    Returns (scale_ratio = meters/pixel, mode_px, door_short_edges_px)
    """
    door_lengths: List[float] = []

    for shape in shapes:
        label = str(shape.get("label"))
        if label not in door_labels:
            continue

        pts = shape.get("points", [])
        if len(pts) == 2:
            poly = _convert_rectangle_to_polygon(pts)
        elif len(pts) >= 3:
            poly = Polygon(pts)
        else:
            continue

        if (not poly.is_valid) or poly.is_empty:
            continue

        coords = list(poly.exterior.coords)
        edge_lengths = [
            float(np.linalg.norm(np.array(coords[i]) - np.array(coords[i + 1])))
            for i in range(len(coords) - 1)
        ]
        if len(edge_lengths) >= 2:
            door_lengths.append(min(edge_lengths))

    if not door_lengths:
        return None

    mode_px, _freq = _get_mode_from_continuous(door_lengths, bin_width=1.0)
    scale_ratio = float(assume_length_m) / float(mode_px)  # meters per pixel
    return float(scale_ratio), float(mode_px), door_lengths


def remove_corridor_and_reconnect_absolute_threshold(
    graph: nx.Graph,
    pos: Dict[object, Tuple[float, float]],
    shapes: list,
    *,
    corridor_label: str = "12",
    threshold_value_m: float = 25.0,
    assume_door_length_m: float = 1.5,
    edge_type_key: str = "type",
) -> nx.Graph:
    """
    Like relative version, but distance threshold derived from an estimated pixel->meter scale.
    threshold_value_m is in meters.
    """
    if not pos:
        return graph.copy()

    scale_result = estimate_scale_from_doors(
        shapes, door_labels=("14",), assume_length_m=float(assume_door_length_m)
    )
    if scale_result is None:
        # no door to estimate scale: return original copy to be safe
        return graph.copy()

    scale_ratio_m_per_px, _mode_px, _door_lengths = scale_result
    distance_threshold_px = float(threshold_value_m) / float(scale_ratio_m_per_px)

    new_graph = graph.copy()
    corridor_label = str(corridor_label)
    to_remove = [n for n in graph.nodes if str(n[1]) == corridor_label]

    for corridor_node in to_remove:
        neighbors = list(graph.neighbors(corridor_node))

        for i in range(len(neighbors)):
            for j in range(i + 1, len(neighbors)):
                n1, n2 = neighbors[i], neighbors[j]
                if str(n1[1]) == corridor_label or str(n2[1]) == corridor_label:
                    continue

                edge1_type = graph.get_edge_data(corridor_node, n1, {}).get(edge_type_key)
                edge2_type = graph.get_edge_data(corridor_node, n2, {}).get(edge_type_key)

                if n1 in pos and n2 in pos:
                    dist = Point(pos[n1]).distance(Point(pos[n2]))
                    if dist < distance_threshold_px and not new_graph.has_edge(n1, n2):
                        if edge1_type == "adjacent" and edge2_type == "adjacent":
                            new_graph.add_edge(n1, n2, **{edge_type_key: "adjacent"})
                        elif edge1_type == "door" and edge2_type == "door":
                            new_graph.add_edge(n1, n2, **{edge_type_key: "corridor"})
                        elif (
                            (edge1_type == "adjacent" and edge2_type == "door") or
                            (edge1_type == "door" and edge2_type == "adjacent")
                        ):
                            new_graph.add_edge(n1, n2, **{edge_type_key: "door"})

        if corridor_node in new_graph:
            new_graph.remove_node(corridor_node)

    # restore pos
    for node in new_graph.nodes:
        if node in pos:
            new_graph.nodes[node]["pos"] = pos[node]

    return new_graph


# ---------- selection: choose best transformed graph ----------
@dataclass(frozen=True)
class TransformVariant:
    name: str  # e.g., "relative20" or "absolute25"
    kind: str  # "relative" | "absolute"
    value: float  # ratio (0.2) or meters (25)


@dataclass(frozen=True)
class Step3Config:
    corridor_label: str = "12"

    # variants to generate
    variants: Tuple[TransformVariant, ...] = (
        TransformVariant("relative20", "relative", 0.2),
        TransformVariant("relative25", "relative", 0.25),
        TransformVariant("relative30", "relative", 0.3),
        TransformVariant("absolute20", "absolute", 20.0),
        TransformVariant("absolute25", "absolute", 25.0),
        TransformVariant("absolute30", "absolute", 30.0),
    )

    # ✅ selection rule: enforce an avg-degree range (default 6–8)
    avg_degree_range: Tuple[float, float] = (6.0, 8.0)

    # attribute keys
    pos_key: str = "pos"
    edge_type_key: str = "type"

    # absolute scale
    assume_door_length_m: float = 1.5


def _avg_degree(G: nx.Graph) -> float:
    n = G.number_of_nodes()
    return (2.0 * G.number_of_edges() / n) if n > 0 else 0.0


def choose_best_variant_by_avgdeg_range(
    metrics: List[dict],
    *,
    n_nodes_base: int,
    avgdeg_min: float = 6.0,
    avgdeg_max: float = 8.0,
) -> dict:
    """
    Use avg degree range -> edge range:
      edge_min = N*avgdeg_min/2
      edge_max = N*avgdeg_max/2

    Selection rule:
      1) If any variants with n_edges in [edge_min, edge_max], pick the one with max n_edges
      2) Else pick the one with smallest distance to the range; tie-break by larger n_edges
    """
    edge_min = (n_nodes_base * float(avgdeg_min)) / 2.0
    edge_max = (n_nodes_base * float(avgdeg_max)) / 2.0

    enriched: List[dict] = []
    for m in metrics:
        E = float(m["n_edges"])
        in_range = (edge_min <= E <= edge_max)
        dist = 0.0 if in_range else (edge_min - E) if (E < edge_min) else (E - edge_max)

        mm = dict(m)
        mm["edge_min"] = float(edge_min)
        mm["edge_max"] = float(edge_max)
        mm["in_edge_range"] = bool(in_range)
        mm["edge_dist_to_range"] = float(dist)
        enriched.append(mm)

    in_range_list = [m for m in enriched if m["in_edge_range"]]
    if in_range_list:
        # ✅ within range: choose the one with MORE edges (your requirement)
        best = max(in_range_list, key=lambda x: (x["n_edges"], x["avg_degree"]))
    else:
        # ✅ none in range: choose closest to range; tie-break by larger edges
        best = min(enriched, key=lambda x: (x["edge_dist_to_range"], -x["n_edges"]))

    return dict(best)


def run_step3_transform_from_basegraphs(
    basegraph_folder: str,
    json_folder: str,
    output_root: str,
    selected_output_folder: str,
    *,
    cfg: Optional[Step3Config] = None,
    save_selection_csv: bool = True,
    save_png:bool=True,
    png_subdir:str="png"
) -> pd.DataFrame:
    """
    Input:
      - basegraph_folder: Step2 outputs (basic_witharea/*.pkl)
      - json_folder: raw jsons (needed for absolute scale estimation)

    Output:
      - output_root/<variant.name>/*.pkl
      - selected_output_folder/*.pkl (best per file)
      - selection CSV: records which variant chosen for each file

    Returns selection dataframe.
    """
    cfg = cfg or Step3Config()
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(selected_output_folder, exist_ok=True)

    # index jsons for absolute transform (by stem)
    json_index = {
        os.path.splitext(f)[0]: os.path.join(json_folder, f)
        for f in os.listdir(json_folder)
        if f.endswith(".json")
    }

    rows: List[dict] = []
    pkl_files = sorted([fn for fn in os.listdir(basegraph_folder) if fn.endswith(".pkl")])
    for fn in tqdm(pkl_files, desc="Step3 CaGs transform"):
        if not fn.endswith(".pkl"):
            continue

        stem = os.path.splitext(fn)[0]
        pkl_path = os.path.join(basegraph_folder, fn)

        with open(pkl_path, "rb") as f:
            G: nx.Graph = pickle.load(f)

        pos = nx.get_node_attributes(G, cfg.pos_key)

        # load shapes once (only needed for absolute)
        shapes = None
        json_path = json_index.get(stem)
        if json_path:
            with open(json_path, "r", encoding="utf-8") as jf:
                data = json.load(jf)
                shapes = data.get("shapes", [])

        variant_metrics: List[dict] = []
        variant_graphs: Dict[str, nx.Graph] = {}

        for var in cfg.variants:
            if var.kind == "relative":
                H = remove_corridor_and_reconnect_relative_threshold(
                    G,
                    pos,
                    corridor_label=cfg.corridor_label,
                    threshold_ratio=float(var.value),
                    edge_type_key=cfg.edge_type_key,
                )
            elif var.kind == "absolute":
                if shapes is None:
                    H = G.copy()
                else:
                    H = remove_corridor_and_reconnect_absolute_threshold(
                        G,
                        pos,
                        shapes,
                        corridor_label=cfg.corridor_label,
                        threshold_value_m=float(var.value),
                        assume_door_length_m=float(cfg.assume_door_length_m),
                        edge_type_key=cfg.edge_type_key,
                    )
            else:
                raise ValueError(f"Unknown variant kind: {var.kind}")

            variant_graphs[var.name] = H
            if save_png:
                png_dir = os.path.join(output_root, png_subdir, var.name)
                os.makedirs(png_dir, exist_ok=True)
                _plot_graph_png(
                    H,
                    out_path=os.path.join(png_dir, stem + ".png"),
                    pos_key=cfg.pos_key,
                    function_key="functiontype", 
                    edge_type_key=cfg.edge_type_key,
                )

            m = {
                "file": fn,
                "variant": var.name,
                "kind": var.kind,
                "value": float(var.value),
                "n_nodes": H.number_of_nodes(),
                "n_edges": H.number_of_edges(),
                "avg_degree": _avg_degree(H),
            }
            variant_metrics.append(m)

            # save each variant graph
            out_dir = os.path.join(output_root, var.name)
            os.makedirs(out_dir, exist_ok=True)
            with open(os.path.join(out_dir, fn), "wb") as out_f:
                pickle.dump(H, out_f)

        # choose best by avg-degree-derived edge range
        dmin, dmax = cfg.avg_degree_range
        best = choose_best_variant_by_avgdeg_range(
            variant_metrics,
            n_nodes_base=G.number_of_nodes(),
            avgdeg_min=float(dmin),
            avgdeg_max=float(dmax),
        )

        best_variant = best["variant"]
        best_graph = variant_graphs[best_variant]

        # save selected
        with open(os.path.join(selected_output_folder, fn), "wb") as f:
            pickle.dump(best_graph, f)
        if save_png:
            png_dir2 = os.path.join(selected_output_folder, "png")
            os.makedirs(png_dir2, exist_ok=True)

            stem = os.path.splitext(fn)[0]
            out_png = os.path.join(png_dir2, f"{stem}_best_{best_variant}.png")
            _plot_graph_png(
                    best_graph,
                    out_path=out_png,
                    pos_key=cfg.pos_key,
                    function_key="functiontype", 
                    edge_type_key=cfg.edge_type_key,
                )

        rows.append({
            "file": fn,
            "best_variant": best_variant,
            "best_kind": best["kind"],
            "best_value": best["value"],
            "best_n_nodes": best["n_nodes"],
            "best_n_edges": best["n_edges"],
            "best_avg_degree": best["avg_degree"],

            # explainability
            "edge_min": best["edge_min"],
            "edge_max": best["edge_max"],
            "in_edge_range": best["in_edge_range"],
            "edge_dist_to_range": best["edge_dist_to_range"],
        })

    df = pd.DataFrame(rows)
    if save_selection_csv:
        df.to_csv(os.path.join(selected_output_folder, "step3_selected_variant.csv"), index=False)

    return df
