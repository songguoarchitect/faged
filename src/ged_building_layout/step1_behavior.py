from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from .viz_style import CATEGORY_COLOR_MAP, EDGE_STYLE_MAP


# ---------------- Config ----------------
@dataclass(frozen=True)
class BehaviorGraphConfig:
    # how to parse group name from csv filename
    group_name_rule: str = "prefix_before_underscore"  # or "full_stem"
    csv_suffix: str = ".csv"

    # node/edge attribute keys
    node_category_key: str = "category"
    edge_weight_key: str = "weight"
    edge_type_key: str = "type"

    # thresholding on normalized weights
    # keep edges strictly greater than th1 (same as your original code: w<=th1 -> drop)
    quantiles: Tuple[float, float, float] = (0.25, 0.50, 0.75)

    # mapping weight bins -> edge type
    # (th1, th2] => corridor, (th2, th3] => door, > th3 => adjacent
    low_type: str = "corridor"
    mid_type: str = "door"
    high_type: str = "adjacent"

    # behavior
    drop_isolates: bool = True
    skip_if_edges_less_than: int = 1  # if <1 edge after filtering, skip saving

    # visualization
    save_png: bool = True
    show_plot: bool = False
    fig_size: Tuple[int, int] = (8, 6)
    spring_seed: int = 42


# ---------------- Helpers ----------------
def _parse_group_name(csv_filename: str, rule: str) -> str:
    stem = os.path.splitext(os.path.basename(csv_filename))[0]
    if rule == "full_stem":
        return stem
    # default: prefix_before_underscore
    return stem.split("_")[0]


def list_groups_in_behavior_csv(csv_dir: str, *, rule: str = "prefix_before_underscore") -> List[str]:
    """Scan csv_dir and return sorted unique group names inferred from filenames."""
    groups: List[str] = []
    for fn in sorted(os.listdir(csv_dir)):
        if fn.endswith(".csv"):
            groups.append(_parse_group_name(fn, rule))
    return sorted(set(groups))


def _compute_thresholds_minmax(weights: List[float], q: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Use min + (max-min)*q as your original code (not statistical quantile).
    """
    wmin, wmax = min(weights), max(weights)
    th1 = wmin + (wmax - wmin) * q[0]
    th2 = wmin + (wmax - wmin) * q[1]
    th3 = wmin + (wmax - wmin) * q[2]
    return float(th1), float(th2), float(th3)


def build_behavior_graph_from_matrix(
    df: pd.DataFrame,
    *,
    node_categories: Dict[str, List[str]],
    people_num: float,
    cfg: Optional[BehaviorGraphConfig] = None,
) -> nx.Graph:
    """
    Input df: adjacency/behavior matrix (index=nodes, columns=nodes), values >=0.
    Output graph:
      - node attr: category (list[str])
      - edge attrs: weight (normalized), type (corridor/door/adjacent)
    """
    cfg = cfg or BehaviorGraphConfig()
    people_num = float(people_num)
    if people_num <= 0:
        raise ValueError(f"people_num must be > 0, got {people_num}")

    # Keep node labels as they appear in df.index (often 'a','b',...)
    nodes = list(df.index)

    # Build weighted graph (normalized)
    G = nx.Graph()
    for n in nodes:
        cats = node_categories.get(str(n), "18")
        G.add_node(n, **{cfg.node_category_key: [cats]})

    for a_idx in range(len(nodes)):
        for b_idx in range(a_idx + 1, len(nodes)):
            i, j = nodes[a_idx], nodes[b_idx]
            val = df.at[i, j]
            if pd.isna(val) or float(val) <= 0:
                continue
            w = float(val) / people_num
            G.add_edge(i, j, **{cfg.edge_weight_key: w})

    weights = [d[cfg.edge_weight_key] for _, _, d in G.edges(data=True)]
    if len(weights) < 2:
        # If no/too few edges, return graph after isolate drop (consistent behavior)
        H = G.copy()
        if cfg.drop_isolates:
            H.remove_nodes_from(list(nx.isolates(H)))
        return H

    th1, th2, th3 = _compute_thresholds_minmax(weights, cfg.quantiles)

    H = nx.Graph()
    for n, attrs in G.nodes(data=True):
        H.add_node(n, **attrs)

    for u, v, d in G.edges(data=True):
        w = float(d[cfg.edge_weight_key])
        if w <= th1:
            continue
        elif w <= th2:
            etype = cfg.low_type
        elif w <= th3:
            etype = cfg.mid_type
        else:
            etype = cfg.high_type
        H.add_edge(u, v, **{cfg.edge_weight_key: w, cfg.edge_type_key: etype})

    if cfg.drop_isolates:
        H.remove_nodes_from(list(nx.isolates(H)))

    return H


def _plot_graph(
    G: nx.Graph,
    *,
    category_color_map: Dict[str, str],
    edge_styles: Dict[str, Dict[str, float]],
    cfg: BehaviorGraphConfig,
    title: str,
    out_png: Optional[str] = None,
) -> None:


    plt.figure(figsize=cfg.fig_size)
    pos = nx.spring_layout(G, seed=cfg.spring_seed)

    # node color: first category
    node_colors = []
    for n in G.nodes():
        cats = G.nodes[n].get(cfg.node_category_key, ["18"])
        main_cat = str(cats[0]) if isinstance(cats, list) and len(cats) > 0 else "18"
        node_colors.append(category_color_map.get(main_cat, "white"))

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, edgecolors="black")

    for u, v, d in G.edges(data=True):
        et = d.get(cfg.edge_type_key, cfg.high_type)
        style = edge_styles.get(et, {"width": 1.0, "alpha": 0.6})
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], width=style.get("width", 1.0),
            alpha=style.get("alpha", 0.6), edge_color="black"
        )

    nx.draw_networkx_labels(G, pos, font_size=10)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()

    if out_png:
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        plt.savefig(out_png, dpi=300, bbox_inches="tight")

    if cfg.show_plot:
        plt.show()
    else:
        plt.close()


# ---------------- Batch runner ----------------
def run_step1_build_behavior_graphs(
    csv_dir: str,
    output_dir: str,
    *,
    node_categories: Dict[str, List[str]],
    people_counts: Dict[str, float],  
    category_color_map: Optional[Dict[str, str]] = None,
    edge_styles: Optional[Dict[str, Dict[str, float]]] = None,
    cfg: Optional[BehaviorGraphConfig] = None,
    save_index_csv: bool = True,
    index_csv_name: str = "step1_behavior_index.csv",
) -> pd.DataFrame:
    """
    Batch:
      - read *.csv behavior matrices
      - build graphs (normalize by people_counts[group])  
      - save pkl to output_dir/Group_<group>.pkl
      - optional save png

    Returns: dataframe summary
    """
    cfg = cfg or BehaviorGraphConfig()
    if not people_counts:
        raise ValueError("people_counts must be provided by user (e.g. {'A':12,'B':8,...}).")

    category_color_map = category_color_map or CATEGORY_COLOR_MAP
    edge_styles = edge_styles or EDGE_STYLE_MAP


    # 1) validate people_counts covers all groups
    groups_needed = list_groups_in_behavior_csv(csv_dir, rule=cfg.group_name_rule)
    missing = [g for g in groups_needed if g not in people_counts]
    if missing:
        raise KeyError(
            "people_counts is missing these groups inferred from CSV filenames: "
            f"{missing}. Please add them before running Step1."
        )

    # 2) run
    os.makedirs(output_dir, exist_ok=True)

    rows: List[dict] = []
    for fn in sorted(os.listdir(csv_dir)):
        if not fn.endswith(cfg.csv_suffix):
            continue

        group = _parse_group_name(fn, cfg.group_name_rule)
        people_num = float(people_counts[group])
        if people_num <= 0:
            raise ValueError(f"people_counts[{group}] must be > 0, got {people_num}. File={fn}")

        csv_path = os.path.join(csv_dir, fn)

        try:
            df = pd.read_csv(csv_path, index_col=0)
        except Exception as e:
            rows.append({
                "file": fn, "group": group, "people_num": people_num,
                "n_nodes": 0, "n_edges": 0,
                "saved_pkl": "", "saved_png": "",
                "reason": f"csv_read_error: {e}",
            })
            continue

        G = build_behavior_graph_from_matrix(
            df,
            node_categories=node_categories,
            people_num=people_num,
            cfg=cfg,
        )

        if G.number_of_edges() < cfg.skip_if_edges_less_than:
            rows.append({
                "file": fn, "group": group, "people_num": people_num,
                "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
                "saved_pkl": "", "saved_png": "",
                "reason": "skip_too_few_edges",
            })
            continue

        out_pkl = os.path.join(output_dir, f"Group_{group}.pkl")
        with open(out_pkl, "wb") as f:
            pickle.dump(G, f)

        out_png = ""
        if cfg.save_png:
            out_png = os.path.join(output_dir, f"Group_{group}.png")
            _plot_graph(
                G,
                category_color_map=category_color_map,
                edge_styles=edge_styles,
                cfg=cfg,
                title=f"Group {group} (Filtered Graph)",
                out_png=out_png,
            )

        rows.append({
            "file": fn, "group": group, "people_num": people_num,
            "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
            "saved_pkl": out_pkl, "saved_png": out_png,
            "reason": "ok",
        })

    df_out = pd.DataFrame(rows)
    if save_index_csv:
        df_out.to_csv(os.path.join(output_dir, index_csv_name), index=False)

    return df_out
