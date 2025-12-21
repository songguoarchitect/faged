from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import networkx as nx
import pandas as pd
from tqdm import tqdm
from collections import defaultdict, Counter
from .viz_style import CATEGORY_COLOR_MAP, EDGE_STYLE_MAP


@dataclass(frozen=True)
class InfomapConfig:
    input_folder: str
    graph_output_root: str          
    community_img_output_root: str  

    # Infomap
    markov_times: Tuple[float, ...] = (0.7,)
    infomap_args_template: str = "--two-level --silent --markov-time {t}"

    type_weight_map: Optional[Dict[str, float]] = None
    edge_type_priority: Optional[Dict[str, int]] = None
    function_priority_order: Optional[List[str]] = None

    main_function_mode: str = "count"  # "count" | "max_area"

    pos_key: str = "pos"
    function_key: str = "functiontype"
    area_key: str = "area"
    edge_type_key: str = "type"

    save_png: bool = True
    category_color_map: Optional[Dict[str, str]] = None


def _get_main_function_by_count(function_list: List[str], priority_order: List[str]) -> str:
    count = Counter(function_list)
    most_common = count.most_common()
    if not most_common:
        return "unknown"
    max_count = most_common[0][1]
    tied = [func for func, cnt in most_common if cnt == max_count]
    if len(tied) == 1:
        return tied[0]
    for p in priority_order:
        if p in tied:
            return p
    return tied[0]


def _get_main_function_by_max_area(
    G: nx.Graph,
    partition: Dict[object, int],
    *,
    function_key: str,
    area_key: str,
    priority_order: List[str],
    default: str = "unknown",
) -> Dict[int, str]:
    """
    return {community_id: main_function}
    """
    comm_to_nodes = defaultdict(list)
    for n, cid in partition.items():
        comm_to_nodes[cid].append(n)

    community_main: Dict[int, str] = {}
    for cid, nodes in comm_to_nodes.items():
        areas = [(n, float(G.nodes[n].get(area_key, 0.0))) for n in nodes]
        max_area = max(a for _, a in areas) if areas else 0.0
        candidates = [n for n, a in areas if a == max_area]
        cand_funcs = [str(G.nodes[n].get(function_key, default)) for n in candidates]

        if len(candidates) == 1:
            community_main[cid] = cand_funcs[0] if cand_funcs else default
            continue

        for p in priority_order:
            if p in cand_funcs:
                community_main[cid] = p
                break
        else:
            community_main[cid] = cand_funcs[0] if cand_funcs else default

    return community_main


def _compute_partition_infomap(
    G_work: nx.Graph,
    *,
    infomap_args: str,
    type_weight_map: Dict[str, float],
) -> Dict[object, int]:
    """
    return {node: community_id}
    """
    from infomap import Infomap  

    # node id mapping
    node_to_id = {node: idx for idx, node in enumerate(G_work.nodes())}
    id_to_node = {idx: node for node, idx in node_to_id.items()}

    im = Infomap(infomap_args)
    for u, v, data in G_work.edges(data=True):
        edge_type = data.get("type", "door")
        weight = type_weight_map.get(edge_type, 1.0)
        im.add_link(node_to_id[u], node_to_id[v], weight)
    im.run()

    partition = {id_to_node[n.node_id]: n.module_id for n in im.nodes}
    return partition


def _build_simplified_graph(
    G_work: nx.Graph,
    partition: Dict[object, int],
    *,
    main_function_mode: str,
    function_key: str,
    area_key: str,
    function_priority_order: List[str],
    edge_type_key: str,
    edge_type_priority: Dict[str, int],
) -> nx.Graph:

    if main_function_mode == "max_area":
        community_main_type = _get_main_function_by_max_area(
            G_work, partition,
            function_key=function_key,
            area_key=area_key,
            priority_order=function_priority_order,
            default="unknown",
        )
    else:
        # count
        community_function_map = defaultdict(list)
        for node, comm_id in partition.items():
            func = str(G_work.nodes[node].get(function_key, "unknown"))
            community_function_map[comm_id].append(func)
        community_main_type = {
            cid: _get_main_function_by_count(funcs, function_priority_order)
            for cid, funcs in community_function_map.items()
        }

    # edge voting
    inter_comm_edge_type_counter = defaultdict(Counter)
    for u, v, data in G_work.edges(data=True):
        cu, cv = partition[u], partition[v]
        if cu == cv:
            continue
        key = (cu, cv) if cu < cv else (cv, cu)
        etype = data.get(edge_type_key, "adjacent")
        inter_comm_edge_type_counter[key][etype] += 1

    G_simplified = nx.Graph()
    for comm_id, main_func in community_main_type.items():
        G_simplified.add_node(comm_id, main_function=main_func)

    for (cu, cv), counter in inter_comm_edge_type_counter.items():
        voted_type = max(
            counter.items(),
            key=lambda kv: (kv[1], edge_type_priority.get(kv[0], 0))
        )[0]
        G_simplified.add_edge(cu, cv, **{edge_type_key: voted_type})

    return G_simplified


def _compute_simplified_positions(
    G_simplified: nx.Graph,
    partition: Dict[object, int],
    pos_original: Dict[object, Tuple[float, float]],
) -> Dict[int, Tuple[float, float]]:
    """
    centroid as pos

    """
    comm_positions: Dict[int, Tuple[float, float]] = {}

    for comm_id in G_simplified.nodes():
        nodes_in_comm = [n for n, c in partition.items() if c == comm_id]
        xs, ys = [], []
        for n in nodes_in_comm:
            if n in pos_original:
                xs.append(pos_original[n][0])
                ys.append(pos_original[n][1])
        if xs and ys:
            comm_positions[comm_id] = (sum(xs)/len(xs), sum(ys)/len(ys))

    if len(comm_positions) < G_simplified.number_of_nodes():
        fallback = nx.spring_layout(G_simplified, seed=42)
        for nid in G_simplified.nodes():
            if nid not in comm_positions:
                comm_positions[nid] = (float(fallback[nid][0]), float(fallback[nid][1]))

    return comm_positions


def run_step4_infomap(cfg: InfomapConfig) -> List[str]:

    # defaults
    type_weight_map = cfg.type_weight_map or {"corridor": 0.5, "door": 1.0, "adjacent": 1.5}
    edge_type_priority = cfg.edge_type_priority or {"adjacent": 3, "door": 2, "corridor": 1}
    function_priority_order = cfg.function_priority_order or ['3', '5', '6', '4', '2']
    category_color_map = cfg.category_color_map or CATEGORY_COLOR_MAP
    saved_pkls: List[str] = []
    markov_partitions: Dict[float, List[pd.DataFrame]] = {t: [] for t in cfg.markov_times}
    cag_pkl_files = sorted([fn for fn in os.listdir(cfg.input_folder) if fn.endswith(".pkl")])
    for file in tqdm(cag_pkl_files, desc="Step4 Prototype extraction"):
        if not file.endswith(".pkl"):
            continue

        input_path = os.path.join(cfg.input_folder, file)
        filename = os.path.splitext(file)[0]

        with open(input_path, "rb") as f:
            G: nx.Graph = pickle.load(f)

        # drop isolates
        isolates = list(nx.isolates(G))
        G_work = G.copy()
        if isolates:
            G_work.remove_nodes_from(isolates)


        if G_work.number_of_nodes() == 0 or G_work.number_of_edges() == 0:
            continue

        # pos_original (fixed for all markov times)
        pos_all = nx.get_node_attributes(G, cfg.pos_key)
        if not pos_all:
            raise ValueError(f"{filename}: no node attribute '{cfg.pos_key}' found in pkl")
        pos_original = {n: pos_all[n] for n in G_work.nodes() if n in pos_all}
        pos_original = {n: (float(x), -float(y)) for n, (x, y) in pos_original.items()}


        # per markov time
        for t in cfg.markov_times:
            markov_str = f"{t}".replace(".", "_")

            graph_out = os.path.join(cfg.graph_output_root, f"markov_{markov_str}")
            comm_img_out = os.path.join(cfg.community_img_output_root, f"markov_{markov_str}")
            os.makedirs(graph_out, exist_ok=True)
            os.makedirs(comm_img_out, exist_ok=True)

            infomap_args = cfg.infomap_args_template.format(t=t)

            partition = _compute_partition_infomap(
                G_work,
                infomap_args=infomap_args,
                type_weight_map=type_weight_map,
            )


            G_simplified = _build_simplified_graph(
                G_work, partition,
                main_function_mode=cfg.main_function_mode,
                function_key=cfg.function_key,
                area_key=cfg.area_key,
                function_priority_order=function_priority_order,
                edge_type_key=cfg.edge_type_key,
                edge_type_priority=edge_type_priority,
            )
            # keep only LCC on simplified graph (after community detection)
            if G_simplified.number_of_nodes() > 0 and not nx.is_connected(G_simplified):
                lcc_nodes_s = max(nx.connected_components(G_simplified), key=len)
                G_simplified = G_simplified.subgraph(lcc_nodes_s).copy()

                # keep partition consistent with simplified LCC
                kept_comm_ids = set(G_simplified.nodes())
                partition = {n: cid for n, cid in partition.items() if cid in kept_comm_ids}
                
            pos_simple = _compute_simplified_positions(G_simplified, partition, pos_original)

            simplified_path = os.path.join(graph_out, file)
            with open(simplified_path, "wb") as f:
                pickle.dump(G_simplified, f)
            saved_pkls.append(simplified_path)

            df_partition = pd.DataFrame(
                [{"filename": filename, "node": node, "community": cid} for node, cid in partition.items()]
            )
            markov_partitions[t].append(df_partition)

            if cfg.save_png:
                _save_plots(
                    filename=filename,
                    G_work=G_work,
                    G_simplified=G_simplified,
                    partition=partition,
                    pos_original=pos_original,
                    pos_simple=pos_simple,
                    graph_out=graph_out,
                    comm_img_out=comm_img_out,
                    category_color_map=category_color_map,
                    edge_type_key=cfg.edge_type_key,
                )

    for t, dfs in markov_partitions.items():
        if not dfs:
            continue
        markov_str = f"{t}".replace(".", "_")
        comm_img_out = os.path.join(cfg.community_img_output_root, f"markov_{markov_str}")
        os.makedirs(comm_img_out, exist_ok=True)
        df_all = pd.concat(dfs, ignore_index=True)
        out_csv = os.path.join(comm_img_out, f"all_graph_communities_infomap_markov_{markov_str}.csv")
        df_all.to_csv(out_csv, index=False)

    return saved_pkls

def _save_plots(
    *,
    filename: str,
    G_work: nx.Graph,
    G_simplified: nx.Graph,
    partition: Dict[object, int],
    pos_original: Dict[object, Tuple[float, float]],
    pos_simple: Dict[int, Tuple[float, float]],
    graph_out: str,
    comm_img_out: str,
    category_color_map: Dict[str, str],
    edge_type_key: str,
):
    import matplotlib.pyplot as plt
    import os

    def _set_limits_with_padding(pos, *, pad_ratio=0.06, node_size=800, lw=8.0):
        if not pos:
            return
        ax = plt.gca()

        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)

        w = max(xmax - xmin, 1e-6)
        h = max(ymax - ymin, 1e-6)

        # 1) bbox padding in data units
        padx = w * pad_ratio
        pady = h * pad_ratio

        # 2) convert display padding (node radius + line width) -> data units
        # node_size is points^2, radius ~ sqrt(node_size)/2 points
        r_pt = (node_size ** 0.5) / 2.0
        extra_pt = r_pt + lw + 6  # +6pt safety
        extra_px = extra_pt * plt.gcf().dpi / 72.0

        # transform: pixels -> data units (x and y may differ)
        p0 = ax.transData.inverted().transform((0, 0))
        px = ax.transData.inverted().transform((extra_px, 0))
        py = ax.transData.inverted().transform((0, extra_px))
        extra_dx = abs(px[0] - p0[0])
        extra_dy = abs(py[1] - p0[1])

        ax.set_xlim(xmin - padx - extra_dx, xmax + padx + extra_dx)
        ax.set_ylim(ymin - pady - extra_dy, ymax + pady + extra_dy)
        ax.set_aspect("equal", adjustable="box")


    FIG_SIZE = (12, 9) 

    # =========================
    # simplified graph plot
    # =========================
    plt.figure(figsize=FIG_SIZE)
    node_colors = [
        category_color_map.get(str(G_simplified.nodes[n].get("main_function", "18")), "#ffffff")
        for n in G_simplified.nodes()
    ]
    nx.draw_networkx_nodes(
        G_simplified, pos_simple,
        node_color=node_colors,
        node_size=800,
        linewidths=1,
        edgecolors="black",
    )

    for u, v, data in G_simplified.edges(data=True):
        etype = data.get(edge_type_key, "adjacent")
        style = {"width": 8.0, "alpha": 0.8} if etype == "adjacent" else {"width": 2.0, "alpha": 0.5}
        nx.draw_networkx_edges(
            G_simplified, pos_simple,
            edgelist=[(u, v)],
            width=style["width"],
            alpha=style["alpha"],
            edge_color="black",
        )

    _set_limits_with_padding(pos_simple, pad_ratio=0.06, node_size=800, lw=8.0)
    plt.title(f"{filename}")
    plt.axis("off")
    os.makedirs(graph_out, exist_ok=True)
    plt.savefig(os.path.join(graph_out, filename + ".png"), dpi=300) 
    plt.close()

    # =========================
    # original graph plot (community coloring)
    # =========================
    plt.figure(figsize=FIG_SIZE)
    node_colors_orig = [partition.get(n, -1) for n in G_work.nodes()]  # safe if filtered
    nx.draw_networkx_nodes(
        G_work, pos_original,
        node_color=node_colors_orig,
        node_size=800,
        linewidths=1,
        edgecolors="black",
    )

    for u, v, data in G_work.edges(data=True):
        etype = data.get(edge_type_key, "adjacent")
        style = {"width": 8.0, "alpha": 0.8} if etype == "adjacent" else {"width": 2.0, "alpha": 0.5}
        nx.draw_networkx_edges(
            G_work, pos_original,
            edgelist=[(u, v)],
            width=style["width"],
            alpha=style["alpha"],
            edge_color="black",
        )

    _set_limits_with_padding(pos_original, pad_ratio=0.06, node_size=800, lw=8.0)
    plt.title(f"{filename}_Original Graph (isolates dropped) Infomap Community Coloring")
    plt.axis("off")
    os.makedirs(comm_img_out, exist_ok=True)
    plt.savefig(os.path.join(comm_img_out, filename + ".png"), dpi=300) 
    plt.close()
