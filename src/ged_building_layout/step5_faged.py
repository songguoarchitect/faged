from __future__ import annotations

import os
import re
import pandas as pd
import pickle
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd
import networkx as nx
from tqdm import tqdm

try:
    from networkx.algorithms.similarity import optimal_edit_paths
except Exception:
    optimal_edit_paths = None


# ---------- I/O: load graphs ----------
def load_graphs_from_folder(
    folder: str,
    *,
    is_target: bool,
    target_category_key: str = "category",
    reference_category_key: str = "main_function",
    default_category: str = "unknown",
) -> Dict[str, nx.Graph]:
    """
    Load .pkl graphs from folder and normalize node attribute to G.nodes[n]["category"].

    Target graphs:
      - read from target_category_key (default "category")
      - if it's a list, take the first element
    Reference graphs:
      - read from reference_category_key (default "main_function")

    Returns: {filename.pkl: nx.Graph}
    """
    graphs: Dict[str, nx.Graph] = {}
    for file in sorted(os.listdir(folder)):
        if not file.endswith(".pkl"):
            continue
        path = os.path.join(folder, file)
        with open(path, "rb") as f:
            G: nx.Graph = pickle.load(f)

        if is_target:
            for n, attr in G.nodes(data=True):
                v = attr.get(target_category_key, default_category)
                if isinstance(v, list):
                    v = v[0] if len(v) > 0 else default_category
                G.nodes[n]["category"] = str(v)
        else:
            for n, attr in G.nodes(data=True):
                v = attr.get(reference_category_key, default_category)
                G.nodes[n]["category"] = str(v)

        graphs[file] = G

    return graphs


# ---------- GED / nGED ----------
@dataclass(frozen=True)
class GedConfig:
    timeout: int = 30
    normalize: bool = False
    output_suffix: str = "GED"  # informational only


def compute_ged_tables(
    target_graphs: Dict[str, nx.Graph],
    reference_graphs: Dict[str, nx.Graph],
    cfg: GedConfig,
    *,
    show_progress: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    For each target graph, compute GED (or nGED) to all reference graphs.
    Returns: dict[target_filename] -> DataFrame([Reference_Graph, Edit_Distance])
    """
    out: Dict[str, pd.DataFrame] = {}

    for t_name, t_graph in target_graphs.items():
        distances: List[Tuple[str, float]] = []

        iterator = reference_graphs.items()
        if show_progress:
            iterator = tqdm(iterator, desc=f"GED with {t_name}", leave=False)

        for r_name, r_graph in iterator:
            try:
                raw = nx.graph_edit_distance(t_graph, r_graph, timeout=cfg.timeout)
                n1, n2 = t_graph.number_of_nodes(), r_graph.number_of_nodes()

                if raw is None:
                    dist = float("inf")
                else:
                    if cfg.normalize and n1 > 0 and n2 > 0:
                        dist = float(raw) / (n1 * n2)
                    else:
                        dist = float(raw)

            except Exception:
                dist = float("inf")

            distances.append((r_name, dist))

        distances.sort(key=lambda x: x[1])
        out[t_name] = pd.DataFrame(distances, columns=["Reference_Graph", "Edit_Distance"])

    return out


def save_tables(
    tables: Dict[str, pd.DataFrame],
    output_dir: str,
    *,
    filename_suffix: str,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    paths: List[str] = []
    for t_name, df in tables.items():
        base = os.path.splitext(t_name)[0]
        p = os.path.join(output_dir, f"{base}_{filename_suffix}.csv")
        df.to_csv(p, index=False)
        paths.append(p)
    return paths


# ---------- FaGED (optimal edit paths) ----------
@dataclass(frozen=True)
class FagedConfig:
    normalize: bool = True
    output_suffix: str = "FaGED"
    save_edit_paths_txt: bool = True


def _node_match(a1: dict, a2: dict) -> bool:
    return a1.get("category") == a2.get("category")


def _edge_match(e1: dict, e2: dict) -> bool:
    return ("type" in e1 and "type" in e2) and (e1["type"] == e2["type"])


def compute_faged_tables(
    target_graphs: Dict[str, nx.Graph],
    reference_graphs: Dict[str, nx.Graph],
    cfg: FagedConfig,
    *,
    show_progress: bool = True,
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict[str, object]]]:
    """
    Compute optimal_edit_paths cost as FaGED.

    Returns:
      - tables: dict[target] -> DataFrame([Reference_Graph, Edit_Distance, Edit_Distance_Norm])
      - records: dict[target] -> {"paths": List[(ref, cost, edit_path)]}
    """
    if optimal_edit_paths is None:
        raise RuntimeError(
            "networkx.algorithms.similarity.optimal_edit_paths is not available in your networkx version."
        )

    tables: Dict[str, pd.DataFrame] = {}
    records: Dict[str, Dict[str, object]] = {}

    for t_name, t_graph in target_graphs.items():
        distances: List[Tuple[str, float, float]] = []
        paths_record: List[Tuple[str, float, object]] = []

        iterator = reference_graphs.items()
        if show_progress:
            iterator = tqdm(iterator, desc=f"FaGED with {t_name}", leave=False)

        for r_name, r_graph in iterator:
            try:
                itr = optimal_edit_paths(
                    t_graph,
                    r_graph,
                    node_match=_node_match,
                    edge_match=_edge_match,
                )
                if isinstance(itr, tuple):
                    edit_path, cost = itr
                else:
                    edit_path, cost = next(itr)

                n1, n2 = t_graph.number_of_nodes(), r_graph.number_of_nodes()
                norm = (float(cost) / (n1 * n2)) if (cfg.normalize and n1 > 0 and n2 > 0) else float(cost)

                distances.append((r_name, float(cost), float(norm)))
                paths_record.append((r_name, float(cost), edit_path))

            #except Exception:
                #distances.append((r_name, float("inf"), float("inf")))
                #paths_record.append((r_name, float("inf"), []))
            except Exception as e:
                print(f"[GED ERROR] {t_name} vs {r_name}: {type(e).__name__}: {e}")
                dist = float("inf")

        distances.sort(key=lambda x: x[1])
        tables[t_name] = pd.DataFrame(
            distances,
            columns=["Reference_Graph", "Edit_Distance_Unnorm", "Edit_Distance"],
        )
        records[t_name] = {"paths": paths_record}

    return tables, records


def save_faged_outputs(
    tables: Dict[str, pd.DataFrame],
    records: Dict[str, Dict[str, object]],
    output_dir: str,
    *,
    csv_suffix: str = "FaGED",
    save_edit_paths_txt: bool = True,
) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    saved: List[str] = []

    for t_name, df in tables.items():
        base = os.path.splitext(t_name)[0]
        csv_path = os.path.join(output_dir, f"{base}_{csv_suffix}.csv")
        df.to_csv(csv_path, index=False)
        saved.append(csv_path)

        if save_edit_paths_txt:
            txt_path = os.path.join(output_dir, f"{base}_edit_paths.txt")
            with open(txt_path, "w", encoding="utf-8") as f:
                for r_name, cost, edit_path in records[t_name]["paths"]:
                    f.write(f"\n--- {r_name} ---\n")
                    f.write(f"Distance: {cost}\n")
                    f.write("Edit Path:\n")
                    f.write(f"{edit_path}\n")
            saved.append(txt_path)

    return saved


# ---------- Merge CSV + rank ----------
def merge_rank_tables(
    csv_folder: str,
    *,
    distance_col: str = "Edit_Distance",
    key_col: str = "Reference_Graph",
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    all_df = pd.DataFrame()

    for csv_file in sorted(os.listdir(csv_folder)):
        if not csv_file.endswith(".csv"):
            continue
        csv_path = os.path.join(csv_folder, csv_file)
        df = pd.read_csv(csv_path)

        if key_col not in df.columns or distance_col not in df.columns:
            continue

        df["rank"] = df[distance_col].rank(method="min", ascending=True).astype(int)
        df_sorted = df.sort_values(by=distance_col).reset_index(drop=True)

        algo_name = os.path.splitext(csv_file)[0]
        rank_col = f"{algo_name}_rank"
        score_col = f"{algo_name}_score"

        piece = df_sorted[[key_col, "rank", distance_col]].rename(
            columns={"rank": rank_col, distance_col: score_col}
        )

        all_df = piece if all_df.empty else all_df.merge(piece, on=key_col, how="outer")

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        all_df.to_csv(output_csv, index=False)

    return all_df


# ---------- Single-folder runner ----------
def run_step5(
    target_folder: str,
    reference_folder: str,
    output_dir: str,
    *,
    do_ged: bool = True,
    do_nged: bool = True,
    do_faged: bool = True,
    timeout: int = 300,
    # IMPORTANT: adapt to step4 outputs
    target_category_key: str = "category",
    reference_category_key: str = "main_function",
    # optional post-merge
    comparison_csv_folder: Optional[str] = None,
    summary_output_csv: Optional[str] = None,
) -> bool:
    """
    Run GED/nGED/FaGED for ONE target folder (e.g., one markov_* folder).
    """
    target_graphs = load_graphs_from_folder(
        target_folder, is_target=True, target_category_key=target_category_key
    )
    reference_graphs = load_graphs_from_folder(
        reference_folder, is_target=False, reference_category_key=reference_category_key
    )
    if do_ged:
        ged_tables = compute_ged_tables(
            target_graphs, reference_graphs, GedConfig(timeout=timeout, normalize=False)
        )
        save_tables(ged_tables, output_dir, filename_suffix="GED")

    if do_nged:
        nged_tables = compute_ged_tables(
            target_graphs, reference_graphs, GedConfig(timeout=timeout, normalize=True)
        )
        save_tables(nged_tables, output_dir, filename_suffix="nGED")

    if do_faged:
        faged_tables, faged_records = compute_faged_tables(
            target_graphs, reference_graphs, FagedConfig(normalize=True)
        )
        save_faged_outputs(
            faged_tables, faged_records, output_dir,
            csv_suffix="FaGED",
            save_edit_paths_txt=True
        )

    if comparison_csv_folder and summary_output_csv:
        merge_rank_tables(comparison_csv_folder, output_csv=summary_output_csv)

    return True


# ---------- Batch runner for Step4 markov_* folders ----------
@dataclass(frozen=True)
class Step5BatchConfig:
    step4_graph_output_root: str

    target_folder: str
    step5_output_root: str

    do_ged: bool = True
    do_nged: bool = True
    do_faged: bool = True
    timeout: int = 30

    markov_folders: Optional[List[str]] = None

    target_category_key: str = "category"
    reference_category_key: str = "main_function"

    write_summary_csv: bool = True
    summary_filename: str = "step5_batch_summary.csv"



def _list_markov_subfolders(root: str) -> List[str]:
    if not os.path.isdir(root):
        raise FileNotFoundError(f"step4_graph_output_root not found: {root}")
    subs: List[str] = []
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name)
        if os.path.isdir(p) and name.startswith("markov_"):
            subs.append(name)
    return subs


def run_step5_batch_from_markov_folders(cfg: Step5BatchConfig) -> pd.DataFrame:
    """
    For each markov_* folder under cfg.step4_graph_output_root, run run_step5(...)
    and save outputs to cfg.step5_output_root/<markov_name>/.

    Returns a summary DataFrame (also saved as CSV if cfg.write_summary_csv=True).
    """    
    if not hasattr(cfg, "target_folder"):
        raise ValueError("Step5BatchConfig must define target_folder")

    target_folder = cfg.target_folder

    markov_names = cfg.markov_folders or _list_markov_subfolders(cfg.step4_graph_output_root)
    os.makedirs(cfg.step5_output_root, exist_ok=True)

    rows: List[dict] = []
    for markov_name in markov_names:
        reference_folder = os.path.join(cfg.step4_graph_output_root, markov_name)  
        out_dir = os.path.join(cfg.step5_output_root, markov_name)
        os.makedirs(out_dir, exist_ok=True)

        ok = run_step5(
            target_folder=cfg.target_folder,        
            reference_folder=reference_folder,     
            output_dir=out_dir,
            do_ged=cfg.do_ged,
            do_nged=cfg.do_nged,
            do_faged=cfg.do_faged,
            timeout=cfg.timeout,
            target_category_key=cfg.target_category_key,
            reference_category_key=cfg.reference_category_key,
        )

        num_targets = len([f for f in os.listdir(cfg.target_folder) if f.endswith(".pkl")])
        num_refs = len([f for f in os.listdir(reference_folder) if f.endswith(".pkl")])

        rows.append({
            "markov_folder": markov_name,
            "target_folder": cfg.target_folder,
            "reference_folder": reference_folder,
            "output_dir": out_dir,
            "num_target_graphs": num_targets,
            "num_reference_graphs": num_refs,
            "ran_ok": bool(ok),
        })


    summary_df = pd.DataFrame(rows)

    if cfg.write_summary_csv:
        summary_path = os.path.join(cfg.step5_output_root, cfg.summary_filename)
        summary_df.to_csv(summary_path, index=False)

    return summary_df



def merge_step5_csvs_to_long_table(
    csv_root: str,
    *,
    output_csv: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read all step5 CSVs like:
      Group_A_GED.csv / Group_A_nGED.csv / Group_A_faGED.csv
    Each CSV has columns: Reference_Graph, Edit_Distance

    Return long table columns:
      Group, Reference_Graph, Edit_Distance_Type, Edit_Distance, Rank
    """
    rows = []

    # match: Group_A_GED.csv
    pat = re.compile(r"^(?P<group>Group_[A-Za-z0-9]+)_(?P<dtype>GED|nGED|faGED)\.csv$", re.IGNORECASE)

    for fn in sorted(os.listdir(csv_root)):
        if not fn.endswith(".csv"):
            continue
        m = pat.match(fn)
        if not m:
            continue

        group = m.group("group")
        dtype = m.group("dtype")  # GED / nGED / faGED

        p = os.path.join(csv_root, fn)
        df = pd.read_csv(p)

        if "Reference_Graph" not in df.columns or "Edit_Distance" not in df.columns:
            continue

        piece = df[["Reference_Graph", "Edit_Distance"]].copy()
        piece["Group"] = group
        piece["Edit_Distance_Type"] = dtype
        # make sure numeric
        piece["Edit_Distance"] = pd.to_numeric(piece["Edit_Distance"], errors="coerce")

        rows.append(piece)

    out_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=["Group", "Reference_Graph", "Edit_Distance_Type", "Edit_Distance", "Rank"]
    )

    if not out_df.empty:
        # rank within each (Group, Type)
        out_df["Rank"] = (
            out_df.groupby(["Group", "Edit_Distance_Type"])["Edit_Distance"]
            .rank(method="min", ascending=True)
            .astype("Int64")
        )

        # optional: sort nicely
        out_df = out_df.sort_values(
            by=["Group", "Edit_Distance_Type", "Rank", "Edit_Distance", "Reference_Graph"],
            ascending=[True, True, True, True, True],
        ).reset_index(drop=True)

    if output_csv:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        out_df.to_csv(output_csv, index=False)

    # final column order as you asked
    return out_df[["Reference_Graph", "Edit_Distance_Type", "Edit_Distance", "Rank", "Group"]]
