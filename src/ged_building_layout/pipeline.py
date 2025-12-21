# src/ged_building_layout/pipeline.py
from __future__ import annotations
from dataclasses import dataclass
import pandas as pd
import os
from typing import Optional, Dict, List, Sequence
from .step0_checks import Step0Config, run_step0_checks
from .step1_behavior import BehaviorGraphConfig, run_step1_build_behavior_graphs
from .step2_basegraph import run_step2_build_basegraphs
from .step3_transform import run_step3_transform_from_basegraphs, Step3Config
from .step4_prototype import InfomapConfig, run_step4_infomap
from .step5_faged import Step5BatchConfig, run_step5_batch_from_markov_folders

def run_step0(
    json_folder: str,
    jpg_folder: str,
    out_root: str,
    *,
    checks: Sequence[str] = ("invalid", "corridor", "labels", "connectivity"),
    cfg: Optional[Step0Config] = None,
) -> Dict[str, str]:
    return run_step0_checks(
        json_folder=json_folder,
        jpg_folder=jpg_folder,
        out_root=out_root,
        checks=checks,
        cfg=cfg,
    )


def run_step1(
    csv_dir: str,
    output_dir: str,
    *,
    node_categories: Dict[str, List[str]],
    people_counts: Dict[str, float],
    cfg: Optional[BehaviorGraphConfig] = None,
    category_color_map: Optional[Dict[str, str]] = None,
    edge_styles: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    return run_step1_build_behavior_graphs(
        csv_dir=csv_dir,
        output_dir=output_dir,
        node_categories=node_categories,
        people_counts=people_counts,
        cfg=cfg,
        category_color_map=category_color_map,
        edge_styles=edge_styles,
    )

def run_step2(
    json_folder: str,
    output_folder: str,
    *,
    save_png: bool = True,
) -> pd.DataFrame:
    """
    Step2: build base graphs WITH functional and area attributes.
    Input: raw json folder
    Output: output_folder/*.pkl (+ optional png)
    Returns: a dataframe summary
    """
    return run_step2_build_basegraphs(
        json_folder=json_folder,
        output_folder=output_folder,
        save_png=save_png,
    )


def run_step3(
    basegraph_folder: str,
    json_folder: str,
    output_root: str,
    selected_output_folder: str,
    *,
    cfg: Optional[Step3Config] = None,
    save_selection_csv: bool = True,
    save_png: bool = True,
    png_subdir: str = "png",
) -> pd.DataFrame:
    """
    Step3: generate transformed variants (relative/absolute) from base graphs,
    auto-select the best variant per file based on avg_degree_range (default 6-8).
    """
    return run_step3_transform_from_basegraphs(
        basegraph_folder=basegraph_folder,
        json_folder=json_folder,
        output_root=output_root,
        selected_output_folder=selected_output_folder,
        cfg=cfg,
        save_selection_csv=save_selection_csv,
        save_png=save_png,
        png_subdir=png_subdir,
    )


def run_step2_then_step3(
    json_folder: str,
    *,
    step2_output_folder: str,
    step3_output_root: str,
    step3_selected_folder: str,
    step3_cfg: Optional[Step3Config] = None,
    save_png_step2: bool = True,
    save_selection_csv_step3: bool = True,
    save_png_step3: bool = False,
    png_subdir_step3: str = "png",
) -> pd.DataFrame:
    """
    Convenience pipeline: Step2 -> Step3

    Returns:
      step3 selection dataframe.
    """
    os.makedirs(step2_output_folder, exist_ok=True)
    os.makedirs(step3_output_root, exist_ok=True)
    os.makedirs(step3_selected_folder, exist_ok=True)

    _ = run_step2(
        json_folder=json_folder,
        output_folder=step2_output_folder,
        save_png=save_png_step2,
    )

    df_sel = run_step3(
        basegraph_folder=step2_output_folder,
        json_folder=json_folder,
        output_root=step3_output_root,
        selected_output_folder=step3_selected_folder,
        cfg=step3_cfg,
        save_selection_csv=save_selection_csv_step3,
        save_png=save_png_step3,
        png_subdir=png_subdir_step3,
    )
    return df_sel


@dataclass(frozen=True)
class Step4Step5Config:
    step4: InfomapConfig
    step5: Step5BatchConfig

def run_step4_then_step5(cfg: Step4Step5Config) -> pd.DataFrame:
    """
    Convenience wrapper: run Step4 (prototype extraction)
    then Step5 (FaGED / GED / nGED batch comparison).
    """
    run_step4_infomap(cfg.step4)
    return run_step5_batch_from_markov_folders(cfg.step5)
