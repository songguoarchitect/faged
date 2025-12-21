import argparse
from pathlib import Path

from .pipeline import Step4Step5Config, run_step4_then_step5
from .step4_prototype import InfomapConfig
from .step5_faged import Step5BatchConfig

def main():
    p = argparse.ArgumentParser(prog="faged")
    p.add_argument("--output-root", type=str, required=True)
    p.add_argument("--markov", type=float, nargs="+", default=[0.7])
    args = p.parse_args()

    OUTPUT_ROOT = Path(args.output_root)

    cfg = Step4Step5Config(
        step4=InfomapConfig(
            input_folder=str(OUTPUT_ROOT / "step3_transform" / "selected"),
            graph_output_root=str(OUTPUT_ROOT / "step4_prototype" / "graphs"),
            community_img_output_root=str(OUTPUT_ROOT / "step4_prototype" / "communities"),
            markov_times=tuple(args.markov),
            main_function_mode="count",
        ),
        step5=Step5BatchConfig(
            step4_graph_output_root=str(OUTPUT_ROOT / "step4_prototype" / "graphs"),
            target_folder=str(OUTPUT_ROOT / "step1_behavior"),
            step5_output_root=str(OUTPUT_ROOT / "step5_faged"),
            do_ged=True, do_nged=True, do_faged=True,
            timeout=30,
        ),
    )
    run_step4_then_step5(cfg)

if __name__ == "__main__":
    main()
