# Faged_building_layout Pipeline

**Faged_building_layout Pipeline** is a **research-oriented** Python library for comparing **spatial layouts** and **human behavioral patterns** using **function-aware graph representations**.

It can also be used independently to **construct graph representations of complex architectural layouts with open spaces** and to **extract layout prototypes** from such configurations.


---

## Key features

- **Step-wise pipeline (Step0–Step5)** for reproducible experiments  
- Layout graph representation for **complex open-space buildings**  
- **Prototype extraction** via community detection (e.g., Infomap)  
- **Toged_building_layout / nged_building_layout / Faged_building_layout** computation + retrieval ranking outputs  
- Designed for architectural research workflows

---

## Installation

```bash
pip install ged_building_layout-building-layout
```

---

## Minimal example (layout graph → CaG)

```python
from ged_building_layout_building_layout import run_step2_then_step3, Step3Config

run_step2_then_step3(
    json_folder="DATA_ROOT/json",
    step2_output_folder="OUTPUT_ROOT/step2_basegraphs",
    step3_output_root="OUTPUT_ROOT/step3_transform/variants",
    step3_selected_folder="OUTPUT_ROOT/step3_transform/selected",
    step3_cfg=Step3Config(),
    save_png_step2=True,
    save_selection_csv_step3=True,
)
```

---

## Full pipeline example (Step0–Step5)

The complete step-by-step notebook and documentation are hosted on GitHub:

- **GitHub repository**: https://github.com/songguoarchitect/ged_building_layout_building_layout  
- Example notebook: `quick_start.ipynb`  

> Note: PyPI does not reliably render local images (relative paths).  
> We keep the PyPI page concise and place full visuals/examples on GitHub.

---

## Data availability

A small example dataset (5 samples) is included for demonstration.  
The complete dataset (150 images) is not publicly released due to data usage agreements.

---

## Citation

If you use this pipeline in academic work, please cite:

```bibtex
@article{GuoKeeZhuang2026,
  title   = {A Function-Aware Graph-Based Layout Retrieval for Open-space Libraries},
  author  = {Guo, Song and Kee, Tris and Zhuang, Weimin},
  journal = {Automation in Construction},
  year    = {2026}
}
```

---

## License

MIT License.
