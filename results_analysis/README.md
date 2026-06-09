# Results Analysis

Quantitative results processing and boxplot generation tools.

## Files

- `analysis.py` — Generates per-structure DSC/HD boxplots from raw result CSVs.
  - Reads per-subject DSC and HD values for each anatomical structure
  - Groups bilateral structures (e.g., Left/Right Thalamus → Thalamus)
  - Outputs BoxPlot CSV files and prints summary statistics

- `IXI_BoxPlot_Data/` — Raw result CSVs for IXI dataset experiments:
  - `IXI-MambaFusev6-hybrid-defconv_ncc_1_reg_1_jac_0_dice_0_lr_0.0001.csv` — Per-subject Dice and HD95
  - `IXI-MambaFusev6-hybrid-defconv-diff_ncc_1_reg_1_jac_0_dice_0_lr_0.0001.csv` — Per-subject Dice and HD95
  - `IXI-MambaFusev6-hybrid-defconv-diff_BoxPlot.csv` — Aggregated per-structure boxplot data

## Usage

```bash
cd HCMReg/
python results_analysis/analysis.py
```

To analyze a different experiment, edit the `file_name` list in `analysis.py`.
