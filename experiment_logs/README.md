# Original MambaFusev6 Experiment Logs

These are the original training logs preserved from the MambaFuse project.
Each directory contains `logfile.log` (training output) and `losses and dice.txt` (per-epoch metrics).

## Experiments

| Directory | Dataset | Model | Best DSC |
|-----------|---------|-------|----------|
| `IXI-MambaFusev6-hybrid-defconv-diff_ncc_1_reg_1_jac_0_dice_0_lr_0.0001/` | IXI | MambaFusev6-hybrid-defconv-diff | 0.769 |
| `LPBA40-MambaFusev6-defconv-diff_ncc_1_reg_1_jac_0_dice_0_lr_0.001/` | LPBA40 | MambaFusev6-defconv-diff | — |
| `MindBoggle-MambaFusev6-hybrid-defconv-diff_ncc_1_reg_1_jac_0_dice_0_lr_0.001/` | MindBoggle | MambaFusev6-hybrid-defconv-diff | 0.637 |

## Pretrained Weights

The corresponding pretrained weights for these experiments (where available)
are in `../weights/` after running `../convert_weights.py`.
