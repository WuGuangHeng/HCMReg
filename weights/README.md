# HCMReg Pretrained Weights

This directory contains pretrained weights for HCMReg, converted from MambaFusev6 checkpoints.

## Available Weights

| Dataset | Source | Best DSC | File |
|---------|--------|----------|------|
| LPBA40 | MambaFusev6-hybrid5 | 0.732 | `LPBA40/HCMReg_LPBA40_dsc0.732.pth.tar` |
| IXI | MambaFusev6-hybrid | 0.769 | `IXI/HCMReg_IXI_dsc0.769.pth.tar` |
| MindBoggle | MambaFusev6-hybrid | 0.637 | `MindBoggle/HCMReg_MindBoggle_dsc0.637.pth.tar` |
| OASIS | — | — | No pretrained weights yet |

## Conversion

To convert MambaFusev6 checkpoints to HCMReg format:

```bash
# Convert all available checkpoints
python convert_weights.py --all

# Convert a specific dataset
python convert_weights.py --dataset LPBA40
```

## Loading Weights

### In inference scripts
```bash
python infer_LPBA.py    # Auto-loads pretrained weights
python infer_IXI.py
python infer_MB.py
```

### In custom code
```python
import torch
from model.HCMReg import HCMReg

model = HCMReg(inshape=(160, 192, 160))
ckpt = torch.load('weights/LPBA40/HCMReg_LPBA40_dsc0.732.pth.tar')
model.load_state_dict(ckpt['state_dict'], strict=False)
model.cuda()
model.eval()
```

## Checkpoint Format

Each checkpoint is a dict with:
- `state_dict`: HCMReg model state dict
- `best_dsc`: Best validation DSC score
- `dataset`: Dataset name
- `source`: Original MambaFusev6 checkpoint path
- `converted_from`: "MambaFusev6"
- `arch`: "HCMReg"

## Architecture Notes

HCMReg was refactored from MambaFusev6 with these key differences:
1. **Removed**: PositionalEncodingLayer (`peblock*`) — only contained LayerNorm
2. **Added**: LayerNorm before each CrossMambaFusion block (`cm*.norm`)
3. **Encoder**: Made byte-for-byte compatible with MambaFusev6's HybridEncoderv2

Weights load with `strict=False` to handle these minor differences.
