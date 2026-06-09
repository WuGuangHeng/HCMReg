"""
Inference script for HCMReg on MindBoggle dataset.
Loads pretrained MambaFusev6 weights (dsc0.637).
Usage: python infer_MB.py
"""
import os
import torch
from config import get_config, get_gpu, get_pretrained_path
from infer_common import run_inference, load_model_for_inference

if __name__ == '__main__':
    GPU_iden = get_gpu('MindBoggle')
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))

    config = get_config('MindBoggle')
    config['dataset'] = 'MindBoggle'

    # Load pretrained MambaFusev6 weights
    ckpt_path = get_pretrained_path('MindBoggle')
    print(f'Loading pretrained weights from: {ckpt_path}')
    model = load_model_for_inference(config, ckpt_path)

    results = run_inference(config, model=model)
    print(f'\nResults: DSC={results["dsc_def"]:.3f}, HD={results["hd_def"]:.2f}, '
          f'Det%={results["det_pct"]:.4f}, Time={results["avg_time"]:.3f}s')
