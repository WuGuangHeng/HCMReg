"""
Inference script for HCMReg on OASIS dataset.
No pretrained weights available yet - uses randomly initialized model.
To use after training, set ckpt_path to the trained checkpoint.
Usage: python infer_OASIS.py
"""
import os
import torch
from config import get_config, get_gpu
from infer_common import run_inference, load_model_for_inference

if __name__ == '__main__':
    GPU_iden = get_gpu('OASIS')
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))

    config = get_config('OASIS')
    config['dataset'] = 'OASIS'

    # OASIS has no pretrained MambaFusev6 weights.
    # Use a random init model or provide a trained checkpoint path.
    # To train first: python train_OASIS.py
    ckpt_path = None  # Set to your trained checkpoint path after training

    model = load_model_for_inference(config, ckpt_path)
    results = run_inference(config, model=model)
    print(f'\nResults: DSC={results["dsc_def"]:.3f}, HD={results["hd_def"]:.2f}, '
          f'Det%={results["det_pct"]:.4f}, Time={results["avg_time"]:.3f}s')
