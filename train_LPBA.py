"""
Train HCMReg on LPBA40 dataset.
Usage: python train_LPBA.py
"""
import torch
from config import get_config, get_gpu
from train_common import main

if __name__ == '__main__':
    GPU_iden = get_gpu('LPBA40')
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))

    config = get_config('LPBA40')
    config['dataset'] = 'LPBA40'
    main(config)
