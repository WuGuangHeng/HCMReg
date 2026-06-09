"""
Centralized configuration for HCMReg training and inference.

Contains per-dataset parameters and pretrained weight paths.
All paths are relative to the project root (HCMReg/).
"""

DATASET_CONFIGS = {
    'LPBA40': {
        'train_dir': 'LPBA40/train',
        'val_dir': 'LPBA40/test',
        'label_dir': 'LPBA40/label',
        'atlas_dir': 'LPBA40/fixed.nii.gz',
        'img_size': (160, 192, 160),
        'lr': 0.001,
        'weights': [1, 1, 0, 0],
        'max_epoch': 500,
        'type': 'A2S',       # Atlas-to-Subject
    },
    'IXI': {
        'train_dir': 'IXI/train',
        'val_dir': 'IXI/val',
        'label_dir': 'IXI/label',
        'atlas_dir': 'IXI/fixed.nii.gz',
        'img_size': (160, 192, 224),
        'lr': 0.0001,
        'weights': [1, 1, 0, 0],
        'max_epoch': 500,
        'type': 'A2S',
    },
    'MindBoggle': {
        'train_dir': 'MindBoggle/train_data',
        'val_dir': 'MindBoggle/test_data',
        'label_dir': 'MindBoggle/label_data',
        'atlas_dir': None,   # S2S datasets don't use an atlas
        'img_size': (160, 192, 160),
        'lr': 0.001,
        'weights': [1, 1, 0, 0],
        'max_epoch': 500,
        'type': 'S2S',       # Subject-to-Subject
    },
    'OASIS': {
        'train_dir': 'OASIS/train',
        'val_dir': 'OASIS/test',
        'label_dir': 'OASIS/label',
        'atlas_dir': None,
        'img_size': (224, 192, 160),
        'lr': 0.001,
        'weights': [1, 1, 0, 0],
        'max_epoch': 500,
        'type': 'S2S',
    },
}

# GPU assignments for each dataset (following MambaFuse conventions)
GPU_ASSIGNMENTS = {
    'LPBA40': 3,
    'IXI': 3,
    'MindBoggle': 2,
    'OASIS': 1,
}

# Pretrained MambaFusev6 checkpoints (from original experiments/)
# These can be loaded into HCMReg via load_state_dict(ckpt['state_dict'], strict=False)
PRETRAINED_WEIGHTS = {
    'LPBA40': {
        'experiment_dir': 'LPBA40-MambaFusev6-hybrid5-defconv-diff_ncc_1_reg_1_jac_0_dice_0_lr_0.001',
        'best_checkpoint': 'dsc0.732.pth.tar',
        'best_dsc': 0.732,
    },
    'IXI': {
        'experiment_dir': 'IXI-MambaFusev6-hybrid-defconv-diff_ncc_1_reg_1_jac_0_dice_0_lr_0.0001',
        'best_checkpoint': 'dsc0.769.pth.tar',
        'best_dsc': 0.769,
    },
    'MindBoggle': {
        'experiment_dir': 'MindBoggle-MambaFusev6-hybrid-defconv-diff_ncc_1_reg_1_jac_0_dice_0_lr_0.001',
        'best_checkpoint': 'dsc0.637.pth.tar',
        'best_dsc': 0.637,
    },
    # OASIS: no pretrained weights available yet
}


def get_config(dataset_name):
    """Get the configuration dict for a given dataset."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. "
                         f"Available: {list(DATASET_CONFIGS.keys())}")
    return DATASET_CONFIGS[dataset_name]


def get_pretrained_path(dataset_name, experiments_root='../experiments'):
    """Get the full path to the best pretrained checkpoint for a dataset."""
    import os
    if dataset_name not in PRETRAINED_WEIGHTS:
        return None
    info = PRETRAINED_WEIGHTS[dataset_name]
    return os.path.join(experiments_root, info['experiment_dir'], info['best_checkpoint'])


def get_gpu(dataset_name):
    """Get the GPU assignment for a given dataset."""
    return GPU_ASSIGNMENTS.get(dataset_name, 0)
