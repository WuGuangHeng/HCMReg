"""
Convert MambaFusev6 checkpoints to HCMReg-compatible format.

Usage:
    python convert_weights.py --dataset LPBA40
    python convert_weights.py --dataset IXI
    python convert_weights.py --dataset MindBoggle
    python convert_weights.py --all
"""
import os
import sys
import argparse
import torch

from config import get_config, PRETRAINED_WEIGHTS, get_pretrained_path


def convert_checkpoint(dataset_name, experiments_root='../experiments',
                       output_dir='weights', verify=True):
    """
    Convert a MambaFusev6 checkpoint to HCMReg format.

    The MambaFusev6 checkpoint contains:
    - encoder.* (compatible after encoder fix in Phase 0)
    - cm1-5.fusion_block.* (100% compatible)
    - peblock1-5.norm.* (removed in HCMReg - will be skipped)
    - transformer.* (compatible)
    - diffs.* (optional, removed in HCMReg by default)

    HCMReg adds:
    - cm1-5.norm.* (new LayerNorm before fusion blocks)
    """
    if dataset_name not in PRETRAINED_WEIGHTS:
        print(f"No pretrained weights configured for {dataset_name}")
        print(f"Available: {list(PRETRAINED_WEIGHTS.keys())}")
        return None

    info = PRETRAINED_WEIGHTS[dataset_name]
    config = get_config(dataset_name)
    config['dataset'] = dataset_name

    # Source checkpoint path
    src_path = os.path.join(experiments_root, info['experiment_dir'],
                            info['best_checkpoint'])
    if not os.path.exists(src_path):
        print(f"Source checkpoint not found: {src_path}")
        return None

    print(f"Loading MambaFusev6 checkpoint: {src_path}")
    ckpt = torch.load(src_path, map_location='cpu')

    # Destination path
    dst_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dst_dir, exist_ok=True)
    dst_name = f"HCMReg_{dataset_name}_dsc{info['best_dsc']:.3f}.pth.tar"
    dst_path = os.path.join(dst_dir, dst_name)

    # Create HCMReg model to verify
    sys.path.insert(0, '.')
    sys.path.insert(0, './model')  # model files use non-relative imports
    from HCMReg import HCMReg

    img_size = config['img_size']
    model = HCMReg(inshape=img_size)

    # Load weights
    state_dict = ckpt.get('state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    print(f"  Missing keys (new HCMReg layers): {len(missing)}")
    for k in missing:
        print(f"    {k}")
    print(f"  Unexpected keys (MambaFusev6-specific): {len(unexpected)}")
    for k in unexpected:
        print(f"    {k}")

    # Create HCMReg-native checkpoint
    # Strip peblock keys and keep only HCMReg-compatible weights
    hcmreg_state = model.state_dict()
    for key in list(state_dict.keys()):
        if key in hcmreg_state:
            hcmreg_state[key] = state_dict[key]

    # Save
    save_dict = {
        'state_dict': hcmreg_state,
        'best_dsc': info['best_dsc'],
        'dataset': dataset_name,
        'source': src_path,
        'converted_from': 'MambaFusev6',
        'arch': 'HCMReg',
    }
    torch.save(save_dict, dst_path)
    print(f"Saved HCMReg checkpoint to: {dst_path}")

    if verify:
        # Quick verification: load back and check
        reloaded = torch.load(dst_path, map_location='cpu')
        model2 = HCMReg(inshape=img_size)
        missing2, unexpected2 = model2.load_state_dict(
            reloaded['state_dict'], strict=False)
        # After conversion, there should be no unexpected keys
        # Missing keys should only be cm*.norm (which were initialized randomly)
        if len(unexpected2) == 0 and all('cm' in k and 'norm' in k for k in missing2):
            print("  ✓ Verification passed: checkpoint loads cleanly")
        else:
            print(f"  ⚠ Verification: {len(missing2)} missing, {len(unexpected2)} unexpected")

    return dst_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert MambaFusev6 checkpoints to HCMReg format')
    parser.add_argument('--dataset', type=str, default=None,
                        help='Dataset name (LPBA40, IXI, MindBoggle)')
    parser.add_argument('--all', action='store_true',
                        help='Convert all available checkpoints')
    parser.add_argument('--experiments-root', type=str,
                        default='../experiments',
                        help='Path to experiments directory')
    parser.add_argument('--output-dir', type=str, default='weights',
                        help='Output directory for converted weights')
    parser.add_argument('--no-verify', action='store_true',
                        help='Skip verification step')
    args = parser.parse_args()

    if args.all:
        datasets = list(PRETRAINED_WEIGHTS.keys())
    elif args.dataset:
        datasets = [args.dataset]
    else:
        parser.print_help()
        return

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Converting {dataset}...")
        print(f"{'='*60}")
        result = convert_checkpoint(
            dataset,
            experiments_root=args.experiments_root,
            output_dir=args.output_dir,
            verify=not args.no_verify
        )
        if result:
            print(f"✓ {dataset} converted successfully")
        else:
            print(f"✗ {dataset} conversion failed")


if __name__ == '__main__':
    main()
