"""
Shared inference/evaluation utilities for HCMReg.
Extracted from the original infer.py to support multi-dataset evaluation.
"""

import os
import numpy as np
import torch
import time
from torch.utils.data import DataLoader
from torchvision import transforms
from natsort import natsorted
from thop import profile
from thop import clever_format
from medpy.metric.binary import hd95

from model.HCMReg import HCMReg
from data import datasets, trans
import utils


def same_seeds(seed):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def compute_hausdorff_distance(pred, gt, dataset='LPBA40'):
    """Compute average HD95 across all labels for a dataset."""
    if dataset == 'LPBA40':
        cls_lst = [i for i in range(1, 57)]
    elif dataset == 'MindBoggle':
        cls_lst = [i for i in range(1, 63)]
    elif dataset == 'OASIS':
        cls_lst = [i for i in range(1, 36)]
    elif dataset == 'IXI':
        cls_lst = [i for i in range(1, 31)]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    hd_lst = []
    for cls in cls_lst:
        hd = hd95(gt == cls, pred == cls)
        hd_lst.append(hd)
    return np.mean(hd_lst)


def build_test_dataset(config):
    """Build test dataset and dataloader."""
    dataset_name = config.get('dataset', 'LPBA40')

    test_composed = transforms.Compose([
        trans.Seg_norm(dataset=dataset_name),
        trans.NumpyType((np.float32, np.int16)),
    ])

    is_s2s = (config['type'] == 'S2S')
    if is_s2s:
        test_set = datasets.BrainInferDatasetS2S(
            config['val_dir'], label_path=config['label_dir'],
            transforms=test_composed, dataset=dataset_name)
    else:
        test_set = datasets.BrainInferDatasetA2S(
            config['val_dir'], label_path=config['label_dir'],
            atlas_path=config['atlas_dir'],
            transforms=test_composed, dataset=dataset_name)

    print('Test set size: {}'.format(len(test_set)))

    test_loader = DataLoader(
        test_set, batch_size=1, shuffle=False,
        num_workers=16, pin_memory=True, drop_last=True)

    return test_loader


def load_model_for_inference(config, checkpoint_path=None, model_idx=-1):
    """
    Load HCMReg model for inference.
    If checkpoint_path is provided, load from that path.
    Otherwise, auto-find the best checkpoint from the experiments directory.
    """
    img_size = config['img_size']
    model = HCMReg(inshape=img_size).cuda()

    if checkpoint_path is None:
        # Try to auto-find checkpoint
        weights = config.get('weights', [1, 1, 0, 0])
        lr = config.get('lr', 0.001)
        dataset_name = config.get('dataset', 'LPBA40')
        model_folder = '{}-HCMReg_ncc_{}_reg_{}_jac_{}_dice_{}_lr_{}/'.format(
            dataset_name, weights[0], weights[1], weights[2], weights[3], lr)
        model_dir = 'experiments/' + model_folder
        if os.path.exists(model_dir):
            checkpoint_path = model_dir + natsorted(os.listdir(model_dir))[model_idx]
        else:
            raise FileNotFoundError(
                f"No checkpoint found at {model_dir}. "
                f"Please provide a checkpoint_path or train the model first.")

    print('Loading checkpoint: {}'.format(checkpoint_path))
    ckpt = torch.load(checkpoint_path, map_location='cuda')

    # Handle both HCMReg-native and MambaFusev6 checkpoints
    state_dict = ckpt.get('state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print('Missing keys (new HCMReg layers): {}'.format(len(missing)))
    if unexpected:
        print('Unexpected keys (MambaFusev6-specific): {}'.format(len(unexpected)))

    model.cuda()
    return model


def run_inference(config, model=None, checkpoint_path=None, output_dir='results'):
    """
    Run full inference evaluation.
    Computes DSC, HD95, Jacobian determinant %, runtime, FLOPs, and Params.
    """
    dataset_name = config.get('dataset', 'LPBA40')
    img_size = config['img_size']
    weights = config.get('weights', [1, 1, 0, 0])
    lr = config.get('lr', 0.001)

    same_seeds(2024)

    # Build model
    if model is None:
        model = load_model_for_inference(config, checkpoint_path)

    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()

    # Build test dataset
    test_loader = build_test_dataset(config)

    # Output file
    model_folder = '{}-HCMReg_ncc_{}_reg_{}_jac_{}_dice_{}_lr_{}/'.format(
        dataset_name, weights[0], weights[1], weights[2], weights[3], lr)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    f = open(os.path.join(output_dir, model_folder[:-1] + ".txt"), "w")

    eval_dsc_def = utils.AverageMeter()
    eval_dsc_raw = utils.AverageMeter()
    eval_det = utils.AverageMeter()
    eval_hd_def = utils.AverageMeter()
    eval_hd_raw = utils.AverageMeter()
    TIME = []

    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            start = time.time()
            x_def, flow = model(x, y)
            TIME.append(time.time() - start)

            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]

            # Jacobian determinant
            jac_det = utils.jacobian_determinant_vxm(
                flow.detach().cpu().numpy()[0, :, :, :, :])
            eval_det.update(
                np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))

            # Dice score
            dsc_def, values = utils.dice_val_VOI(
                def_out.long(), y_seg.long(), dataset=dataset_name)
            dsc_raw, _ = utils.dice_val_VOI(
                x_seg.long(), y_seg.long(), dataset=dataset_name)

            print('{}: Deformed dsc: {:.4f}, Raw dsc: {:.4f}'.format(
                stdy_idx, dsc_def.item(), dsc_raw.item()), file=f)
            eval_dsc_def.update(dsc_def.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))

            # Hausdorff distance
            hd_def = compute_hausdorff_distance(
                def_out.detach().cpu().numpy()[0, 0, :, :, :],
                y_seg.detach().cpu().numpy()[0, 0, :, :, :],
                dataset=dataset_name)
            hd_raw = compute_hausdorff_distance(
                x_seg.detach().cpu().numpy()[0, 0, :, :, :],
                y_seg.detach().cpu().numpy()[0, 0, :, :, :],
                dataset=dataset_name)
            print('{}: Deformed hd: {:.4f}, Raw hd: {:.4f}'.format(
                stdy_idx, hd_def, hd_raw), file=f)
            eval_hd_def.update(hd_def, x.size(0))
            eval_hd_raw.update(hd_raw, x.size(0))
            print('Processing {} / {}'.format(stdy_idx, len(test_loader.dataset)))
            stdy_idx += 1

        # Print summary
        print('Deformed DSC: {:.3f} +- {:.3f}, Raw DSC: {:.3f} +- {:.3f}'.format(
            eval_dsc_def.avg, eval_dsc_def.std,
            eval_dsc_raw.avg, eval_dsc_raw.std), file=f)
        print('Deformed HD: {:.3f} +- {:.3f}, Raw HD: {:.3f} +- {:.3f}'.format(
            eval_hd_def.avg, eval_hd_def.std,
            eval_hd_raw.avg, eval_hd_raw.std), file=f)
        print('Deformed Det(%): {:.6f} +- {:.6f}'.format(
            eval_det.avg * 100, eval_det.std * 100), file=f)
        print('Average time: {:.3f}'.format(np.mean(TIME)), file=f)

        # FLOPs and params
        x1 = torch.randn(1, 1, img_size[0], img_size[1], img_size[2]).cuda()
        x2 = torch.randn(1, 1, img_size[0], img_size[1], img_size[2]).cuda()
        macs, params = profile(model, inputs=(x1, x2))
        flops, params = clever_format([macs * 2, params], "%.3f")
        print('FLOPs: {}, Params: {}'.format(flops, params), file=f)

    f.close()

    return {
        'dsc_def': eval_dsc_def.avg,
        'dsc_raw': eval_dsc_raw.avg,
        'hd_def': eval_hd_def.avg,
        'hd_raw': eval_hd_raw.avg,
        'det_pct': eval_det.avg * 100,
        'avg_time': np.mean(TIME),
    }
