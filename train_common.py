"""
Shared training utilities for HCMReg.
Extracted from the original train.py to support multi-dataset training.
"""

import glob
import os
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import matplotlib.pyplot as plt
from natsort import natsorted
from model.HCMReg import HCMReg
import losses
import utils

import random
import wandb


def same_seeds(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


class Logger(object):
    """Tee stdout to a log file."""
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    """Save checkpoint and keep only the last max_model_num files."""
    torch.save(state, save_dir + filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    """Polynomial learning rate decay."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(
            INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def comput_fig(img):
    """Create a figure grid for visualization."""
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    """Create a grid image for deformation visualization."""
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j + line_thickness - 1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i + line_thickness - 1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


def build_datasets(config):
    """
    Build train/val datasets and dataloaders from a config dict.
    Config must contain: train_dir, val_dir, label_dir, atlas_dir, type, img_size
    """
    dataset_name = config.get('dataset', 'LPBA40')

    train_composed = transforms.Compose([
        trans.Seg_norm(dataset=dataset_name),
        trans.NumpyType((np.float32, np.int16)),
    ])

    val_composed = transforms.Compose([
        trans.Seg_norm(dataset=dataset_name),
        trans.NumpyType((np.float32, np.int16)),
    ])

    is_s2s = (config['type'] == 'S2S')
    if is_s2s:
        train_set = datasets.BrainDatasetS2S(
            config['train_dir'], label_path=config['label_dir'],
            transforms=train_composed, dataset=dataset_name)
        val_set = datasets.BrainInferDatasetS2S(
            config['val_dir'], label_path=config['label_dir'],
            transforms=val_composed, dataset=dataset_name)
    else:  # A2S: LPBA40, IXI
        train_set = datasets.BrainDatasetA2S(
            config['train_dir'], label_path=config['label_dir'],
            atlas_path=config['atlas_dir'],
            transforms=train_composed, dataset=dataset_name)
        val_set = datasets.BrainInferDatasetA2S(
            config['val_dir'], label_path=config['label_dir'],
            atlas_path=config['atlas_dir'],
            transforms=val_composed, dataset=dataset_name)

    print('Train set size: ', len(train_set))
    print('Val set size: ', len(val_set))
    print('Train image size: ', train_set[0][0].shape)
    print('Val image size: ', val_set[0][0].shape)

    train_loader = DataLoader(
        train_set, batch_size=1, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(
        val_set, batch_size=1, shuffle=False, num_workers=16,
        pin_memory=True, drop_last=True)

    return train_loader, val_loader


def build_model(config):
    """Build the HCMReg model from config."""
    img_size = config['img_size']
    model = HCMReg(inshape=img_size)
    model.cuda()
    return model


def build_criterions(config):
    """Build loss criterions."""
    dataset_name = config.get('dataset', 'LPBA40')
    criterions = [losses.NCC_vxm()]
    criterions += [losses.Grad3d(penalty='l2')]
    criterions += [losses.JacboianLoss(), losses.DiceLoss(dataset=dataset_name)]
    return criterions


def train_epoch(model, reg_model, optimizer, criterions, weights, train_loader, epoch, diff=False, diff_transform=None):
    """
    Run one training epoch (bidirectional).
    Now correctly handles 4-value unpack from all dataset types.
    """
    loss_all = utils.AverageMeter()
    idx = 0
    for data in train_loader:
        idx += 1
        model.train()
        data = [t.cuda() for t in data]
        x = data[0]
        y = data[1]
        x_seg = data[2]
        y_seg = data[3]

        # Forward: moving -> fixed
        output = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss = 0
        ncc_loss = criterions[0](output[0], y) * weights[0]
        reg_loss = criterions[1](output[1], y) * weights[1]

        if weights[2] != 0:
            jac_loss = criterions[2](output[1]) * weights[2]
        else:
            jac_loss = torch.tensor(0)

        if weights[3] != 0:
            moved_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
            dice_loss = criterions[3](moved_out, y_seg) * weights[3]
        else:
            dice_loss = torch.tensor(0)

        curr_loss = ncc_loss + reg_loss + jac_loss + dice_loss
        loss += curr_loss
        loss_all.update(loss.item(), y.numel())

        loss.backward()
        optimizer.step()

        print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}, Jac: {:.6f}, Dice: {:.6f}'.format(
            idx, len(train_loader), loss.item(), ncc_loss.item(),
            reg_loss.item(), jac_loss.item(), dice_loss.item()))

        # Backward: fixed -> moving
        output = model(y, x)
        if diff and diff_transform is not None:
            output[1] = diff_transform(output[1])
        loss = 0

        ncc_loss = criterions[0](output[0], x) * weights[0]
        reg_loss = criterions[1](output[1], x) * weights[1]

        if weights[2] != 0:
            jac_loss = criterions[2](output[1]) * weights[2]
        else:
            jac_loss = torch.tensor(0)

        if weights[3] != 0:
            moved_out = reg_model([y_seg.cuda().float(), output[1].cuda()])
            dice_loss = criterions[3](moved_out, x_seg) * weights[3]
        else:
            dice_loss = torch.tensor(0)

        curr_loss = ncc_loss + reg_loss + jac_loss + dice_loss
        loss += curr_loss
        loss_all.update(loss.item(), y.numel())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}, Jac: {:.6f}, Dice: {:.6f}'.format(
            idx, len(train_loader), loss.item(), ncc_loss.item(),
            reg_loss.item(), jac_loss.item(), dice_loss.item()))

    return loss_all.avg


def validate_epoch(model, reg_model, val_loader, dataset_name='LPBA40'):
    """Run one validation epoch."""
    eval_dsc = utils.AverageMeter()
    with torch.no_grad():
        for data in val_loader:
            model.eval()
            reg_model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]
            output = model(x, y)
            def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
            dsc, _ = utils.dice_val_VOI(
                def_out.long(), y_seg.long(), dataset=dataset_name)
            eval_dsc.update(dsc.item(), x.size(0))
    return eval_dsc.avg


def main(config):
    """
    Main training entry point.
    Config dict must contain all dataset-specific parameters.
    """
    dataset_name = config.get('dataset', 'LPBA40')
    weights = config['weights']
    lr = config['lr']
    img_size = config['img_size']
    max_epoch = config.get('max_epoch', 500)
    cont_training = config.get('cont_training', False)
    diff = config.get('diff', False)
    seed = config.get('seed', 2024)

    same_seeds(seed)

    # Build save directory name (consistent with MambaFuse naming convention)
    save_dir = '{}-HCMReg_ncc_{}_reg_{}_jac_{}_dice_{}_lr_{}/'.format(
        dataset_name, weights[0], weights[1], weights[2], weights[3], lr)

    if not os.path.exists('experiments/' + save_dir):
        os.makedirs('experiments/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)
    sys.stdout = Logger('logs/' + save_dir)
    f = open(os.path.join('logs/' + save_dir, 'losses and dice' + ".txt"), "a")

    epoch_start = 0

    # Initialize model
    model = build_model(config)

    # Wandb logging
    wandb_config = dict(
        model=model.__class__.__name__,
        dataset=dataset_name,
        batch_size=1,
        lr=lr,
        weights=weights,
        save_dir=save_dir,
        epoch_start=epoch_start,
        max_epoch=max_epoch,
        img_size=img_size,
        cont_training=cont_training,
    )
    wandb.init(project='HCMRegExps', name=save_dir, config=wandb_config, resume=False)

    # Spatial transformation
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()

    diff_transform = None
    if diff:
        diff_transform = utils.VecInt(img_size).cuda()

    # Continue training
    updated_lr = lr
    if cont_training:
        model_dir = 'experiments/' + save_dir
        updated_lr = round(lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)
        best_model = torch.load(
            model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        model.load_state_dict(best_model)
        print(model_dir + natsorted(os.listdir(model_dir))[-1])

    # Build datasets
    train_loader, val_loader = build_datasets(config)

    # Optimizer and losses
    optimizer = optim.Adam(
        model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterions = build_criterions(config)

    best_dsc = 0
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')

        # Training
        train_loss = train_epoch(
            model, reg_model, optimizer, criterions, weights,
            train_loader, epoch, diff=diff, diff_transform=diff_transform)

        print('{} Epoch {} loss {:.4f}'.format(save_dir, epoch, train_loss))
        print('Epoch {} loss {:.4f}'.format(epoch, train_loss), file=f, end=' ')
        wandb.log({'train_loss': train_loss}, step=epoch)

        # Validation
        eval_dsc = validate_epoch(
            model, reg_model, val_loader, dataset_name=dataset_name)

        best_dsc = max(eval_dsc, best_dsc)
        print(eval_dsc, file=f)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir='experiments/' + save_dir,
           filename='dsc{:.3f}.pth.tar'.format(eval_dsc))
        wandb.log({'val_dsc': eval_dsc}, step=epoch)
        loss_all = utils.AverageMeter()  # reset for next epoch
        wandb.summary['best_dsc'] = best_dsc

    wandb.finish()
    f.close()
