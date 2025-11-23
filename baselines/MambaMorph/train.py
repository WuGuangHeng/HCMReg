import glob
# from torch.utils.tensorboard import SummaryWriter
import os
import losses
import utils
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from torch import optim
import torch.nn as nn
import matplotlib.pyplot as plt
from natsort import natsorted
from models.MambaMorph import MambaMorph, CONFIGS, MambaMorphFeat
import random
import wandb


def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True


same_seeds(2024)


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir+"logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main():
    batch_size = 1
    dataset = 'LPBA40'
    train_dir = dataset+'/train'
    val_dir = dataset+'/test'
    label_dir = dataset+'/label'
    atlas_dir = dataset+'/fixed.nii.gz'

    weights = [1, 1, 0, 0]  # loss weights
    lr = 0.001 # 0.001 for LPBA40
    save_dir = '{}-VMambaMorphFeat_ncc_{}_reg_{}_jac_{}_dice_{}_lr_{}/'.format(dataset,
                                                                weights[0], weights[1], weights[2], weights[3], lr)
    if not os.path.exists('experiments/' + save_dir):
        os.makedirs('experiments/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)
    sys.stdout = Logger('logs/' + save_dir)
    f = open(os.path.join('logs/'+save_dir,
             'losses and dice' + ".txt"), "a")

    epoch_start = 0
    max_epoch = 500
    if dataset == 'LPBA40':
        img_size = (160, 192, 160)
    elif dataset == 'OASIS':
        img_size = (224, 192, 160)
    elif dataset =='IXI':
        img_size = (160, 192, 224)
    else: # MindBoggle
        img_size = (160, 192, 160)

    cont_training = False

    '''
    Initialize model
    '''
    # MamaMorph
    model = MambaMorphFeat(CONFIGS["MambaMorph"])

    model.cuda()

    # log in wandb
    config = dict(
        model=model.__class__.__name__,
        dataset=dataset,
        batch_size=batch_size,
        lr=lr,
        weights=weights,
        save_dir=save_dir,
        epoch_start=epoch_start,
        max_epoch=max_epoch,
        img_size=img_size,
        cont_training=cont_training,
    )
    wandb.init(project='MambaFuseExps', name=save_dir, config=config, resume=False)
    
    # spatial_trans = SpatialTransformer(img_size).cuda()
    '''
    Initialize spatial transformation function
    '''
    reg_model = utils.register_model(img_size, 'nearest')

    diff = False
    if diff:
        diff_transform = utils.VecInt(img_size).cuda()
    
    reg_model.cuda()

    '''
    If continue from previous training
    '''
    if cont_training:
        model_dir = 'experiments/'+save_dir
        updated_lr = round(
            lr * np.power(1 - (epoch_start) / max_epoch, 0.9), 8)
        best_model = torch.load(
            model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        model.load_state_dict(best_model)
        print(model_dir + natsorted(os.listdir(model_dir))[-1])
    else:
        updated_lr = lr

    '''
    Initialize training
    '''
    train_composed = transforms.Compose(
        [
            # trans.CenterCropBySize(img_size),
            # trans.MinMax_norm(),
            trans.Seg_norm(dataset=dataset),
            trans.NumpyType((np.float32, np.int16)),
        ])

    val_composed = transforms.Compose([
        # trans.CenterCropBySize(img_size),
        # trans.MinMax_norm(),
        trans.Seg_norm(dataset=dataset),
        trans.NumpyType((np.float32, np.int16)),
    ])
    if dataset == 'OASIS' or dataset == 'MindBoggle':
        train_set = datasets.BrainDatasetS2S(
            train_dir, label_path=label_dir, transforms=train_composed, dataset=dataset)
        val_set = datasets.BrainInferDatasetS2S(
            val_dir, label_path=label_dir, transforms=val_composed, dataset=dataset)
    else: # IXI, LPBA40
        train_set = datasets.BrainDatasetA2S(
            train_dir, label_path=label_dir, atlas_path=atlas_dir, transforms=train_composed, dataset=dataset)
        val_set = datasets.BrainInferDatasetA2S(
            val_dir, label_path=label_dir, atlas_path=atlas_dir, transforms=val_composed, dataset=dataset)
        
    print('Train set size: ', len(train_set))
    print('Val set size: ', len(val_set))
    print('Train image size: ', train_set[0][0].shape)
    print('Val image size: ', val_set[0][0].shape)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False,
                            num_workers=16, pin_memory=True, drop_last=True)

    optimizer = optim.Adam(
        model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterion = losses.NCC_vxm()
    criterions = [criterion]
    criterions += [losses.Grad3d(penalty='l2')]
    criterions += [losses.JacboianLoss(), losses.DiceLoss(dataset=dataset)]

    best_dsc = 0
    for epoch in range(epoch_start, max_epoch):
        print('Training Starts')
        '''
        Training
        '''
        loss_all = utils.AverageMeter()
        idx = 0
        for data in train_loader:
            idx += 1
            model.train()
            # adjust_learning_rate(optimizer, epoch, max_epoch, lr) # no need for LPBA40
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            # x_seg = data[2]
            # y_seg = data[3]

            
            output = model(x, y)
            # if diff:
            #     output[1] = diff_transform(output[1])
            
            loss = 0
            ncc_loss = criterions[0](output[0], y)*weights[0]
            reg_loss = criterions[1](output[1], y)*weights[1]
            if weights[2] != 0:
                jac_loss = criterions[2](output[1])*weights[2]
            else:
                jac_loss = torch.tensor(0)
    
            if weights[3] != 0:
                moved_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                dice_loss = criterions[3](moved_out, y_seg)*weights[3]
            else:
                dice_loss = torch.tensor(0)
        
            curr_loss = ncc_loss + reg_loss + jac_loss + dice_loss
            loss += curr_loss
            loss_all.update(loss.item(), y.numel())
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}, Jac: {:.6f}, Dice: {:.6f}'.format(
                idx, len(train_loader), loss.item(), ncc_loss.item(), reg_loss.item(), jac_loss.item(), dice_loss.item()))
            
            # # backward
            # output = model(y, x)
            # if diff:
            #     output[1] = diff_transform(output[1])
            # loss = 0
            
            # ncc_loss = criterions[0](output[0], x)*weights[0]
            # reg_loss = criterions[1](output[1], x)*weights[1]
            # if weights[2] != 0:
            #     jac_loss = criterions[2](output[1])*weights[2]
            # else:
            #     jac_loss = torch.tensor(0)
            
            # if weights[3] != 0:
            #     moved_out = reg_model([y_seg.cuda().float(), output[1].cuda()])
            #     dice_loss = criterions[3](moved_out, x_seg)*weights[3]
            # else:
            #     dice_loss = torch.tensor(0)
            
            # curr_loss = ncc_loss + reg_loss + jac_loss + dice_loss
            # loss += curr_loss
            # loss_all.update(loss.item(), y.numel())
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            # print('Iter {} of {} loss {:.4f}, Img Sim: {:.6f}, Reg: {:.6f}, Jac: {:.6f}, Dice: {:.6f}'.format(
            #     idx, len(train_loader), loss.item(), ncc_loss.item(), reg_loss.item(), jac_loss.item(), dice_loss.item()))

        print('{} Epoch {} loss {:.4f}'.format(save_dir, epoch, loss_all.avg))
        print('Epoch {} loss {:.4f}'.format(
            epoch, loss_all.avg), file=f, end=' ')
        wandb.log({'train_loss': loss_all.avg}, step=epoch)
        '''
        Validation
        '''
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
                # if diff:
                #     output[1] = diff_transform(output[1])
                def_out = reg_model([x_seg.cuda().float(), output[1].cuda()])
                dsc = utils.dice_val_VOI(
                    def_out.long(), y_seg.long(), dataset=dataset)
                eval_dsc.update(dsc.item(), x.size(0))
                print(epoch, ':', eval_dsc.avg)
                # log the images with wandb
                # grid_img = mk_grid_img(8, 1, img_size)
                # def_grid = spatial_trans(grid_img.float(), output[1].cuda())
                # grid_fig = comput_fig(def_grid)
                # pred_fig = comput_fig(output[0])  # wrapped_moving
                # moving_fig = comput_fig(x_seg)
                # fixed_fig = comput_fig(y_seg)
                # wandb.log({'pred': wandb.Image(pred_fig),
                #            'moving': wandb.Image(moving_fig),
                #            'fixed': wandb.Image(fixed_fig),
                #            'grid': wandb.Image(grid_fig)}, step=epoch)
                # del grid_img, def_grid, grid_fig, pred_fig, moving_fig, fixed_fig

        best_dsc = max(eval_dsc.avg, best_dsc)
        print(eval_dsc.avg, file=f)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_dsc': best_dsc,
            'optimizer': optimizer.state_dict(),
        }, save_dir='experiments/' + save_dir, filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg))
        wandb.log({'val_dsc': eval_dsc.avg}, step=epoch)
        loss_all.reset()
        wandb.summary['best_dsc'] = best_dsc
    wandb.finish()


def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j+line_thickness-1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i+line_thickness-1] = 1
    grid_img = grid_img[None, None, ...]
    grid_img = torch.from_numpy(grid_img).cuda()
    return grid_img


def adjust_learning_rate(optimizer, epoch, MAX_EPOCHES, INIT_LR, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(
            INIT_LR * np.power(1 - (epoch) / MAX_EPOCHES, power), 8)


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir+filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))


if __name__ == '__main__':
    '''

    GPU configuration
    '''
    GPU_iden = 3
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()
