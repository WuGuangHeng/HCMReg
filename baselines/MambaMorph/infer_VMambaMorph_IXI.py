import glob
import os
import losses
import utils
from torch.utils.data import DataLoader
import torch.nn as nn
from data import datasets, trans
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from modelsv2.MambaMorph import VMambaMorphFeat,CONFIGS
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from natsort import natsorted

import random
from monai.transforms import Resize

class ResizeTransform(nn.Module):
    """
    调整变换的大小，这涉及调整矢量场的大小并重新缩放它。
    """

    def __init__(self, source_size, target_size, ndims):
        super().__init__()

        self.source_size = source_size
        self.target_size = target_size
        self.factors = [t / s for s, t in zip(source_size, target_size)]
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factors[0] < 1:
            # resize first to save memory
            x = F.interpolate(x, align_corners=True, size=self.target_size, mode=self.mode)
            x[:,0,:,:,:] = self.factors[0] * x[:,0,:,:,:]
            x[:,1,:,:,:] = self.factors[1] * x[:,1,:,:,:]
            x[:,2,:,:,:] = self.factors[2] * x[:,2,:,:,:]

        elif self.factors[0] > 1:
            # multiply first to save memory
            x[:,0,:,:,:] = self.factors[0] * x[:,0,:,:,:]
            x[:,1,:,:,:] = self.factors[1] * x[:,1,:,:,:]
            x[:,2,:,:,:] = self.factors[2] * x[:,2,:,:,:]
            x = F.interpolate(x, align_corners=True, size=self.target_size, mode=self.mode)

        # don't do anything if resize is 1
        return x

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
    # torch.backends.cudnn.deterministic = True


same_seeds(24)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.vals = []
        self.std = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.vals.append(val)
        self.std = np.std(self.vals)


def main():
    pref =""
    dataset = 'IXI'
    val_dir = pref+dataset+'/test'
    label_dir = pref+dataset+'/label'
    atlas_dir = pref+dataset+'/fixed.nii.gz'
    weights = [1, 1, 0, 0]  # loss weights
    lr = 0.001
    model_folder = '{}-VMambaMorphFeat_ncc_{}_reg_{}_jac_{}_dice_{}_lr_{}/'.format(dataset,
                                                                weights[0], weights[1], weights[2], weights[3], lr)
    model_idx = -1
    model_dir = pref+'experiments/' + model_folder

    if 'val' in val_dir:
        csv_name = model_folder[:-1]+'_Val'
    else:
        csv_name = model_folder[:-1]
    dict = utils.process_label()
    if not os.path.exists('Quantitative_Results/'):
        os.makedirs('Quantitative_Results/')
    if os.path.exists('Quantitative_Results/'+csv_name+'.csv'):
        os.remove('Quantitative_Results/'+csv_name+'.csv')
    csv_writter(model_folder[:-1], 'Quantitative_Results/' + csv_name)
    line = ''
    for i in range(46):
        line = line + ',' + dict[i]
    csv_writter(line +','+'non_jec', 'Quantitative_Results/' + csv_name)

    img_size_s = (128, 128, 128)
    img_size = (160, 192, 224)

    model = VMambaMorphFeat(CONFIGS["VMambaMorph"])
    
    best_model = torch.load(
        model_dir + natsorted(os.listdir(model_dir))[model_idx],map_location='cuda')['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    test_composed = transforms.Compose([
        # trans.CenterCropBySize(img_size),
        # trans.Resize_img(img_size_s),
        trans.Seg_norm(dataset=dataset),
        trans.NumpyType((np.float32, np.int16)),
    ])
    if dataset == 'OASIS' or dataset == 'MindBoggle':
        test_set = datasets.BrainInferDatasetS2S(
            val_dir, label_path=label_dir, transforms=test_composed, dataset=dataset)
    else: # LPBA40, IXI
        test_set = datasets.BrainInferDatasetA2S(
            val_dir, label_path=label_dir, atlas_path=atlas_dir, transforms=test_composed, dataset=dataset)
    
    print('Test set size: {}'.format(len(test_set)))

    test_loader = DataLoader(test_set, batch_size=1, shuffle=False,
                             num_workers=8, pin_memory=True, drop_last=True)
    
    resize_trans = ResizeTransform(img_size_s, img_size, 3).cuda()

    eval_dsc_def = AverageMeter()
    eval_dsc_raw = AverageMeter()
    eval_det = AverageMeter()
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            model.eval()
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            x_seg = data[2]
            y_seg = data[3]

            # interpolate to the 128x128x128 size
            x = F.interpolate(x, size=img_size_s, mode='trilinear', align_corners=True)
            y = F.interpolate(y, size=img_size_s, mode='trilinear', align_corners=True)

            x_def, flow = model(x, y)
            # resize back
            flow = resize_trans(flow)
        
            x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=46)
            x_seg_oh = torch.squeeze(x_seg_oh, 1)
            x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
            #x_segs = model.spatial_trans(x_seg.float(), flow.float())
            x_segs = []
            for i in range(46):
                def_seg = reg_model([x_seg_oh[:, i:i + 1, ...].float(), flow.float()])
                x_segs.append(def_seg)
            x_segs = torch.cat(x_segs, dim=1)
            def_out = torch.argmax(x_segs, dim=1, keepdim=True)
            del x_segs, x_seg_oh

            # def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            jac_det = utils.jacobian_determinant_vxm(flow.detach().cpu().numpy()[0, :, :, :, :])
            line = utils.dice_val_substruct(def_out.long(), y_seg.long(), stdy_idx)
            line = line +','+str(np.sum(jac_det <= 0)/np.prod(tar.shape))
            csv_writter(line, 'Quantitative_Results/' + csv_name)
            eval_det.update(np.sum(jac_det <= 0) / np.prod(tar.shape), x.size(0))
            print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(tar.shape)))
            dsc_trans = utils.dice_val(def_out.long(), y_seg.long(), 46)
            dsc_raw = utils.dice_val(x_seg.long(), y_seg.long(), 46)
            print('Trans dsc: {:.4f}, Raw dsc: {:.4f}'.format(dsc_trans.item(),dsc_raw.item()))
            eval_dsc_def.update(dsc_trans.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            stdy_idx += 1

        print('Deformed DSC: {:.3f} +- {:.3f}, Affine DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std))
        print('deformed det: {}, std: {}'.format(eval_det.avg, eval_det.std))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

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
