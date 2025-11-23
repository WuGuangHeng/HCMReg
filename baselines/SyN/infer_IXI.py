import os
import utils
from torch.utils.data import DataLoader
import torch.nn as nn
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms

import random
import ants
import nibabel as nib

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

def nib_load(file_name):
    if not os.path.exists(file_name):
        return np.array([1])

    proxy = nib.load(file_name)
    data = proxy.get_fdata()
    proxy.uncache()
    return data

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
   

    csv_name = 'SyN_IXI'
    dict = utils.process_label()
    if not os.path.exists('Quantitative_Results/'):
        os.makedirs('Quantitative_Results/')
    if os.path.exists('Quantitative_Results/'+csv_name+'.csv'):
        os.remove('Quantitative_Results/'+csv_name+'.csv')
    csv_writter(csv_name, 'Quantitative_Results/' + csv_name)
    line = ''
    for i in range(46):
        line = line + ',' + dict[i]
    csv_writter(line +','+'non_jec', 'Quantitative_Results/' + csv_name)

    img_size = (160, 192, 224)
   
    
    test_composed = transforms.Compose([
        trans.CenterCropBySize(img_size),
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
                             num_workers=1, pin_memory=True, drop_last=True)
    eval_dsc_def = AverageMeter()
    eval_dsc_raw = AverageMeter()
    eval_det = AverageMeter()
    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            x = data[0].squeeze(0).squeeze(0).detach().cpu().numpy()
            y = data[1].squeeze(0).squeeze(0).detach().cpu().numpy()
            x_seg = data[2]  # .squeeze(0).squeeze(0).detach().cpu().numpy()

            x_seg_oh = nn.functional.one_hot(x_seg.long(), num_classes=46)
            x_seg_oh = torch.squeeze(x_seg_oh, 1)
            x_seg_oh = x_seg_oh.permute(0, 4, 1, 2, 3).contiguous()
            x_seg_oh = x_seg_oh.squeeze(0).detach().cpu().numpy()

            y_seg = data[3].squeeze(0).squeeze(0).detach().cpu().numpy()

            x = ants.from_numpy(x)
            y = ants.from_numpy(y)

            y_ants = ants.from_numpy(y_seg.astype(np.float32))

            reg12 = ants.registration(y, x, 'SyNOnly', reg_iterations=(160, 80, 40), syn_metric='meansquares')
            def_segs = []
            for i in range(x_seg_oh.shape[0]):
                x_chan = ants.from_numpy(x_seg_oh[i].astype(np.float32))
                def_seg = ants.apply_transforms(fixed=y_ants,
                                                moving=x_chan,
                                                transformlist=reg12['fwdtransforms'], )
                # whichtoinvert=[True, False, True, False]
                def_segs.append(def_seg.numpy()[None, ...])
            def_segs = np.concatenate(def_segs, axis=0)
            def_seg = np.argmax(def_segs, axis=0)
            flow = np.array(nib_load(reg12['fwdtransforms'][0]), dtype='float32', order='C')
            flow = flow[:,:,:,0,:].transpose(3, 0, 1, 2)
            def_seg = torch.from_numpy(def_seg[None, None, ...])
            y_seg = torch.from_numpy(y_seg[None, None, ...])
            dsc_raw = utils.dice_val(x_seg.long(), y_seg.long(), 46)
            dsc_trans = utils.dice_val(def_seg.long(), y_seg.long(), 46)
            eval_dsc_raw.update(dsc_raw.item(), 1)
            eval_dsc_def.update(dsc_trans.item(), 1)
            jac_det = utils.jacobian_determinant_vxm(flow)
            line = utils.dice_val_substruct(def_seg.long(), y_seg.long(), stdy_idx)
            line = line + ',' + str(np.sum(jac_det <= 0) / np.prod(y_seg.shape))
            print('det < 0: {}'.format(np.sum(jac_det <= 0) / np.prod(y_seg.shape)))
            csv_writter(line, 'Quantitative_Results/' + csv_name)
            print('DSC: {:.4f}'.format(dsc_trans.item()))
            print('DSC_raw: {:.4f}'.format(dsc_raw.item()))
            stdy_idx += 1
        print('Deformed DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg, eval_dsc_def.std))
        print('Raw DSC: {:.3f} +- {:.3f}'.format(eval_dsc_raw.avg, eval_dsc_raw.std))

def csv_writter(line, name):
    with open(name+'.csv', 'a') as file:
        file.write(line)
        file.write('\n')

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 2
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
