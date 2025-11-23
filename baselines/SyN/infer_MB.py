import os
import utils
from torch.utils.data import DataLoader
import torch.nn as nn
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from medpy.metric.binary import hd95
import random
import ants
import nibabel as nib
import time
import csv

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

def compute_hausdorff_distance(pred, gt, dataset='LPBA40'):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    if dataset == 'LPBA40':
        cls_lst = [i for i in range(1, 57)]
    elif dataset == 'MindBoggle':
        cls_lst = [i for i in range(1, 63)]
    elif dataset == 'OASIS':
        cls_lst = [i for i in range(1, 36)]
    hd_lst = []
    for cls in cls_lst:
        if np.count_nonzero(pred == cls) == 0 or np.count_nonzero(gt == cls) == 0:
            hd = np.nan
        else:
            hd = hd95(gt == cls, pred == cls)
        hd_lst.append(hd)
    hd_lst = [i for i in hd_lst if not np.isnan(i)]
    return np.mean(hd_lst)

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

def csv_writer(file_name, header, data):
    with open(file_name, 'a') as file:
        # check if has header
        if header is not None:
            writer = csv.writer(file)
            writer.writerow(header)
        else:
            writer = csv.writer(file)
        # write data
            writer.writerow(data)



def main():
    pref =""
    dataset = 'OASIS'
    val_dir = pref+dataset+'/test'
    label_dir = pref+dataset+'/label'
    atlas_dir = pref+dataset+'/fixed.nii.gz'
   
    img_size = (160, 192, 160)

    if dataset == 'OASIS':
        img_size = (224, 192, 160)
    
    test_composed = transforms.Compose([
        # trans.CenterCropBySize(img_size),
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
    
    result_dir = 'results/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    f = open(os.path.join(result_dir, dataset + '_SyN' + ".txt"), "w")

    # boxplot file
    boxplot_file = os.path.join(result_dir, dataset + '_SyN_BoxPlot' + '.csv')
    structs = ['frontal', 'parietal','occipital', 'temporal', 'cingulate']
    # csv_writer(boxplot_file, structs, [])

    eval_dsc_def = AverageMeter()
    eval_dsc_raw = AverageMeter()
    eval_det = AverageMeter()
    eval_hd_def = AverageMeter()
    eval_hd_raw = AverageMeter()
    TIME = []

    with torch.no_grad():
        stdy_idx = 0
        for data in test_loader:
            x = data[0].squeeze(0).squeeze(0).detach().cpu().numpy()
            y = data[1].squeeze(0).squeeze(0).detach().cpu().numpy()
            x_seg = data[2].squeeze(0).squeeze(0).detach().cpu().numpy()
            y_seg = data[3].squeeze(0).squeeze(0).detach().cpu().numpy()

            x = ants.from_numpy(x)
            y = ants.from_numpy(y)

            x_ants = ants.from_numpy(x_seg.astype(np.float32))
            y_ants = ants.from_numpy(y_seg.astype(np.float32))

            start = time.time()
            reg12 = ants.registration(y, x, 'SyNOnly', reg_iterations=(160, 80, 40), syn_metric='meansquares')
            def_seg = ants.apply_transforms(fixed=y_ants,
                                            moving=x_ants,
                                            transformlist=reg12['fwdtransforms'],
                                            interpolator='nearestNeighbor',)
                                            #whichtoinvert=[True, False, True, False]
            TIME.append(time.time()-start)

            flow = np.array(nib_load(reg12['fwdtransforms'][0]), dtype='float32', order='C')
            flow = flow[:,:,:,0,:].transpose(3, 0, 1, 2)
            def_seg = torch.from_numpy(def_seg[None, None, ...])
            x_seg = torch.from_numpy(x_seg[None, None, ...])
            y_seg = torch.from_numpy(y_seg[None, None, ...])
            # DSC
            dsc_raw, _ = utils.dice_val_VOI(x_seg.long(), y_seg.long(), dataset=dataset)
            dsc_trans, values = utils.dice_val_VOI(def_seg.long(), y_seg.long(), dataset=dataset)
            # boxplot
            # print(values)
            # csv_writer(boxplot_file, None, values)

            eval_dsc_raw.update(dsc_raw.item(), 1)
            eval_dsc_def.update(dsc_trans.item(), 1)
            print('{}: Deformed dsc: {:.4f}, Raw dsc: {:.4f}'.format(
                stdy_idx, dsc_trans.item(), dsc_raw.item()), file=f)
            # hausdorff distance
            hd_def = compute_hausdorff_distance(def_seg.detach().cpu().numpy()[0, 0, :, :, :], 
                                            y_seg.detach().cpu().numpy()[0, 0, :, :, :], dataset=dataset)
            hd_raw = compute_hausdorff_distance(x_seg.detach().cpu().numpy()[0, 0, :, :, :], y_seg.detach().cpu().numpy()[0, 0, :, :, :], dataset=dataset)
            print('{}: Deformed hd: {:.4f}, Raw hd: {:.4f}'.format(
                stdy_idx, hd_def, hd_raw), file=f)
            eval_hd_def.update(hd_def, 1)
            eval_hd_raw.update(hd_raw, 1)

            print('Processing {} / {}'.format(stdy_idx, len(test_set)))
            stdy_idx += 1

            jac_det = utils.jacobian_determinant_vxm(flow)
            eval_det.update(np.sum(jac_det <= 0) /
                            np.prod(img_size), 1)

        print('Deformed DSC: {:.3f} +- {:.3f}, Raw DSC: {:.3f} +- {:.3f}'.format(eval_dsc_def.avg,
                                                                                    eval_dsc_def.std,
                                                                                    eval_dsc_raw.avg,
                                                                                    eval_dsc_raw.std),
                                                                                    file=f)
        print('Deformed HD: {:.3f} +- {:.3f}, Raw HD: {:.3f} +- {:.3f}'.format(eval_hd_def.avg,
                                                                                eval_hd_def.std,
                                                                                eval_hd_raw.avg,
                                                                                eval_hd_raw.std),
                                                                                file=f)
        print('Deformed Det(%): {:.6f} +- {:.6f}'.format(eval_det.avg*100, eval_det.std*100), file=f)
        print('Average time: {:.3f}'.format(np.mean(TIME)),file=f)    

    f.close()

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
