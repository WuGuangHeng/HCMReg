import os
import utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms

from natsort import natsorted
from model.HCMReg import HCMReg
import random
from medpy.metric.binary import hd95
import time
from thop import profile
from thop import clever_format

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

def compute_hausdorff_distance(pred, gt, dataset='LPBA40'):
    # 需要计算的标签类别，不包括背景和图像中不存在的区域
    if dataset == 'LPBA40':
        cls_lst = [i for i in range(1, 57)]
    elif dataset == 'Mindboggle':
        cls_lst = [i for i in range(1, 63)]
    hd_lst = []
    for cls in cls_lst:
        hd = hd95(gt == cls, pred == cls)
        hd_lst.append(hd)
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
    import csv
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
    dataset = 'LPBA40'
    val_dir = dataset+'/test'
    label_dir = dataset+'/label'
    atlas_dir = dataset+'/fixed.nii.gz'
    weights = [1, 1, 0, 0]  # loss weights
    lr = 0.001
    model_folder = '{}-HCMReg_ncc_{}_reg_{}_jac_{}_dice_{}_lr_{}/'.format(dataset,
                                                                weights[0], weights[1], weights[2], weights[3], lr)
    model_idx = -1
    model_dir = 'experiments/' + model_folder

    if dataset == 'LPBA40':
        img_size = (160, 192, 160)
    elif dataset == 'OASIS':
        img_size = (224, 192, 160)
    elif dataset =='IXI':
        img_size = (160, 192, 224)
    else: # MindBoggle
        img_size = (160, 192, 160)

    model = HCMReg(inshape=img_size).cuda()

    best_model = torch.load(
        model_dir + natsorted(os.listdir(model_dir))[model_idx],map_location='cuda')['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model, strict=True)
    model.cuda()
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()
    test_composed = transforms.Compose([
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
                             num_workers=16, pin_memory=True, drop_last=True)
    
    result_dir = 'results/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    f = open(os.path.join(result_dir, model_folder[:-1] + ".txt"), "w")

    eval_dsc_def = AverageMeter()
    eval_dsc_raw = AverageMeter()
    eval_det = AverageMeter()
    eval_hd_def = AverageMeter()
    eval_hd_raw = AverageMeter()
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
            TIME.append(time.time()-start)

            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])
            tar = y.detach().cpu().numpy()[0, 0, :, :, :]
            # jacobian determinant
            jac_det = utils.jacobian_determinant_vxm(
                flow.detach().cpu().numpy()[0, :, :, :, :])
            eval_det.update(np.sum(jac_det <= 0) /
                            np.prod(tar.shape), x.size(0))
            # dice score
            dsc_def, values = utils.dice_val_VOI(
                def_out.long(), y_seg.long(), dataset=dataset)
            dsc_raw, _ = utils.dice_val_VOI(
                x_seg.long(), y_seg.long(), dataset=dataset)

            print('{}: Deformed dsc: {:.4f}, Raw dsc: {:.4f}'.format(
                stdy_idx, dsc_def.item(), dsc_raw.item()), file=f)
            eval_dsc_def.update(dsc_def.item(), x.size(0))
            eval_dsc_raw.update(dsc_raw.item(), x.size(0))
            # hausdorff distance
            hd_def = compute_hausdorff_distance(def_out.detach().cpu().numpy()[0, 0, :, :, :], 
                                            y_seg.detach().cpu().numpy()[0, 0, :, :, :], dataset=dataset)
            hd_raw = compute_hausdorff_distance(x_seg.detach().cpu().numpy()[0, 0, :, :, :],
                                            y_seg.detach().cpu().numpy()[0, 0, :, :, :], dataset=dataset)
            print('{}: Deformed hd: {:.4f}, Raw hd: {:.4f}'.format(
                stdy_idx, hd_def, hd_raw), file=f)
            eval_hd_def.update(hd_def, x.size(0))
            eval_hd_raw.update(hd_raw, x.size(0))
            print('Processing {} / {}'.format(stdy_idx, len(test_set)))
            stdy_idx += 1
        
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
        print('Average time: {:.3f}'.format(np.mean(TIME)), file=f)
    
    #计算模型参数量和计算量
    x1 = torch.randn(1, 1, img_size[0], img_size[1], img_size[2]).cuda()
    x2 = torch.randn(1, 1, img_size[0], img_size[1], img_size[2]).cuda()
    macs, params = profile(model, inputs=(x1, x2))
    flops, params = clever_format([macs*2, params], "%.3f")
    print('FLOPs: {}, Params: {}'.format(flops,params), file=f)

    f.close()

if __name__ == '__main__':
    '''
    GPU configuration
    '''
    GPU_iden = 1
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
