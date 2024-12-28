import math
import numpy as np
import torch.nn.functional as F
import torch
import sys
from torch import nn
import pystrum.pynd.ndutils as nd


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


class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    """

    def __init__(self, size, mode='bilinear'):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor).cuda()

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer('grid', grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * \
                (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=self.mode,align_corners=True)

class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps=7):
        super().__init__()

        assert nsteps >= 0, "nsteps should be >= 0, found: %d" % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2**self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class register_model(nn.Module):
    def __init__(self, img_size=(64, 256, 256), mode='bilinear'):
        super(register_model, self).__init__()
        self.spatial_trans = SpatialTransformer(img_size, mode)

    def forward(self, x):
        img = x[0].cuda()
        flow = x[1].cuda()
        out = self.spatial_trans(img, flow)
        return out

# For IXI dataset
import re
def process_label():
    #process labeling information for FreeSurfer
    seg_table = [0, 2, 3, 4, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17, 18, 24, 26,
                          28, 30, 31, 41, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 54, 58, 60, 62,
                          63, 72, 77, 80, 85, 251, 252, 253, 254, 255]


    file1 = open('/home/wgh/experiments/exps/Proposed_methods/IXI/label_info.txt', 'r')
    Lines = file1.readlines()
    dict = {}
    seg_i = 0
    seg_look_up = []
    for seg_label in seg_table:
        for line in Lines:
            line = re.sub(' +', ' ',line).split(' ')
            try:
                int(line[0])
            except:
                continue
            if int(line[0]) == seg_label:
                seg_look_up.append([seg_i, int(line[0]), line[1]])
                dict[seg_i] = line[1]
        seg_i += 1
    return dict
# For IXI dataset
def dice_val(y_pred, y_true, num_clus):
    y_pred = nn.functional.one_hot(y_pred, num_classes=num_clus)
    y_pred = torch.squeeze(y_pred, 1)
    y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
    y_true = nn.functional.one_hot(y_true, num_classes=num_clus)
    y_true = torch.squeeze(y_true, 1)
    y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    intersection = y_pred * y_true
    intersection = intersection.sum(dim=[2, 3, 4])
    union = y_pred.sum(dim=[2, 3, 4]) + y_true.sum(dim=[2, 3, 4])
    dsc = (2.*intersection) / (union + 1e-5)
    return torch.mean(torch.mean(dsc, dim=1))

def dice_val_VOI(y_pred, y_true, dataset='LPBA40'):
    
    if dataset == 'LPBA40':
        VOI_lbls = [i for i in range(1, 57)] # 56 labels
        structs = ['frontal', 'parietal','occipital', 'temporal', 'cingulate', 'putamen', 'hippocampus']
        structures = [[1,2,3,4,5,6], [17,18], [25,26,27,28,29,30], 
                      [33,34,35,36,37,38], [47,48], [51,52], [53,54]]
    
        struct_DSCs = {key: [] for key in structs}
        
    elif dataset == 'MindBoggle':
        VOI_lbls = [i for i in range(1, 63)] # 62 labels
        structs = ['frontal', 'parietal','occipital', 'temporal', 'cingulate']
        structures = [[2, 25, 26, 33, 56, 57], [6, 27, 37, 58], [9, 40], 
                      [7, 13, 28, 30, 38, 44, 59, 61], [1, 8, 21, 24, 32, 39, 52, 55]]
        struct_DSCs = {key: [] for key in structs}

    elif dataset == 'OASIS':
        VOI_lbls = [i for i in range(1, 36)] # 35 labels
        # place holder
        structs = [i for i in range(1, 7)]
        structures = [[0]]*7
        struct_DSCs = {key: [] for key in structs}
        
    elif dataset == 'IXI':
        VOI_lbls = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 
                    20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36] # 30 labels 
        # place holder
        structs = [i for i in range(1, 7)]
        structures = [[0]]*7
        struct_DSCs = {key: [] for key in structs}


    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    DSCs = np.zeros((len(VOI_lbls), 1))
    
    idx = 0
    for i in VOI_lbls:
        pred_i = pred == i
        true_i = true == i
        intersection = pred_i * true_i
        intersection = np.sum(intersection)
        union = np.sum(pred_i) + np.sum(true_i)
        dsc = (2.*intersection) / (union + 1e-5)
        DSCs[idx] = dsc
        # pick structures
        for j in range(len(structures)):
            if i in structures[j]:
                struct_DSCs[structs[j]].append(dsc)
                break
    
        idx += 1
    # print(struct_DSCs.values())
    return np.mean(DSCs), [np.mean(struct_DSCs[i]) for i in structs]


# For IXI dataset
from medpy.metric.binary import hd95

def dice_val_substruct(y_pred, y_true, std_idx):
    with torch.no_grad():
        y_pred = nn.functional.one_hot(y_pred, num_classes=46)
        y_pred = torch.squeeze(y_pred, 1)
        y_pred = y_pred.permute(0, 4, 1, 2, 3).contiguous()
        y_true = nn.functional.one_hot(y_true, num_classes=46)
        y_true = torch.squeeze(y_true, 1)
        y_true = y_true.permute(0, 4, 1, 2, 3).contiguous()
    y_pred = y_pred.detach().cpu().numpy()
    y_true = y_true.detach().cpu().numpy()

    line = 'p_{}'.format(std_idx)
    for i in range(46):
        pred_clus = y_pred[0, i, ...]
        true_clus = y_true[0, i, ...]
        # print(np.unique(pred_clus), np.unique(true_clus))

        intersection = pred_clus * true_clus
        intersection = intersection.sum()
        union = pred_clus.sum() + true_clus.sum()
        dsc = (2.*intersection) / (union + 1e-5)
        
        if np.count_nonzero(pred_clus) == 0 or np.count_nonzero(true_clus) == 0:
            hd = np.nan # cannot set hd to 0      
        else:
            hd = hd95(true_clus==1, pred_clus==1) 
        
        line = line+','+str(dsc)+'-'+str(hd)
    return line


def jacobian_determinant_vxm(disp):
    """
    jacobian determinant of a displacement field.
    NB: to compute the spatial gradients, we use np.gradient.
    Parameters:
        disp: 2D or 3D displacement field of size [*vol_shape, nb_dims],
              where vol_shape is of len nb_dims
    Returns:
        jacobian determinant (scalar)
    """

    # check inputs
    disp = disp.transpose(1, 2, 3, 0)
    volshape = disp.shape[:-1]
    nb_dims = len(volshape)
    assert len(volshape) in (2, 3), 'flow has to be 2D or 3D'

    # compute grid
    grid_lst = nd.volsize2ndgrid(volshape)
    grid = np.stack(grid_lst, len(volshape))

    # compute gradients
    J = np.gradient(disp + grid)

    # 3D glow
    if nb_dims == 3:
        dx = J[0]
        dy = J[1]
        dz = J[2]

        # compute jacobian components
        Jdet0 = dx[..., 0] * (dy[..., 1] * dz[..., 2] -
                              dy[..., 2] * dz[..., 1])
        Jdet1 = dx[..., 1] * (dy[..., 0] * dz[..., 2] -
                              dy[..., 2] * dz[..., 0])
        Jdet2 = dx[..., 2] * (dy[..., 0] * dz[..., 1] -
                              dy[..., 1] * dz[..., 0])

        return Jdet0 - Jdet1 + Jdet2

    else:  # must be 2

        dfdx = J[0]
        dfdy = J[1]

        return dfdx[..., 0] * dfdy[..., 1] - dfdy[..., 0] * dfdx[..., 1]
