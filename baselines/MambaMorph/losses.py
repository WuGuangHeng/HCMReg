import torch
import torch.nn.functional as F
import numpy as np
import math
import pystrum.pynd.ndutils as nd

class Grad3d(torch.nn.Module):
    """
    N-D gradient loss.
    """

    def __init__(self, penalty='l1', loss_mult=None):
        super(Grad3d, self).__init__()
        self.penalty = penalty
        self.loss_mult = loss_mult

    def forward(self, y_pred, y_true):
        dy = torch.abs(y_pred[:, :, 1:, :, :] - y_pred[:, :, :-1, :, :])
        dx = torch.abs(y_pred[:, :, :, 1:, :] - y_pred[:, :, :, :-1, :])
        dz = torch.abs(y_pred[:, :, :, :, 1:] - y_pred[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dy = dy * dy
            dx = dx * dx
            dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
        grad = d / 3.0

        if self.loss_mult is not None:
            grad *= self.loss_mult
        return grad


class NCC_vxm(torch.nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win=None):
        super(NCC_vxm, self).__init__()
        self.win = win

    def forward(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = len(list(Ii.size())) - 2
        assert ndims in [
            1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [9] * ndims if self.win is None else self.win

        # compute filters
        sum_filt = torch.ones([1, 1, *win]).to("cuda")

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1)
            padding = (pad_no)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-5)

        return -torch.mean(cc)

'''
class JacboianLoss(torch.nn.Module):
    """
    Jacboian loss.
    """

    def __init__(self):
        super(JacboianLoss, self).__init__()

    def forward(self, flow):
        J = flow
        dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
        dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
        dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

        Jdet0 = dx[:, :, :, :, 0] * (
            dy[:, :, :, :, 1] * dz[:, :, :, :, 2] -
            dy[:, :, :, :, 2] * dz[:, :, :, :, 1]
        )
        Jdet1 = dx[:, :, :, :, 1] * (
            dy[:, :, :, :, 0] * dz[:, :, :, :, 2] -
            dy[:, :, :, :, 2] * dz[:, :, :, :, 0]
        )
        Jdet2 = dx[:, :, :, :, 2] * (
            dy[:, :, :, :, 0] * dz[:, :, :, :, 1] -
            dy[:, :, :, :, 1] * dz[:, :, :, :, 0]
        )

        Jdet = Jdet0 - Jdet1 + Jdet2
        neg_Jdet = -1.0 * Jdet
        selected_neg_Jdet = F.relu(neg_Jdet)

        return torch.mean(selected_neg_Jdet)


def JacboianDet(flow):  # jcobian loss
    J = flow
    dy = J[:, 1:, :-1, :-1, :] - J[:, :-1, :-1, :-1, :]
    dx = J[:, :-1, 1:, :-1, :] - J[:, :-1, :-1, :-1, :]
    dz = J[:, :-1, :-1, 1:, :] - J[:, :-1, :-1, :-1, :]

    Jdet0 = dx[:, :, :, :, 0] * (
        dy[:, :, :, :, 1] * dz[:, :, :, :, 2] -
        dy[:, :, :, :, 2] * dz[:, :, :, :, 1]
    )
    Jdet1 = dx[:, :, :, :, 1] * (
        dy[:, :, :, :, 0] * dz[:, :, :, :, 2] -
        dy[:, :, :, :, 2] * dz[:, :, :, :, 0]
    )
    Jdet2 = dx[:, :, :, :, 2] * (
        dy[:, :, :, :, 0] * dz[:, :, :, :, 1] -
        dy[:, :, :, :, 1] * dz[:, :, :, :, 0]
    )

    Jdet = Jdet0 - Jdet1 + Jdet2

    return Jdet


def neg_Jdet_loss(flow):
    neg_Jdet = -1.0 * JacboianDet(flow)
    selected_neg_Jdet = F.relu(neg_Jdet)

    return torch.mean(selected_neg_Jdet)
'''
class JacboianLoss(torch.nn.Module):
    """
    Jacboian loss.
    """

    def __init__(self):
        super(JacboianLoss, self).__init__()

    def jacobian_determinant_vxm(self,disp):
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
        disp = disp.squeeze(0).permute(1, 2, 3, 0).detach().cpu().numpy()
        # disp = disp.transpose(1, 2, 0)
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
    
    def forward(self, flow):
        
        Jdet = self.jacobian_determinant_vxm(flow)
        neg_Jdet = -1.0 * Jdet
        neg_Jdet =  torch.tensor(neg_Jdet)
        selected_neg_Jdet = F.relu(neg_Jdet)

        return torch.mean(selected_neg_Jdet)


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
    disp = disp.transpose(1, 2, 0)
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


def neg_Jdet_loss(flow):
    neg_Jdet = -1.0 * jacobian_determinant_vxm(flow)
    selected_neg_Jdet = F.relu(neg_Jdet)

    return torch.mean(selected_neg_Jdet)

class DiceLoss(torch.nn.Module):
    def __init__(self, dataset='LPBA40'):
        super(DiceLoss, self).__init__()
        self.eps = 1e-5
        self.dataset = dataset

    def dice_val_VOI(self, y_pred, y_true):
        if self.dataset == 'LPBA40':
            VOI_lbls = [i for i in range(1, 57)] # 56 labels
        elif self.dataset == 'Mindboggle':
            VOI_lbls = [i for i in range(1, 63)] # 62 labels
        elif self.dataset == 'OASIS':
            VOI_lbls = [i for i in range(1, 36)] # 35 labels
        elif self.dataset == 'IXI':
            VOI_lbls = [1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 
                    20, 21, 22, 23, 25, 26, 27, 28, 29, 30, 31, 32, 34, 36] # 30 labels
        elif self.dataset == 'SR_Reg':
            VOI_lbls = [i for i in range(1, 19)] # 18 labels

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
            idx += 1
        return np.mean(DSCs)

    def forward(self, input, target):
        
        dice = self.dice_val_VOI(input, target)

        return torch.as_tensor(1.0 - dice)