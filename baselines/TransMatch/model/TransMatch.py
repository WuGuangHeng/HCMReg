import torch.nn as nn
import torch.nn.functional as F

class Conv3dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        relu = nn.LeakyReLU(inplace=True)
        if not use_batchnorm:
            nm = nn.InstanceNorm3d(out_channels)
        else:
            nm = nn.BatchNorm3d(out_channels)

        super(Conv3dReLU, self).__init__(conv, nm, relu)

# LWSA
'''
LWSA module

A partial code was retrieved from:
https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration

Swin-Transformer code was retrieved from:
https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation

Original Swin-Transformer paper:
Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021).
Swin transformer: Hierarchical vision transformer using shifted windows.
arXiv preprint arXiv:2103.14030.
'''

import model.basic_LWSA as basic
import torch.nn as nn
'''
Zeyuan Chen
Shandong Normal University
snowbplus@gmail.com


Thanks to
Junyu Chen
Johns Hopkins Unversity
jchen245@jhmi.edu
'''

import ml_collections
'''
********************************************************
                   Swin Transformer
********************************************************
if_transskip (bool): Enable skip connections from Transformer Blocks
if_convskip (bool): Enable skip connections from Convolutional Blocks
patch_size (int | tuple(int)): Patch size. Default: 4
in_chans (int): Number of input image channels. Default: 2 (for moving and fixed images)
embed_dim (int): Patch embedding dimension. Default: 96
depths (tuple(int)): Depth of each Swin Transformer layer.
num_heads (tuple(int)): Number of attention heads in different layers.
window_size (tuple(int)): Image size should be divisible by window size, 
                     e.g., if image has a size of (160, 192, 224), then the window size can be (5, 6, 7)
mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
pat_merg_rf (int): Embed_dim reduction factor in patch merging, e.g., N*C->N/4*C if set to four. Default: 4. 
qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
drop_rate (float): Dropout rate. Default: 0
drop_path_rate (float): Stochastic depth rate. Default: 0.1
ape (bool): Enable learnable position embedding. Default: False
spe (bool): Enable sinusoidal position embedding. Default: False
patch_norm (bool): If True, add normalization after patch embedding. Default: True
use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False 
                       (Carried over from Swin Transformer, it is not needed)
out_indices (tuple(int)): Indices of Transformer blocks to output features. Default: (0, 1, 2, 3)
reg_head_chan (int): Number of channels in the registration head (i.e., the final convolutional layer) 
img_size (int | tuple(int)): Input image size, e.g., (160, 192, 224)
'''
def get_TransMatch_LPBA40_config():
    config = ml_collections.ConfigDict()
    config.if_transskip = True
    config.if_convskip = True
    config.patch_size = 4
    config.in_chans = 1
    config.embed_dim = 96
    config.depths = (2, 2, 4, 2)
    config.num_heads = (4, 4, 8, 8)
    # config.num_heads = (4, 4, 8, 8)
    config.window_size = (5, 6, 5)
    config.mlp_ratio = 4
    config.pat_merg_rf = 4
    config.qkv_bias = False
    config.drop_rate = 0
    config.drop_path_rate = 0.3
    config.ape = False
    config.spe = False
    config.patch_norm = True
    config.use_checkpoint = False
    config.out_indices = (0, 1, 2, 3)
    config.reg_head_chan = 16
    config.img_size = (160, 192, 160)
    return config


class LWSA(nn.Module):
    def __init__(self, config):
        super(LWSA, self).__init__()
        if_convskip = config.if_convskip
        self.if_convskip = if_convskip
        if_transskip = config.if_transskip
        self.if_transskip = if_transskip
        embed_dim = config.embed_dim
        self.transformer = basic.SwinTransformer(patch_size=config.patch_size,
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf
                                           )
        self.c1 = Conv3dReLU(1, embed_dim//2, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(1, config.reg_head_chan, 3, 1, use_batchnorm=False)

        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)

    def forward(self, x):
        # print('The shape of x(Input of transformer function):', x.shape)
        source = x[:, 0:1, :, :, :]
        #print('The shape of source:', source.shape)
        if self.if_convskip:
            x_s0 = x.clone()  # 用于concat AB的直接卷积的input
            # print('The shape of x_s0(X.clone):', x_s0.shape)
            x_s1 = self.avg_pool(x)  # 用于concat AB后下采样1/2后的卷积的input
            # print('The shape of x_s1:', x_s1.shape)
            f4 = self.c1(x_s1)  # 下采样后的卷积
            # print('The shape of f4', f4.shape)
            f5 = self.c2(x_s0)  # 原始图像的卷积
            # print('The shape of f5:', f5.shape)
        else:
            f4 = None
            f5 = None

        out_feats = self.transformer(x)

        if self.if_transskip:
            f1 = out_feats[-2]
            f2 = out_feats[-3]
            f3 = out_feats[-4]
        else:
            f1 = None
            f2 = None
            f3 = None
        return f3, f2, f1, out_feats[-1]


CONFIGS = {
    'TransMatch_LPBA40': get_TransMatch_LPBA40_config()
}

# LWCA

'''
LWCA module

A partial code was retrieved from:
https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration

Swin-Transformer code was retrieved from:
https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation

Original Swin-Transformer paper:
Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021).
Swin transformer: Hierarchical vision transformer using shifted windows.
arXiv preprint arXiv:2103.14030.
'''

import model.basic_LWCA as basic_lwca
import torch.nn as nn

class LWCA(nn.Module):
    def __init__(self, config, dim_diy):
        super(LWCA, self).__init__()
        self.transformer = basic_lwca.SwinTransformer(patch_size=config.patch_size,
                                           in_chans=config.in_chans,
                                           embed_dim=config.embed_dim,
                                           depths=config.depths,
                                           num_heads=config.num_heads,
                                           window_size=config.window_size,
                                           mlp_ratio=config.mlp_ratio,
                                           qkv_bias=config.qkv_bias,
                                           drop_rate=config.drop_rate,
                                           drop_path_rate=config.drop_path_rate,
                                           ape=config.ape,
                                           spe=config.spe,
                                           patch_norm=config.patch_norm,
                                           use_checkpoint=config.use_checkpoint,
                                           out_indices=config.out_indices,
                                           pat_merg_rf=config.pat_merg_rf,
                                           dim_diy=dim_diy
                                           )

    def forward(self, x, y):
        moving_fea_cross = self.transformer(x, y)
        return moving_fea_cross


# Decoder
import torch
from torch import nn
from torch.distributions.normal import Normal


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv3dReLU(
            in_channels + skip_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv3dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv3 = Conv3dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

    def forward(self, x, skip=None, skip2=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        if skip2 is not None:
            x = torch.cat([x, skip2], dim=1)
            x = self.conv1(x)
        if skip2 is None:
            x = self.conv3(x)
        x = self.conv2(x)
        return x

class RegistrationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv3d = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        conv3d.weight = nn.Parameter(Normal(0, 1e-5).sample(conv3d.weight.shape))
        conv3d.bias = nn.Parameter(torch.zeros(conv3d.bias.shape))
        super().__init__(conv3d)

class SpatialTransformer(nn.Module):
    """
    N-D Spatial Transformer
    Obtained from https://github.com/voxelmorph/voxelmorph
    """

    def __init__(self, size, mode="bilinear"):
        super().__init__()

        self.mode = mode

        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)

        # registering the grid as a buffer cleanly moves it to the GPU, but it also
        # adds it to the state dict. this is annoying since everything in the state dict
        # is included when saving weights to disk, so the model files are way bigger
        # than they need to be. so far, there does not appear to be an elegant solution.
        # see: https://discuss.pytorch.org/t/how-to-register-buffer-without-polluting-state-dict
        self.register_buffer("grid", grid)

    def forward(self, src, flow):
        # new locations
        new_locs = self.grid + flow
        shape = flow.shape[2:]

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        # move channels dim to last position
        # also not sure why, but the channels need to be reversed
        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, align_corners=False, mode=self.mode)
    
# TransMatch
import torch.nn as nn
import torch

class TransMatch(nn.Module):
    def __init__(self, args=None):
        super(TransMatch, self).__init__()

        self.avg_pool = nn.AvgPool3d(3, stride=2, padding=1)
        self.c1 = Conv3dReLU(2, 48, 3, 1, use_batchnorm=False)
        self.c2 = Conv3dReLU(2, 16, 3, 1, use_batchnorm=False)

        config2 = get_TransMatch_LPBA40_config()
        self.moving_lwsa = LWSA(config2)
        self.fixed_lwsa = LWSA(config2)

        self.lwca1 = LWCA(config2, dim_diy=96)
        self.lwca2 = LWCA(config2, dim_diy=192)
        self.lwca3 = LWCA(config2, dim_diy=384)
        self.lwca4 = LWCA(config2, dim_diy=768)

        self.up0 = DecoderBlock(768, 384, skip_channels=384, use_batchnorm=False)
        self.up1 = DecoderBlock(384, 192, skip_channels=192, use_batchnorm=False)
        self.up2 = DecoderBlock(192, 96, skip_channels=96, use_batchnorm=False)
        self.up3 = DecoderBlock(96, 48, skip_channels=48, use_batchnorm=False)
        self.up4 = DecoderBlock(48, 16, skip_channels=16, use_batchnorm=False)
        self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)

        self.reg_head = RegistrationHead(
            in_channels=48,
            out_channels=3,
            kernel_size=3,
        )
        self.spatial_trans = SpatialTransformer(config2.img_size)

    def forward(self, moving_Input, fixed_Input):

        input_fusion = torch.cat((moving_Input, fixed_Input), dim=1)

        x_s1 = self.avg_pool(input_fusion)  # 用于concat AB后下采样1/2后的卷积的input

        f4 = self.c1(x_s1)  # 下采样后的卷积
        f5 = self.c2(input_fusion)  # 原始图像的卷积

        B, _, _, _, _ = moving_Input.shape  # Batch, channel, height, width, depth

        moving_fea_4, moving_fea_8, moving_fea_16, moving_fea_32 = self.moving_lwsa(moving_Input)
        fixed_fea_4, fixed_fea_8, fixed_fea_16, fixed_fea_32 = self.moving_lwsa(fixed_Input)

        moving_fea_4_cross = self.lwca1(moving_fea_4, fixed_fea_4)
        moving_fea_8_cross = self.lwca2(moving_fea_8, fixed_fea_8)
        moving_fea_16_cross = self.lwca3(moving_fea_16, fixed_fea_16)
        moving_fea_32_cross = self.lwca4(moving_fea_32, fixed_fea_32)

        fixed_fea_4_cross = self.lwca1(fixed_fea_4, moving_fea_4)
        fixed_fea_8_cross = self.lwca2(fixed_fea_8, moving_fea_8)
        fixed_fea_16_cross = self.lwca3(fixed_fea_16, moving_fea_16)
        fixed_fea_32_cross = self.lwca4(fixed_fea_32, moving_fea_32)


        x = self.up0(moving_fea_32_cross, moving_fea_16_cross, fixed_fea_16_cross)
        x = self.up1(x, moving_fea_8_cross, fixed_fea_8_cross)
        x = self.up2(x, moving_fea_4_cross, fixed_fea_4_cross)
        x = self.up3(x, f4)
        x = self.up(x)
        flow = self.reg_head(x)
        warped_moving = self.spatial_trans(moving_Input, flow)
        return warped_moving, flow

if __name__ == "__main__":
    inshape = (1, 1, 160, 192, 160)
    model = TransMatch(None)
    # print(str(model))
    A = torch.ones(inshape)
    B = torch.ones(inshape)
    _, flow = model(A, B)
   
    import thop
    macs, params = thop.profile(model, inputs=(A, B))
    flops, params = thop.clever_format([macs*2, params], "%.3f")
    print("FLOPs: {}, Params: {}".format(flops, params))
