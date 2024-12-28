import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

class ConvInsBlock(nn.Module):
    """
    Specific convolutional block followed by leakyrelu for unet.
    """

    def __init__(
        self, in_channels, out_channels, kernal_size=3, stride=1, padding=1, alpha=0.1
    ):
        super().__init__()

        self.main = nn.Conv3d(in_channels, out_channels,
                              kernal_size, stride, padding)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.activation = nn.LeakyReLU(alpha)

    def forward(self, x):
        out = self.main(x)
        out = self.norm(out)
        out = self.activation(out)
        return out

class ELK_block(nn.Module):
    """Efficient Local Kernel (ELK) block."""

    def __init__(self, channel_num: int):
        super().__init__()
        
        self.Conv_1 = nn.Conv3d(channel_num, channel_num//2, kernel_size=(3,3,3), stride=1, padding='same')
        self.Conv_2 = nn.Conv3d(channel_num, channel_num//2, kernel_size=(5,1,1), stride=1, padding='same')
        self.Conv_3 = nn.Conv3d(channel_num, channel_num//2, kernel_size=(1,5,1), stride=1, padding='same')
        self.Conv_4 = nn.Conv3d(channel_num, channel_num//2, kernel_size=(1,1,5), stride=1, padding='same')
        self.Conv = nn.Conv3d(channel_num*2, channel_num, kernel_size=1, stride=1, padding='same')
        
    def forward(self, x_in):

        x_1 = self.Conv_1(x_in)
        x_2 = self.Conv_2(x_in)
        x_3 = self.Conv_3(x_in)
        x_4 = self.Conv_4(x_in)
        
        x = torch.cat([x_1, x_2, x_3, x_4], dim=1)
        x = self.Conv(x)
        x = x + x_in
        
        return x
    
class ConvMambaLayer(nn.Module):
    def __init__(self, dim, d_state = 16, d_conv = 4, expand = 2, ELK=True):
        super().__init__()
        self.dim = dim
        self.div_dim = dim // 2
        self.norm = nn.LayerNorm(self.div_dim)
        self.mamba = Mamba(
                d_model=self.div_dim, # Model dimension d_model
                d_state=d_state,  # SSM state expansion factor
                d_conv=d_conv,    # Local convolution width
                expand=expand,    # Block expansion factor
                bimamba_type='v2',
        )

        if ELK:
          self.conv_path = nn.Sequential(
                # nn.Conv3d(in_channels=self.div_dim, out_channels=self.div_dim, kernel_size=1, stride=1, padding=0),
                # nn.InstanceNorm3d(self.div_dim),
                # nn.LeakyReLU(0.1),
                ConvInsBlock(self.div_dim, self.div_dim, kernal_size=1, stride=1, padding=0),
                ELK_block(self.div_dim),
                nn.InstanceNorm3d(self.div_dim),
                nn.LeakyReLU(0.1),
                # nn.Conv3d(in_channels=self.div_dim, out_channels=self.div_dim, kernel_size=1, stride=1, padding=0),
                # nn.InstanceNorm3d(self.div_dim),
                # nn.LeakyReLU(0.1),
                ConvInsBlock(self.div_dim, self.div_dim, kernal_size=1, stride=1, padding=0),
            )  
        else: # if not ELK, then use the default conv path(3x3, 3x3, 1x1)
            # self.conv_path = None
            # raise ValueError('Please specify the type of block to use, either ELK or DLK')
            self.conv_path = nn.Sequential(
                # nn.Conv3d(in_channels=self.div_dim, out_channels=self.div_dim, kernel_size=3, stride=1, padding=1),
                # nn.InstanceNorm3d(self.div_dim),
                # nn.LeakyReLU(0.1),
                ConvInsBlock(self.div_dim, self.div_dim, kernal_size=3, stride=1, padding=1),
                # nn.Conv3d(in_channels=self.div_dim, out_channels=self.div_dim, kernel_size=3, stride=1, padding=1),
                # nn.InstanceNorm3d(self.div_dim),
                # nn.LeakyReLU(0.1),
                ConvInsBlock(self.div_dim, self.div_dim, kernal_size=3, stride=1, padding=1),
                # nn.Conv3d(in_channels=self.div_dim, out_channels=self.div_dim, kernel_size=1, stride=1, padding=0),
                # nn.InstanceNorm3d(self.div_dim),
                # nn.LeakyReLU(0.1),
                ConvInsBlock(self.div_dim, self.div_dim, kernal_size=1, stride=1, padding=0),
            )


        # Feature Fusion
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv_atten = nn.Sequential(
            nn.Conv3d(dim, dim, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.conv1 = nn.Conv3d(dim, dim, kernel_size=1, bias=False)

        self.conv2 = nn.Conv3d(dim, 1, kernel_size=1, stride=1, bias=True)
        self.nonlin = nn.Sigmoid()

    def channel_shuffle(self, x, groups: int):

        batch_size, num_channels, depth, height, width = x.size()
        channels_per_group = num_channels // groups

        # reshape
        # [batch_size, num_channels, depth, height, width] -> [batch_size, groups, channels_per_group, depth, height, width]
        x = x.view(batch_size, groups, channels_per_group, depth, height, width)

        x = torch.transpose(x, 1, 2).contiguous()

        # flatten
        x = x.view(batch_size, -1, depth, height, width)

        return x
   
    def forward(self, x):
       
        x_conv, x_mamba = x.chunk(2, dim=1)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_mamba = x_mamba.reshape(B, C//2, n_tokens).transpose(-1, -2)
        x_mamba = self.norm(x_mamba)
        x_mamba = self.mamba(x_mamba)
        x_mamba = x_mamba.transpose(-1, -2).reshape(B, C//2, *img_dims)

        # conv path
        x_conv = self.conv_path(x_conv)

        x = torch.cat([x_conv, x_mamba], dim=1)
        fused = self.channel_shuffle(x, 2)

        attn = self.conv_atten(self.avg_pool(fused))
        attn = self.nonlin(attn)
        fused = fused * attn
        fused = self.conv1(fused)

        attn = self.conv2(x)
        attn = self.nonlin(attn)
        out = fused * attn

        return out 
    
class MambaResBlock(nn.Module):
    """
    VoxRes module
    """

    def __init__(self, channel, depth=1, alpha=0.1):
        super(MambaResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha),
            *[ConvMambaLayer(channel) for j in range(depth)]
        )
        self.actout = nn.Sequential(
            nn.InstanceNorm3d(channel),
            nn.LeakyReLU(alpha),
        )
    def forward(self, x):
        out = self.block(x) + x
        return self.actout(out)


            
class HybridEncoder(nn.Module):
    """
    Main model
    """

    def __init__(self, in_channel=1, first_out_channel=4):
        super(HybridEncoder, self).__init__()

        c = first_out_channel
        self.conv0 = nn.Sequential(
            ConvInsBlock(in_channel, c),
            ConvInsBlock(c, 2*c),
        )

        self.conv1 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(2*c, 4*c),#80
            MambaResBlock(4*c),
            ELK_block(4*c)
        )

        self.conv2 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(4*c, 8*c),#40
            MambaResBlock(8*c),
            ELK_block(8*c)
        )

        self.conv3 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(8*c, 16*c),#20
            MambaResBlock(16*c),
            ELK_block(16*c)
        )
        self.conv4 = nn.Sequential(
            nn.AvgPool3d(2),
            ConvInsBlock(16*c, 32*c),#10
            MambaResBlock(32*c),
            ELK_block(32*c)
        )

    def forward(self, x):
        out0 = self.conv0(x)  # 1
        out1 = self.conv1(out0)  # 1/2
        out2 = self.conv2(out1)  # 1/4
        out3 = self.conv3(out2)  # 1/8
        out4 = self.conv4(out3)  # 1/16

        return out0, out1, out2, out3, out4 
    

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    model = HybridEncoder().cuda()
    x = torch.randn(1, 1, 160, 192, 160).cuda()
    out0, out1, out2, out3, out4 = model(x)
    print(out0.shape, out1.shape, out2.shape, out3.shape, out4.shape)
    print('done')