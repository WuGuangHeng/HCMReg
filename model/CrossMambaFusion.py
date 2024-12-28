import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from einops import rearrange, repeat
from selective_scan import selective_scan_fn as selective_scan_fn_v1
import selective_scan_cuda_core as selective_scan_cuda
from functools import partial

######################## Cross Mamaba Fusion ############################################

class Cross_Mamba_SSM(nn.Module):
    def __init__(
            self,
            # basic dims ===========
            d_model=96,
            d_state=4,
            ssm_ratio=2,
            dt_rank="auto",
            # dt init ==============
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            # ======================
            **kwargs,
    ):
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # x proj; dt proj ============================
        self.x_proj_1 = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.x_proj_2 = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)

        self.dt_proj_1 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                    **factory_kwargs)
        self.dt_proj_2 = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                    **factory_kwargs)

        # A, D =======================================
        self.A_log_1 = self.A_log_init(self.d_state, self.d_inner)  # (D, N)
        self.A_log_2 = self.A_log_init(self.d_state, self.d_inner)  # (D)
        self.D_1 = self.D_init(self.d_inner)  # (D)
        self.D_2 = self.D_init(self.d_inner)  # (D)

        # out norm ===================================
        self.out_norm_1 = nn.LayerNorm(self.d_inner)
        self.out_norm_2 = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner*2, self.d_inner, **factory_kwargs)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        # dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward(self, x1: torch.Tensor, x2: torch.Tensor):
        selective_scan = selective_scan_fn_v1
        B, L, d = x1.shape
        x1 = x1.permute(0, 2, 1)
        x2 = x2.permute(0, 2, 1)
        x1_dbl = self.x_proj_1(rearrange(x1, "b d l -> (b l) d"))  # (bl d)
        x2_dbl = self.x_proj_2(rearrange(x2, "b d l -> (b l) d"))  # (bl d)
        dt_1, B_1, C_1 = torch.split(x1_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_2, B_2, C_2 = torch.split(x2_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt_1 = self.dt_proj_1.weight @ dt_1.t()
        dt_2 = self.dt_proj_2.weight @ dt_2.t()
        dt_1 = rearrange(dt_1, "d (b l) -> b d l", l=L)
        dt_2 = rearrange(dt_2, "d (b l) -> b d l", l=L)
        A_1 = -torch.exp(self.A_log_1.float())  # (k * d, d_state)
        A_2 = -torch.exp(self.A_log_2.float())  # (k * d, d_state)
        B_1 = rearrange(B_1, "(b l) dstate -> b dstate l", l=L).contiguous()
        B_2 = rearrange(B_2, "(b l) dstate -> b dstate l", l=L).contiguous()
        C_1 = rearrange(C_1, "(b l) dstate -> b dstate l", l=L).contiguous()
        C_2 = rearrange(C_2, "(b l) dstate -> b dstate l", l=L).contiguous()

        y1 = selective_scan(
            x1, dt_2,
            A_2, B_2, C_2, self.D_1.float(),
            delta_bias=self.dt_proj_1.bias.float(),
            delta_softplus=True,
        )
        y2 = selective_scan(
            x2, dt_1,
            A_1, B_1, C_1, self.D_2.float(),
            delta_bias=self.dt_proj_2.bias.float(),
            delta_softplus=True,
        )

        # # Ablation: no cross fusion
        # y1 = selective_scan(
        #     x1, dt_1,
        #     A_1, B_1, C_1, self.D_1.float(),
        #     delta_bias=self.dt_proj_1.bias.float(),
        #     delta_softplus=True,
        # )
        # y2 = selective_scan(
        #     x2, dt_2,
        #     A_2, B_2, C_2, self.D_2.float(),
        #     delta_bias=self.dt_proj_2.bias.float(),
        #     delta_softplus=True,
        # )

        y1 = rearrange(y1, "b d l -> b l d")
        y1 = self.out_norm_1(y1)
        y2 = rearrange(y2, "b d l -> b l d")
        y2 = self.out_norm_2(y2)
        y = torch.cat([y1, y2], dim=-1)
        y = self.out_proj(y)
        return y

class CrossMambaFusion_3DSSM(nn.Module):
    '''
    Cross Mamba Fusion Selective Scan 3D Module with SSM
    '''
    def __init__(
        self,
        # basic dims ===========
        d_model=96,
        d_state=16,
        ssm_ratio=2,
        dt_rank="auto",
        # dwconv ===============
        d_conv=3, # < 2 means no conv 
        conv_bias=True,
        # ======================
        bias=False,
        # dt init ==============
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        # ======================
        softmax_version=False,
        # ======================
        **kwargs,
    ):            
        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj1 = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.in_proj2 = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        
        # conv =======================================
        if self.d_conv > 1:
            self.conv3d = nn.Conv3d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        self.CM_ssm = Cross_Mamba_SSM(
            d_model=self.d_model,
            d_state=self.d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            dt_min=dt_min,
            dt_max=dt_max,
            dt_init=dt_init,
            dt_scale=dt_scale,
            dt_init_floor=dt_init_floor,
            **kwargs,
        )
        # DefConv
        self.def_conv = nn.Conv3d(self.d_inner, 3, kernel_size=3, padding=1)
        self.def_conv.weight = nn.Parameter(Normal(0, 1e-5).sample(self.def_conv.weight.shape))
        self.def_conv.bias = nn.Parameter(torch.zeros_like(self.def_conv.bias))
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor):

        '''
        B  D H W C, B D H W C -> B 3 D H W
        '''
        x1 = self.in_proj1(x1)
        x2 = self.in_proj2(x2)
        B, D, H, W, C = x1.shape
        if self.d_conv > 1:
            x1_trans = x1.permute(0, 4, 1, 2, 3).contiguous()
            x2_trans = x2.permute(0, 4, 1, 2, 3).contiguous()
            x1_conv = self.act(self.conv3d(x1_trans)) # (b, c, d, h, w)
            x2_conv = self.act(self.conv3d(x2_trans)) # (b, c, d, h, w)
            x1_conv = rearrange(x1_conv, "b c d h w -> b (d h w) c")
            x2_conv = rearrange(x2_conv, "b c d h w -> b (d h w) c")

            y = self.CM_ssm(x1_conv, x2_conv) # b (d h w) c 
            y = y.reshape(B, D, H, W, C).permute(0, 4, 1, 2, 3).contiguous() # (B, C, D, H, W)
        
        flow = self.def_conv(y)
   
        return flow
    

class CrossMambaFusion(nn.Module):
    def __init__(self, dim, d_state=16):
        super().__init__()

        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.fusion_block = CrossMambaFusion_3DSSM(dim, d_state=d_state)

    def forward(self, x1, x2):
        
        x1 = x1.permute(0, 2, 3, 4, 1).contiguous()
        x2 = x2.permute(0, 2, 3, 4, 1).contiguous()
        x1, x2 = self.norm(x1), self.norm(x2)

        flow = self.fusion_block(x1, x2)

        return flow
    

if __name__ == "__main__":
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    cm = CrossMambaFusion(4).cuda()
    x1 = torch.randn(1, 4, 160, 192, 160).cuda()
    x2 = torch.randn(1, 4, 160, 192, 160).cuda()
    y = cm(x1, x2)
    print(y.shape)