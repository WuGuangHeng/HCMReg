"""
HCMReg
Mamba-based registration model
"""

import torch
import torch.nn as nn
from func import SpatialTransformer, VecInt
from HybridEnc import HybridEncoder
from CrossMambaFusion import CrossMambaFusion

class HCMReg(nn.Module):
    def __init__(
        self,
        inshape=(160, 192, 160),
        in_channel=1,
        dims = [8, 16, 32, 64, 128],
        diff = False,
    ):
        super(HCMReg, self).__init__()

        self.step = 7
        self.inshape = inshape
        self.dims = dims

        self.encoder = HybridEncoder(in_channel=in_channel, first_out_channel=dims[0]//2) ### NOTE this one

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.upsample_trilin = nn.Upsample(
            scale_factor=2, mode="trilinear", align_corners=True
        )  
       
        self.cm1 = CrossMambaFusion(dims[0])
        self.cm2 = CrossMambaFusion(dims[1])
        self.cm3 = CrossMambaFusion(dims[2])
        self.cm4 = CrossMambaFusion(dims[3])
        self.cm5 = CrossMambaFusion(dims[4])

        self.transformer = nn.ModuleList()
        for i in range(4):
            self.transformer.append(
                SpatialTransformer([s // 2**(i) for s in inshape]))
        
        self.diffs = None
        if diff:
            self.diffs = nn.ModuleList()
            for i in range(5):
                self.diffs.append(VecInt([s // 2**(i) for s in inshape]))
            
    def feture_shape(self, inshape, scale):
        return tuple([s // scale for s in inshape])
    
    def forward(self, moving, fixed, multi_flows=False):
        flows = []
        # encode stage
        M1, M2, M3, M4, M5 = self.encoder(moving)
        F1, F2, F3, F4, F5 = self.encoder(fixed)
    
        w = self.cm5(F5, M5) 
        # print("w.shape", w.shape)
        if self.diffs:
            w = self.diffs[4](w)
        flow = self.upsample_trilin(2 * w)
        flows.append(flow)

        M4 = self.transformer[3](M4, flow)
        w = self.cm4(F4, M4)
        if self.diffs:
            w = self.diffs[3](w)
        flow = self.upsample_trilin(2 * (self.transformer[3](flow, w) + w))
        flows.append(flow)

        M3 = self.transformer[2](M3, flow)
        w = self.cm3(F3, M3)
        if self.diffs:
            w = self.diffs[2](w)
        flow = self.upsample_trilin(2 * (self.transformer[2](flow, w) + w))
        flows.append(flow)

        M2 = self.transformer[1](M2, flow)
        w = self.cm2(F2, M2)
        if self.diffs:
            w = self.diffs[1](w)
        flow = self.upsample_trilin(2*(self.transformer[1](flow, w) + w))
        flows.append(flow)

        M1 = self.transformer[0](M1, flow)
        w = self.cm1(F1, M1)
        if self.diffs:
            w = self.diffs[0](w)
        flow = self.transformer[0](flow, w) + w
        flows.append(flow)

        y_moved = self.transformer[0](moving, flow)

        return (y_moved, flow) if not multi_flows else (y_moved, flows)
    
    


if __name__ == "__main__":
    import os 
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    inshape = (1, 1, 160, 192, 160)
    model = HCMReg(inshape=inshape[2:]).cuda()
    A = torch.ones(inshape).cuda()
    B = torch.ones(inshape).cuda()
    out, flow = model(A, B)
    print(out.shape, flow.shape)
    import thop
    macs, params = thop.profile(model, inputs=(A, B))
    flops, params = thop.clever_format([macs*2, params], "%.3f")
    print("FLOPs: {}, Params: {}".format(flops, params))
