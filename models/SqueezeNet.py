"""squeezenet in pytorch
[1] Song Han, Jeff Pool, John Tran, William J. Dally
    squeezenet: Learning both Weights and Connections for Efficient Neural Networks
    https://arxiv.org/abs/1506.02626
"""

import torch
import torch.nn as nn
from models.build import ENCODER_REGISTRY


class Fire(nn.Module):

    def __init__(self, in_channel, out_channel, squzee_channel):

        super().__init__()
        self.squeeze = nn.Sequential(
            nn.Conv2d(in_channel, squzee_channel, 1),
            nn.GroupNorm(2,squzee_channel),
            nn.ReLU(inplace=True)
        )

        self.expand_1x1 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 1),
            nn.GroupNorm(2,int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

        self.expand_3x3 = nn.Sequential(
            nn.Conv2d(squzee_channel, int(out_channel / 2), 3, padding=1),
            nn.GroupNorm(2,int(out_channel / 2)),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        x = self.squeeze(x)
        x = torch.cat([
            self.expand_1x1(x),
            self.expand_3x3(x)
        ], 1)

        return x


@ENCODER_REGISTRY.register()
class SqueezeNet_base(nn.Module):

    def __init__(self, args, num_classes=100 ,**kwargs):

        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 96, 3, padding=1),
            nn.GroupNorm(2,96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.fire2 = Fire(96, 128, 16)
        self.fire3 = Fire(128, 128, 16)
        self.fire4 = Fire(128, 256, 32)
        self.fire5 = Fire(256, 256, 32)
        self.fire6 = Fire(256, 384, 48)
        self.fire7 = Fire(384, 384, 48)
        self.fire8 = Fire(384, 512, 64)
        self.fire9 = Fire(512, 512, 64)

        self.conv10 = nn.Conv2d(512, num_classes, 1)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.num_layers = 9

    def forward(self, x, return_feature = False, level = 0, *args ,**kwargs):

        if level<=0:
            f1 = self.stem(x)
        else:
            f1 = x
        
        if level<=1:
            f2 = self.fire2(f1)
        else:
            f2 = f1
        
        if level<=2:
            f3 = self.fire3(f2) + f2
        else:
            f3 = f2
        
        if level<=3:
            f4 = self.fire4(f3)
            f4 = self.maxpool(f4)
        else:
            f4 = f3
          
        if level<=4:

            f5 = self.fire5(f4) + f4
        else:
            f5 = f4
        
        if level<=5:
            
            f6 = self.fire6(f5)
        else:
            f6 = f5
            
        if level<=6:
            f7 = self.fire7(f6) + f6
        else:
            f7 = f6
        
        if level<=7:            
            f8 = self.fire8(f7)
            f8 = self.maxpool(f8)
        else:
            f8 = f7

        f9 = self.fire9(f8)
        c10 = self.conv10(f9)

        x = self.avg(c10)
        x = x.view(x.size(0), -1)
        
        results = {}
        all_outputs = [f1,f2,f3,f4,f5,f6,f7,f8,x]
        for idx, out in enumerate(all_outputs):
            if idx == len(all_outputs) - 1:
                results['logit'] = out
            if idx == len(all_outputs) - 2:
                results['feature'] = out           

            results['layer'+str(idx)]=out

        return results
