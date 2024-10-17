#!/usr/bin/env python
# coding: utf-8

#mobilenetv2 code refer : https://github.com/weiaicunzai/pytorch-cifar100

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.build import ENCODER_REGISTRY


class WSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=False, rho=1e-3):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.rho = rho

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight) * self.rho
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)
        
    def set_rho(self, rho):
        self.rho = rho
        

class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, rho=1e-3, num_classes=100):
        super().__init__()

        self.residual = nn.Sequential(
            WSConv2d(in_channels, in_channels * t, 1, bias=False, rho=rho),
            nn.GroupNorm(2, in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t, bias=False),
            nn.GroupNorm(2,in_channels * t),
            nn.ReLU6(inplace=True),

            WSConv2d(in_channels * t, out_channels, 1, bias=False, rho=rho),
            nn.GroupNorm(2,out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual


@ENCODER_REGISTRY.register()
class MobileNetV2_WS(nn.Module):

    def __init__(self, args, num_classes=100, rho=1e-3, **kwargs):
        super().__init__()

        self.pre = nn.Sequential(
            WSConv2d(3, 32, 1, padding=1, bias=False, rho=rho),
            nn.GroupNorm(2, 32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 1, rho=rho)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6, rho=rho)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6, rho=rho)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6, rho=rho)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6, rho=rho)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6, rho=rho)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6, rho=rho)

        self.conv1 = nn.Sequential(
            WSConv2d(320, 1280, 1, bias=False, rho=rho),
            nn.GroupNorm(2,1280),
            nn.ReLU6(inplace=True)
        )

        print("num_classes:", num_classes)
        self.conv2 = nn.Conv2d(1280, num_classes, 1)
        self.num_layers = 8

        
    def _make_stage(self, repeat, in_channels, out_channels, stride, t, rho=1e-3):

        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t, rho=rho))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t, rho=rho))
            repeat -= 1

        return nn.Sequential(*layers)        
        
    def forward(self, x, return_feature = False, level = 0, *args ,**kwargs):
        

        if level<=0:
            out0 = self.pre(x)                     
        else:
            out0 = x
        
        if level<=1:
            out1 = self.stage1(out0)   
        else:
            out1 = out0
        
        if level<=2:
            out2 = self.stage2(out1)        
        else:
            out2 = out1
        
        if level<=3:
            out3 = self.stage3(out2)
        else:
            out3 = out2
            
        if level<=4:
            out4 = self.stage4(out3)
        else:
            out4 = out3
        
        if level<=5:
            out5 = self.stage5(out4) 
        else:
            out5 = out4
        
        if level<=6:
            out6 = self.stage6(out5)
        else:
            out6 = out5

        x = self.stage7(out6)
            
        
        x = self.conv1(x) 
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        logit = x.view(x.size(0), -1)

        results = {}
        all_outputs = [out0,out1,out2,out3,out4,out5,out6,logit]
        for idx, out in enumerate(all_outputs):
            if idx == len(all_outputs) - 1:
                results['logit'] = out
            if idx == len(all_outputs) - 2:
                results['feature'] = out           

            results['layer'+str(idx)]=out

        return results

