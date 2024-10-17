#Code and refer from FedMA 


#!/usr/bin/env python
# coding: utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from models.build import ENCODER_REGISTRY


'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init
import torchinfo
from models.build import ENCODER_REGISTRY


class WSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, rho=1e-3):
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


@ENCODER_REGISTRY.register()
class VGG9_WS(nn.Module):
    
    def __init__(self, args, num_classes=10, rho=1e-3, **kwargs):
        super(VGG9_WS, self).__init__()
        
        self.layer1 = nn.Sequential(WSConv2d(3, 32, kernel_size=3, padding=1, bias=False, rho=rho), nn.GroupNorm(2, 32), nn.ReLU())
        self.layer2 = nn.Sequential(WSConv2d(32, 64, kernel_size=3, padding=1, bias=False, rho=rho),nn.GroupNorm(2, 64), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(WSConv2d(64, 128, kernel_size=3, padding=1, bias=False, rho=rho) ,nn.GroupNorm(2, 128), nn.ReLU())
        self.layer4 = nn.Sequential(WSConv2d(128, 128, kernel_size=3, padding=1, bias=False, rho=rho) ,nn.GroupNorm(2, 128), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2) )
        self.layer5 = nn.Sequential(WSConv2d(128, 256, kernel_size=3, padding=1, bias=False, rho=rho) ,nn.GroupNorm(2, 256), nn.ReLU() )
        self.layer6 = nn.Sequential(WSConv2d(256, 256, kernel_size=3, padding=1, bias=False, rho=rho) ,nn.GroupNorm(2, 256), nn.ReLU() , nn.MaxPool2d(kernel_size=2, stride=2) )

        self.classifier = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        self.pool =  nn.AdaptiveAvgPool2d((4,4))
        self.num_layers = 7


    def forward(self, x, return_feature = False, level = 0, *args ,**kwargs):

        if level <= 0:
            out0 = self.layer1(x)
        else:
            out0 = x

        if level<= 1:
            out1 = self.layer2(out0)
        else:
            out1 = out0

        if level <= 2:
            out2 = self.layer3(out1)
        else:
            out2 = out1

        if level<=3:
            out3 = self.layer4(out2)
        else:
            out3 = out2

        if level <= 4:
            out4 = self.layer5(out3)
        else:
            out4 = out3

        if level<=5:
            out5 = self.layer6(out4)
            out5 = self.pool(out5)
            out5 = out5.view(out5.size(0), -1)
        else:
            out5 = out4

        logit = self.classifier(out5)

        results = {}
        all_outputs = [out0,out1,out2,out3,out4,out5,logit]
        for idx, out in enumerate(all_outputs):
            if idx == len(all_outputs) - 1:
                results['logit'] = out
            if idx == len(all_outputs) - 2:
                results['feature'] = out           

            results['layer'+str(idx)]=out

        return results        

