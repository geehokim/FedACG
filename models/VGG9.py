#Code and refer from FedMA 


#!/usr/bin/env python
# coding: utf-8

# In[ ]:

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


class VGG9(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, num_classes=10, *args ,**kwargs):
        super(VGG9, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1) , nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1) , nn.ReLU(inplace=True))
        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1) , nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=2, stride=2) ,  nn.Dropout(p = 0.05))
        self.layer5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1) , nn.ReLU(inplace=True) )
        self.layer6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1) , nn.ReLU(inplace=True) , nn.MaxPool2d(kernel_size=2, stride=2) , nn.Dropout(p = 0.1))
                
        self.classifier = nn.Sequential(
            nn.Linear(4096, 512),
            nn.ReLU(True),
            nn.Dropout(p = 0.1),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Linear(512, num_classes),
        )
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x, *args ,**kwargs):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x



@ENCODER_REGISTRY.register()
class VGG9_base(nn.Module):
    
    def __init__(self, args, num_classes=10 ,**kwargs):
        super(VGG9_base, self).__init__()
        
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3, padding=1) ,nn.GroupNorm(2, 32), nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3, padding=1),nn.GroupNorm(2, 64), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3, padding=1) ,nn.GroupNorm(2, 128), nn.ReLU())
        self.layer4 = nn.Sequential(nn.Conv2d(128, 128, kernel_size=3, padding=1) ,nn.GroupNorm(2, 128), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2) )
        self.layer5 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3, padding=1) ,nn.GroupNorm(2, 256), nn.ReLU() )
        self.layer6 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, padding=1) ,nn.GroupNorm(2, 256), nn.ReLU() , nn.MaxPool2d(kernel_size=2, stride=2) )

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
                m.bias.data.zero_()

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

