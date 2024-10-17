#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.build import ENCODER_REGISTRY

'''
MobileNetV3 From <Searching for MobileNetV3>, arXiv:1905.02244.
Ref: https://github.com/d-li14/mobilenetv3.pytorch/blob/master/mobilenetv3.py
     https://github.com/kuan-wang/pytorch-mobilenet-v3/blob/master/mobilenetv3.py
     
'''

def _ensure_divisible(number, divisor, min_value=None):
    '''
    Ensure that 'number' can be 'divisor' divisible
    Reference from original tensorflow repo:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    '''
    if min_value is None:
        min_value = divisor
    new_num = max(min_value, int(number + divisor / 2) // divisor * divisor)
    if new_num < 0.9 * number:
        new_num += divisor
    return new_num


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

    
class H_sigmoid(nn.Module):
    '''
    hard sigmoid
    '''
    def __init__(self, inplace=True):
        super(H_sigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3, inplace=self.inplace) / 6

class H_swish(nn.Module):
    '''
    hard swish
    '''
    def __init__(self, inplace=True):
        super(H_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3, inplace=self.inplace) / 6

class SEModule(nn.Module):
    '''
    SE Module
    Ref: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py
    '''
    def __init__(self, in_channels_num, reduction_ratio=4):
        super(SEModule, self).__init__()

        if in_channels_num % reduction_ratio != 0:
            raise ValueError('in_channels_num must be divisible by reduction_ratio(default = 4)')

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels_num, in_channels_num // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels_num // reduction_ratio, in_channels_num, bias=False),
            H_sigmoid()
        )

    def forward(self, x):
        batch_size, channel_num, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channel_num)
        y = self.fc(y).view(batch_size, channel_num, 1, 1)
        return x * y


class Bottleneck(nn.Module):
    '''
    The basic unit of MobileNetV3
    '''
    def __init__(self, in_channels_num, exp_size, out_channels_num, kernel_size, stride, use_SE, NL, rho=1e-3):
        '''
        use_SE: True or False -- use SE Module or not
        NL: nonlinearity, 'RE' or 'HS'
        '''
        super(Bottleneck, self).__init__()

        assert stride in [1, 2]
        NL = NL.upper()
        assert NL in ['RE', 'HS']

        use_HS = NL == 'HS'
        
        # Whether to use residual structure or not
        self.use_residual = (stride == 1 and in_channels_num == out_channels_num)

        if exp_size == in_channels_num:
            # Without expansion, the first depthwise convolution is omitted
            self.conv1 = nn.Sequential(
                # Depthwise Convolution
                WSConv2d(in_channels=in_channels_num, 
                          out_channels=exp_size, 
                          kernel_size=kernel_size, 
                          stride=stride, 
                          padding=(kernel_size-1)//2, 
                          groups=in_channels_num, 
                          bias=False,
                          rho=rho),
                nn.GroupNorm(num_groups=2, 
                             num_channels=exp_size),
                # SE Module
                SEModule(exp_size) if use_SE else nn.Sequential(),
                H_swish() if use_HS else nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(
                # Linear Pointwise Convolution
                WSConv2d(in_channels=exp_size, 
                          out_channels=out_channels_num, 
                          kernel_size=1, 
                          stride=1, 
                          padding=0, 
                          bias=False,
                          rho=rho),
                #nn.BatchNorm2d(num_features=out_channels_num, momentum=BN_momentum)
                nn.GroupNorm(num_groups=2, 
                             num_channels=out_channels_num)
            )
        else:
            # With expansion
            self.conv1 = nn.Sequential(
                # Pointwise Convolution for expansion
                WSConv2d(in_channels=in_channels_num, 
                          out_channels=exp_size, 
                          kernel_size=1, 
                          stride=1, 
                          padding=0, 
                          bias=False,
                          rho=rho),
                nn.GroupNorm(num_groups=2,
                             num_channels=exp_size),
                H_swish() if use_HS else nn.ReLU(inplace=True))
            self.conv2 = nn.Sequential(
                # Depthwise Convolution
                WSConv2d(in_channels=exp_size, 
                          out_channels=exp_size, 
                          kernel_size=kernel_size, 
                          stride=stride, 
                          padding=(kernel_size-1)//2, 
                          groups=exp_size, 
                          bias=False,
                          rho=rho),
                nn.GroupNorm(num_groups=2, 
                             num_channels=exp_size),
                # SE Module
                SEModule(exp_size) if use_SE else nn.Sequential(),
                H_swish() if use_HS else nn.ReLU(inplace=True),
                # Linear Pointwise Convolution
                WSConv2d(in_channels=exp_size, 
                          out_channels=out_channels_num, 
                          kernel_size=1, 
                          stride=1, 
                          padding=0, 
                          bias=False,
                          rho=rho),
                #nn.BatchNorm2d(num_features=out_channels_num, momentum=BN_momentum)
                nn.GroupNorm(num_groups=2,
                             num_channels=out_channels_num)
            )

    def forward(self, x, expand=False):
        out1 = self.conv1(x)
        out = self.conv2(out1)
        if self.use_residual:
            out = out + x
        if expand:
            return out, out1
        else:
            return out

@ENCODER_REGISTRY.register()
class MobileNetV3_WS(nn.Module):
    '''
    
    '''
    def __init__(self, args, num_classes=100, width_multiplier=1.0, dropout=0.2, rho=1e-3, **kwargs):
        '''
        configs: setting of the model
        mode: type of the model, 'large' or 'small'
        '''
        super(MobileNetV3_WS, self).__init__()
        s = 1
        # if input_size == 32 or input_size == 56:
        #     # using cifar-10, cifar-100 or Tiny-ImageNet
        #     s = 1

        # setting of the model
        # Configuration of a MobileNetV3-Small Model
        configs = [
            #kernel_size, exp_size, out_channels_num, use_SE, NL, stride
            [3, 16, 16, True, 'RE', s],
            [3, 72, 24, False, 'RE', 2],    # (1): out0
            [3, 88, 24, False, 'RE', 1],
            [5, 96, 40, True, 'HS', 2],     # (3): out1
            [5, 240, 40, True, 'HS', 1],
            [5, 240, 40, True, 'HS', 1],
            [5, 120, 48, True, 'HS', 1],
            [5, 144, 48, True, 'HS', 1],
            [5, 288, 96, True, 'HS', 2],    # (8): out2
            [5, 576, 96, True, 'HS', 1],
            [5, 576, 96, True, 'HS', 1]
        ]

        first_channels_num = 16

        # last_channels_num = 1280
        # according to https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v3.py
        # if small -- 1024, if large -- 1280
        last_channels_num = 1024

        divisor = 8

        ########################################################################################################################
        # feature extraction part
        # input layer
        input_channels_num = _ensure_divisible(first_channels_num * width_multiplier, divisor)
        last_channels_num = _ensure_divisible(last_channels_num * width_multiplier, divisor) if width_multiplier > 1 else last_channels_num
        feature_extraction_layers = []
        self.first_layer = nn.Sequential(
            WSConv2d(in_channels=3, 
                      out_channels=input_channels_num, 
                      kernel_size=3, 
                      stride=s, 
                      padding=1, 
                      bias=False,
                      rho=rho),
            nn.GroupNorm(num_groups=2, 
                         num_channels=input_channels_num),
            H_swish()
        )

        # Overlay of multiple bottleneck structures
        for kernel_size, exp_size, out_channels_num, use_SE, NL, stride in configs:
            output_channels_num = _ensure_divisible(out_channels_num * width_multiplier, divisor)
            exp_size = _ensure_divisible(exp_size * width_multiplier, divisor)
            feature_extraction_layers.append(Bottleneck(input_channels_num, exp_size, output_channels_num, kernel_size, stride, use_SE, NL, rho=rho))
            input_channels_num = output_channels_num
        
        # the last stage
        # last_stage_channels_num = _ensure_divisible(exp_size * width_multiplier, divisor)
        self.last_stage_layer1 = nn.Sequential(
                WSConv2d(in_channels=input_channels_num, 
                          out_channels=last_channels_num, 
                          kernel_size=1, 
                          stride=1, 
                          padding=0, 
                          bias=False,
                          rho=rho),
                nn.GroupNorm(num_groups=2, 
                             num_channels=last_channels_num),
                H_swish()
            )
        
        self.featureList = nn.ModuleList(feature_extraction_layers)

        # SE Module
        # remove the last SE Module according to https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet_v3.py
        # feature_extraction_layers.append(SEModule(last_stage_channels_num) if mode == 'small' else nn.Sequential())

        last_stage = []
        last_stage.append(nn.AdaptiveAvgPool2d(1))
        # last_stage.append(nn.Conv2d(in_channels=last_stage_channels_num, 
        #                             out_channels=last_channels_num, 
        #                             kernel_size=1, 
        #                             stride=1, 
        #                             padding=0, 
        #                             bias=False))
        # last_stage.append(H_swish())

        self.last_stage_layers = nn.Sequential(*last_stage)

        ########################################################################################################################
        # Classification part
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(last_channels_num, num_classes)
        )
        
        self.num_layers = 13
        self._initialize_weights()

        '''
        self.extras = nn.ModuleList([
            InvertedResidual(576, 512, 2, 0.2),
            InvertedResidual(512, 256, 2, 0.25),
            InvertedResidual(256, 256, 2, 0.5),
            InvertedResidual(256, 64, 2, 0.25)
        ])
        '''

    def forward(self, x):
        x = self.first_layer(x)
        for i, f in enumerate(self.featureList):
            x = f(x)
            
            if i == 1:
                out0 = x
            elif i == 3:
                out1 = x
            elif i == 8:
                out2 = x
        out3 = self.last_stage_layer1(x)
        x = self.last_stage_layers(out3)
        x = x.view(x.size(0), -1)
        logit = self.classifier(x)
        
        results = {}
        all_outputs = [out0,out1,out2,out3,logit]
        for idx, out in enumerate(all_outputs):
            if idx == len(all_outputs) - 1:
                results['logit'] = out
            if idx == len(all_outputs) - 2:
                results['feature'] = out           

            results['layer'+str(idx)]=out

        return results

    def _initialize_weights(self):
        '''
        Initialize the weights
        '''
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, WSConv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
