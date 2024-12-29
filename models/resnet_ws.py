'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from utils import *
from models.build import ENCODER_REGISTRY
from typing import Dict
from omegaconf import DictConfig
import torch.nn.init as init

import logging
logger = logging.getLogger(__name__)

class WSConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, rho=1e-3, init_mode="kaiming_uniform"):
        super(WSConv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)
        self.rho = rho
        self.init_mode = init_mode
        self._reset_parameters()
        
    # TODO: Check!
    def _reset_parameters(self):
        if self.init_mode == "kaiming_uniform":
            init.kaiming_uniform_(self.weight)
        elif self.init_mode == "kaiming_normal":
            init.kaiming_normal_(self.weight)
        else:
            raise ValueError(f"{self.init_mode} is not supported.")

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


class BasicBlockWS(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, use_bn_layer=False, rho=1e-3, init_mode="kaiming_uniform"):
        super(BasicBlockWS, self).__init__()
        self.conv1 = WSConv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, rho=rho, init_mode=init_mode)
        self.bn1 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes) 
        self.conv2 = WSConv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False, rho=rho, init_mode=init_mode)
        self.bn2 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes) 

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                WSConv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False, rho=rho, init_mode=init_mode),
                nn.GroupNorm(2, self.expansion*planes) if not use_bn_layer else nn.BatchNorm2d(self.expansion*planes) 
            )
            
    def set_rho(self, rho):
        self.conv1.set_rho(rho)
        self.conv2.set_rho(rho)
        if len(self.downsample) > 0:
            self.downsample[0].set_rho(rho)

    def forward_intermediate(self, x: torch.Tensor, no_relu: bool = False) -> torch.Tensor:
        out_i = self.bn1(self.conv1(x))
        out = F.relu(out_i)
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        if not no_relu:
            out = F.relu(out)
        else:
            out = out
        return out, out_i

    def forward(self, x: torch.Tensor, no_relu: bool = False) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        if not no_relu:
            out = F.relu(out)
        else:
            out = out
        return out


class BottleneckWS(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1, use_bn_layer=False, rho=1e-3, init_mode="kaiming_uniform"):
        super(BottleneckWS, self).__init__()
        self.conv1 = WSConv2d(in_planes, planes, kernel_size=1, bias=False, rho=rho, init_mode=init_mode)
        self.bn1 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes)
        self.conv2 = WSConv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False, rho=rho, init_mode=init_mode)
        self.bn2 = nn.GroupNorm(2, planes) if not use_bn_layer else nn.BatchNorm2d(planes)
        self.conv3 = WSConv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False, rho=rho, init_mode=init_mode)
        self.bn3 = nn.GroupNorm(2, self.expansion*planes) if not use_bn_layer else nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                WSConv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False, rho=rho, init_mode=init_mode),
                nn.GroupNorm(2, self.expansion*planes) if not use_bn_layer else nn.BatchNorm2d(planes)
            )
            
    def set_rho(self, rho):
        self.conv1.set_rho(rho)
        self.conv2.set_rho(rho)
        self.conv3.set_rho(rho)
        if len(self.downsample) > 0:
            self.downsample[0].set_rho(rho)

    def forward(self, x: torch.Tensor, no_relu: bool = False) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        if not no_relu:
            out = F.relu(out)
        else:
            out = out
        return out


class ResNet_WSConv(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, l2_norm=False, use_pretrained=False, use_bn_layer=False,
                 last_feature_dim=512, rho=1e-3, init_mode="kaiming_uniform", **kwargs):
        
        #use_pretrained means whether to use torch torchvision.models pretrained model, and use conv1 kernel size as 7
        
        super(ResNet_WSConv, self).__init__()
        self.l2_norm = l2_norm
        self.in_planes = 64
        conv1_kernel_size = 3
        if use_pretrained:
            conv1_kernel_size = 7

        Linear = self.get_linear()   
        self.conv1 = WSConv2d(3, 64, kernel_size=conv1_kernel_size,
                               stride=1, padding=1, bias=False, rho=rho, init_mode=init_mode)
        self.bn1 = nn.GroupNorm(2, 64) if not use_bn_layer else nn.BatchNorm2d(64) 
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, use_bn_layer=use_bn_layer, rho=rho, init_mode=init_mode)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, use_bn_layer=use_bn_layer, rho=rho, init_mode=init_mode)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, use_bn_layer=use_bn_layer, rho=rho, init_mode=init_mode)
        self.layer4 = self._make_layer(block, last_feature_dim, num_blocks[3], stride=2, use_bn_layer=use_bn_layer, rho=rho, init_mode=init_mode)

        self.logit_detach = False        

        if use_pretrained:
            resnet = torchvision.models.resnet18(weights='ResNet18_Weights.DEFAULT')
            self.layer1.load_state_dict(resnet.layer1.state_dict(), strict=False)
            self.layer2.load_state_dict(resnet.layer2.state_dict(), strict=False)
            self.layer3.load_state_dict(resnet.layer3.state_dict(), strict=False)
            self.layer4.load_state_dict(resnet.layer4.state_dict(), strict=False)

        self.num_layers = 6 # layer0 to layer5 (fc)

        if l2_norm:
            self.fc = Linear(last_feature_dim * block.expansion, num_classes, bias=False)
        else:
            self.fc = Linear(last_feature_dim * block.expansion, num_classes)
    
    def set_rho(self, rho):
        self.conv1.rho
        self.layer1.set_rho(rho)
        self.layer2.set_rho(rho)
        self.layer3.set_rho(rho)
        self.layer4.set_rho(rho)        

    def get_linear(self):
        return nn.Linear

    def _make_layer(self, block, planes, num_blocks, stride, use_bn_layer=False, rho=1e-3, init_mode="kaiming_uniform"):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, use_bn_layer=use_bn_layer, rho=rho, init_mode=init_mode))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_feature=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)

        if self.l2_norm:
            self.fc.weight.data = F.normalize(self.fc.weight.data, p=2, dim=1)
            out = F.normalize(out, dim=1)
            logit = self.fc(out)
        else:
            logit = self.fc(out)
            
        if return_feature==True:
            return out, logit
        else:
            return logit
        
    
    def forward_classifier(self,x):
        logit = self.fc(x)
        return logit        
    
    
    def sync_online_and_global(self):
        state_dict=self.state_dict()
        for key in state_dict:
            if 'global' in key:
                x=(key.split("_global"))
                online=(x[0]+x[1])
                state_dict[key]=state_dict[online]
        self.load_state_dict(state_dict)


class ResNet_WS(ResNet_WSConv):

    def forward_layer(self, layer, x, no_relu=True):

        if isinstance(layer, nn.Linear):
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.view(x.size(0), -1)
            out = layer(x)
        else:
            if no_relu:
                out = x
                for sublayer in layer[:-1]:
                    out = sublayer(out)
                out = layer[-1](out, no_relu=no_relu)
            else:
                out = layer(x)

        return out
    
    def forward_layer_by_name(self, layer_name, x, no_relu=True):
        layer = getattr(self, layer_name)
        return self.forward_layer(layer, x, no_relu)

    def forward_layer0(self, x: torch.Tensor, no_relu: bool = False) -> torch.Tensor:
        out0 = self.bn1(self.conv1(x))
        if not no_relu:
            out0 = F.relu(out0)
        return out0

    def freeze_backbone(self):
        for n, p in self.named_parameters():
            if 'fc' not in n:
            # if True:
                p.requires_grad = False
        logger.warning('Freeze backbone parameters (except fc)')
        return
    
    def forward(self, x: torch.Tensor, no_relu: bool = True) -> Dict[str, torch.Tensor]:
        results = {}

        if no_relu:
            out0 = self.bn1(self.conv1(x))
            results['layer0'] = out0
            out0 = F.relu(out0)

            out = out0
            for i, sublayer in enumerate(self.layer1):
                sub_norelu = (i == len(self.layer1) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer1'] = out
            out = F.relu(out)

            for i, sublayer in enumerate(self.layer2):
                sub_norelu = (i == len(self.layer2) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer2'] = out
            out = F.relu(out)

            for i, sublayer in enumerate(self.layer3):
                sub_norelu = (i == len(self.layer3) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer3'] = out
            out = F.relu(out)

            for i, sublayer in enumerate(self.layer4):
                sub_norelu = (i == len(self.layer4) - 1)
                out = sublayer(out, no_relu=sub_norelu)
            results['layer4'] = out
            out = F.relu(out)
            
        else:
            out0 = self.bn1(self.conv1(x))
            out0 = F.relu(out0)
            results['layer0'] = out0

            out = out0
            out = self.layer1(out)
            results['layer1'] = out
            out = self.layer2(out)
            results['layer2'] = out
            out = self.layer3(out)
            results['layer3'] = out
            out = self.layer4(out)
            results['layer4'] = out
            

        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)

        if self.logit_detach:
            logit = self.fc(out.detach())
        else:
            logit = self.fc(out)

        results['feature'] = out
        results['logit'] = logit
        results['layer5'] = logit

        return results

@ENCODER_REGISTRY.register()
class ResNet18_WS(ResNet_WS):
    
    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(BasicBlockWS, [2, 2, 2, 2], num_classes=num_classes, **kwargs
                        #  l2_norm=args.model.l2_norm,
                        #  use_pretrained=args.model.pretrained, use_bn_layer=args.model.use_bn_layer
                         )

@ENCODER_REGISTRY.register()
class ResNet34_WS(ResNet_WS):

    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(BasicBlockWS, [3, 4, 6, 3], num_classes=num_classes, **kwargs
                        #  l2_norm=args.model.l2_norm,
                        #  use_pretrained=args.model.pretrained, use_bn_layer=args.model.use_bn_layer
                         )
     