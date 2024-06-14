'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import copy
from utils import *
import numpy as np
from models.resnet import BasicBlock, Bottleneck, ResNet
from models.build import ENCODER_REGISTRY
from typing import Callable, Dict, Tuple, Union, List, Optional
from omegaconf import DictConfig

import logging
logger = logging.getLogger(__name__)

class ResNet_base(ResNet):

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
class ResNet18_base(ResNet_base):
    
    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs
                        #  l2_norm=args.model.l2_norm,
                        #  use_pretrained=args.model.pretrained, use_bn_layer=args.model.use_bn_layer
                         )
    


@ENCODER_REGISTRY.register()
class ResNet34_base(ResNet_base):

    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs
                        #  l2_norm=args.model.l2_norm,
                        #  use_pretrained=args.model.pretrained, use_bn_layer=args.model.use_bn_layer
                         )
     