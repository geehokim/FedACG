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
from models.resnet_base import ResNet_base
from typing import Callable, Dict, Tuple, Union, List, Optional
from omegaconf import DictConfig

import logging
logger = logging.getLogger(__name__)

class ResNet_MLB(ResNet_base):

    def inv_norelu_feat(self, x: torch.Tensor, no_relu: bool) -> torch.Tensor:
        if no_relu:
            return F.relu(x)
        else:
            return x

    def forward(self, x: torch.Tensor, no_relu: bool = True, mlb_level= None) -> Dict[str, torch.Tensor]:
        results = {}

        if mlb_level is None:
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

        else:   #MLB global branch forward
                x = self.inv_norelu_feat(x, no_relu)

                if mlb_level <= 0:
                    out0 = self.bn1(self.conv1(x))
                    out0 = F.relu(out0)
                else:
                    out0 = x
                    
                results['layer0'] = out0

                if mlb_level <= 1:
                    out1 = self.layer1(out0)
                else:
                    out1 = out0
                results['layer1'] = out1

                if mlb_level <= 2:
                    out2 = self.layer2(out1)
                else:
                    out2 = out1
                results['layer2'] = out2

                if mlb_level <= 3:
                    out3 = self.layer3(out2)
                else:
                    out3 = out2
                results['layer3'] = out3

                if mlb_level <= 4:
                    out4 = self.layer4(out3)
                else:
                    out4 = out3
                results['layer4'] = out4

                out4 = F.adaptive_avg_pool2d(out4, 1)
                out4 = out4.view(out4.size(0), -1)

                if self.logit_detach:
                    logit = self.fc(out4.detach())
                else:
                    logit = self.fc(out4)

                results['feature'] = out4
                results['logit'] = logit
                results['layer5'] = logit

                return results



    

@ENCODER_REGISTRY.register()
class ResNet18_MLB(ResNet_MLB):
    
    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs
                        #  l2_norm=args.model.l2_norm,
                        #  use_pretrained=args.model.pretrained, use_bn_layer=args.model.use_bn_layer
                         )
    


@ENCODER_REGISTRY.register()
class ResNet34_MLB(ResNet_MLB):

    def __init__(self, args: DictConfig, num_classes: int = 10, **kwargs):
        super().__init__(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs
                        #  l2_norm=args.model.l2_norm,
                        #  use_pretrained=args.model.pretrained, use_bn_layer=args.model.use_bn_layer
                         )
     