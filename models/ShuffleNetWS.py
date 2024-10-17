"""shufflenet in pytorch

[1] Xiangyu Zhang, Xinyu Zhou, Mengxiao Lin, Jian Sun.

    ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices
    https://arxiv.org/abs/1707.01083v2
"""

from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
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


class BasicWSConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, bias=False, rho=1e-3, **kwargs):
        super().__init__()
        self.conv = WSConv2d(input_channels, output_channels, kernel_size, bias=bias, rho=rho, **kwargs)
        self.bn = nn.GroupNorm(2, output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class ChannelShuffle(nn.Module):

    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, x):
        batchsize, channels, height, width = x.data.size()
        channels_per_group = int(channels / self.groups)

        #"""suppose a convolutional layer with g groups whose output has
        #g x n channels; we first reshape the output channel dimension
        #into (g, n)"""
        x = x.view(batchsize, self.groups, channels_per_group, height, width)

        #"""transposing and then flattening it back as the input of next layer."""
        x = x.transpose(1, 2).contiguous()
        x = x.view(batchsize, -1, height, width)

        return x
    
    
class DepthwiseConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, bias=False, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size, bias=bias, **kwargs),
            nn.GroupNorm(2,output_channels)
        )

    def forward(self, x):
        return self.depthwise(x)


class PointwiseWSConv2d(nn.Module):
    def __init__(self, input_channels, output_channels, bias=False, rho=1e-3, **kwargs):
        super().__init__()
        self.pointwise = nn.Sequential(
            WSConv2d(input_channels, output_channels, 1, bias=bias, rho=rho, **kwargs),
            nn.GroupNorm(2, output_channels)
        )

    def forward(self, x):
        return self.pointwise(x)

class ShuffleNetWSUnit(nn.Module):

    def __init__(self, input_channels, output_channels, stage, stride, groups, bias, rho):
        super().__init__()

        #"""Similar to [9], we set the number of bottleneck channels to 1/4
        #of the output channels for each ShuffleNet unit."""
        self.bottlneck = nn.Sequential(
            PointwiseWSConv2d(
                input_channels,
                int(output_channels / 4),
                groups=groups,
                bias=bias,
                rho=rho
            ),
            nn.ReLU(inplace=True)
        )

        #"""Note that for Stage 2, we do not apply group convolution on the first pointwise
        #layer because the number of input channels is relatively small."""
        if stage == 2:
            self.bottlneck = nn.Sequential(
                PointwiseWSConv2d(
                    input_channels,
                    int(output_channels / 4),
                    groups=groups,
                    bias=bias,
                    rho=rho
                ),
                nn.ReLU(inplace=True)
            )

        self.channel_shuffle = ChannelShuffle(groups)

        self.depthwise = DepthwiseConv2d(
            int(output_channels / 4),
            int(output_channels / 4),
            3,
            groups=int(output_channels / 4),
            stride=stride,
            padding=1,
            bias=bias,
        )

        self.expand = PointwiseWSConv2d(
            int(output_channels / 4),
            output_channels,
            groups=groups,
            bias=bias,
            rho=rho
        )

        self.relu = nn.ReLU(inplace=True)
        self.fusion = self._add
        self.shortcut = nn.Sequential()

        #"""As for the case where ShuffleNet is applied with stride,
        #we simply make two modifications (see Fig 2 (c)):
        #(i) add a 3 × 3 average pooling on the shortcut path;
        #(ii) replace the element-wise addition with channel concatenation,
        #which makes it easy to enlarge channel dimension with little extra
        #computation cost.
        if stride != 1 or input_channels != output_channels:
            self.shortcut = nn.AvgPool2d(3, stride=2, padding=1)

            self.expand = PointwiseWSConv2d(
                int(output_channels / 4),
                output_channels - input_channels,
                groups=groups,
                bias=bias,
                rho=rho
            )

            self.fusion = self._cat

    def _add(self, x, y):
        return torch.add(x, y)

    def _cat(self, x, y):
        return torch.cat([x, y], dim=1)

    def forward(self, x):
        shortcut = self.shortcut(x)

        shuffled = self.bottlneck(x)
        shuffled = self.channel_shuffle(shuffled)
        shuffled = self.depthwise(shuffled)
        shuffled = self.expand(shuffled)

        output = self.fusion(shortcut, shuffled)
        output = self.relu(output)

        return output
    

@ENCODER_REGISTRY.register()
class ShuffleNet_WS(nn.Module):

    def __init__(self,args, num_classes=100, groups=3, bias=False, rho=1e-3, **kwargs):
        super().__init__()
        num_blocks = [4, 8, 4]
        if groups == 1:
            out_channels = [24, 144, 288, 567]
        elif groups == 2:
            out_channels = [24, 200, 400, 800]
        elif groups == 3:
            out_channels = [24, 240, 480, 960]
        elif groups == 4:
            out_channels = [24, 272, 544, 1088]
        elif groups == 8:
            out_channels = [24, 384, 768, 1536]

        self.conv1 = BasicWSConv2d(3, out_channels[0], 3, padding=1, stride=1, bias=bias, rho=rho)
        self.input_channels = out_channels[0]

        self.stage2 = self._make_stage(
            ShuffleNetWSUnit,
            num_blocks[0],
            out_channels[1],
            stride=2,
            stage=2,
            groups=groups,
            bias=bias,
            rho=rho
        )

        self.stage3 = self._make_stage(
            ShuffleNetWSUnit,
            num_blocks[1],
            out_channels[2],
            stride=2,
            stage=3,
            groups=groups,
            bias=bias,
            rho=rho
        )

        self.stage4 = self._make_stage(
            ShuffleNetWSUnit,
            num_blocks[2],
            out_channels[3],
            stride=2,
            stage=4,
            groups=groups,
            bias=bias,
            rho=rho
        )

        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(out_channels[3], num_classes)
        self.num_layers = 5


    def _make_stage(self, block, num_blocks, output_channels, stride, stage, groups, bias=False, rho=1e-3):

        strides = [stride] + [1] * (num_blocks - 1)
        stage = []
        for stride in strides:
            stage.append(
                block(
                    self.input_channels,
                    output_channels,
                    stride=stride,
                    stage=stage,
                    groups=groups,
                    bias=bias,
                    rho=rho
                )
            )
            self.input_channels = output_channels

        return nn.Sequential(*stage)


    def forward(self, x, return_feature = False, level = 0, *args ,**kwargs):
        if level<=0:
            out0 = self.conv1(x)
        else:
            out0 = x

        if level<=1:
            out1 = self.stage2(out0)
        else:
            out1 = out0

        if level<=2:
            out2 = self.stage3(out1)
        else:
            out2 = out1

        if level<=3:
            out3 = self.stage4(out2)
        else:
            out3 = out2

        x = self.avg(out3)
        x = x.view(x.size(0), -1)
        logit = self.fc(x)

        results = {}
        all_outputs = [out0,out1,out2,out3,logit]
        for idx, out in enumerate(all_outputs):
            if idx == len(all_outputs) - 1:
                results['logit'] = out
            if idx == len(all_outputs) - 2:
                results['feature'] = out           

            results['layer'+str(idx)]=out

        return results


