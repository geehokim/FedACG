from models.quant import *
from models.layers import norm

def AQD_update(model, args):

    for name, param in model.named_parameters():
        if hasattr(args.model, 'keyword'):
            if 'first-last' in args.model.keyword and name == 'conv1.weight':
                first_quant_conv = quant_conv(param.shape[0], param.shape[1], kernel_size=param.shape[2], args=args)
                param.data.copy_(first_quant_conv(param.data))
            elif name != "conv1.weight" and ("conv1.weight" in name or "conv2.weight" in name):
                layer_quant_conv = quant_conv(param.shape[0], param.shape[1], kernel_size=param.shape[2], args=args)
                param.data.copy_(layer_quant_conv(param.data))
            elif "downsample.0.weight" in name:
                quant_conv1x1 = nn.Sequential(
                    quant_conv(param.shape[0], param.shape[1], kernel_size=1, args=args),
                    norm(param.shape[1], args=args)
                )
                param.data.copy_(quant_conv1x1(param.data))
            elif 'first-last' in args.model.keyword and name == 'fc.weight':
                last_quant_linear = quant_linear(args.model.last_feature_dim, args.num_classes, bias=True, args=args)
                param.data.copy_(last_quant_linear(param.data))

