from utils import get_numclasses
from utils.registry import Registry
import models

ENCODER_REGISTRY = Registry("ENCODER")
ENCODER_REGISTRY.__doc__ = """
Registry for encoder
"""

__all__ = ['get_model', 'build_encoder']

# def get_model(args,trainset = None):
#     num_classes = get_numclasses(args, trainset)
#     print("=> Creating model '{}'".format(args.arch))
#     print("Model Option")
#     print(" 1) use_pretrained =", args.use_pretrained)
#     print(" 2) No_transfer_learning =", args.No_transfer_learning)
#     print(" 3) use_bn =", args.use_bn)
#     print(" 4) use_pre_fc =", args.use_pre_fc)
#     print(" 5) use_bn_layer =", args.use_bn_layer)
#     model = models.__dict__[args.arch](args, num_classes=num_classes, l2_norm=args.l2_norm, use_pretrained = args.use_pretrained, transfer_learning = not(args.No_transfer_learning), use_bn = args.use_bn, use_pre_fc = args.use_pre_fc, use_bn_layer = args.use_bn_layer)
#     #model = models.__dict__[args.arch](num_classes=num_classes, l2_norm=args.l2_norm, use_pretrained = args.use_pretrained, transfer_learning = not(args.No_transfer_learning), use_bn = args.use_bn, use_pre_fc = args.use_pre_fc, use_bn_layer = args.use_bn_layer)
#     return model

def build_encoder(args):

    num_classes = get_numclasses(args)

    if args.verbose:
        print(ENCODER_REGISTRY)

    print(f"=> Creating model '{args.model.name}, pretrained={args.model.pretrained}'")
    
    encoder = ENCODER_REGISTRY.get(args.model.name)(args, num_classes, **args.model) if len(args.model.name) > 0 else None

    return encoder
