from utils import get_numclasses
from utils.registry import Registry
import models

EVALER_REGISTRY = Registry("EVALER")
EVALER_REGISTRY.__doc__ = """
Registry for local updater
"""

__all__ = ['get_evaler_type']


def get_evaler_type(args):
    if args.verbose:
        print(EVALER_REGISTRY)
    print("=> Creating evaler '{}'".format(args.evaler.type))
    evaler_type = EVALER_REGISTRY.get(args.evaler.type)
    return evaler_type


# def build_evaler(args, datasets):
#     trainer_type = get_evaler_type(args)
#     trainer = trainer_type(args)
#     return trainer
