from utils import get_numclasses
from utils.registry import Registry
import models

TRAINER_REGISTRY = Registry("TRAINER")
TRAINER_REGISTRY.__doc__ = """
Registry for local updater
"""

__all__ = ['get_trainer_type']


def get_trainer_type(args):
    if args.verbose:
        print(TRAINER_REGISTRY)
    print("=> Creating trainer '{}'".format(args.trainer.type))
    trainer_type = TRAINER_REGISTRY.get(args.trainer.type)
    return trainer_type
