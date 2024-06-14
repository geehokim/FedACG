from utils import get_numclasses
from utils.registry import Registry
import models

CLIENT_REGISTRY = Registry("CLIENT")
CLIENT_REGISTRY.__doc__ = """
Registry for local updater
"""

__all__ = ['get_client_type', 'get_client_type_compare']


def get_client_type(args):
    if args.verbose:
        print(CLIENT_REGISTRY)
    print("=> Getting client type '{}'".format(args.client.type))
    client_type = CLIENT_REGISTRY.get(args.client.type)
    return client_type


def get_client_type_compare(args):
    if args.verbose:
        print(CLIENT_REGISTRY)
    print("=> Getting client_compare type '{}'".format(args.client_compare.type))
    client_type = CLIENT_REGISTRY.get(args.client_compare.type)
    return client_type

def build_client(args):
    return
