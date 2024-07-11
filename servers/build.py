from utils import get_numclasses
from utils.registry import Registry
import models

SERVER_REGISTRY = Registry("SERVER")
SERVER_REGISTRY.__doc__ = """
Registry for local updater
"""

__all__ = ['get_server_type', 'build_server']


def get_server_type(args):
    if args.verbose:
        print(SERVER_REGISTRY)
    print("=> Getting server type '{}'".format(args.server.type))
    server_type = SERVER_REGISTRY.get(args.server.type)
    
    return server_type

def build_server(args):
    server_type = get_server_type(args)
    server = server_type(args)
    return server
