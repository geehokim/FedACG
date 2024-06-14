import torch
import copy
from torch.utils.data import DataLoader
import numpy as np
import random


__all__ = ['initalize_random_seed', 'terminate_processes']

def terminate_processes(queues, processes):
    # Signal all processes to exit by putting None in each queue
    for queue in queues:
        queue.put(None)

    # Wait for all processes to finish
    for p in processes:
        p.terminate()

def initalize_random_seed(args):
    random_seed = args.seed
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    # torch.backends.cudnn.enabled = True
    if args.enable_benchmark:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)


