from pathlib import Path
from typing import Callable, Dict, Tuple, Union, List, Type
from argparse import Namespace
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import tqdm
import wandb
import gc

import pickle, os
import numpy as np

import logging
logger = logging.getLogger(__name__)


import time, io, copy

from evalers.build import EVALER_REGISTRY

from servers import Server
from clients import Client

from utils import DatasetSplit, get_dataset
from utils.logging_utils import AverageMeter

from torch.utils.data import DataLoader

from utils import terminate_processes, initalize_random_seed
from omegaconf import DictConfig

from sklearn import metrics
import matplotlib.pyplot as plt

import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from evalers.base_evaler import Evaler


@EVALER_REGISTRY.register()
class CKAEvaler(Evaler):

    def __init__(self,
                 test_loader: torch.utils.data.DataLoader,
                device: torch.device,
                args: DictConfig,
                gallery_loader: torch.utils.data.DataLoader = None,
                query_loader: torch.utils.data.DataLoader = None,
                distance_metric: str = 'cosine',
                **kwargs) -> None:

        self.args = args
        self.device = device

        self.test_loader = test_loader
        self.gallery_loader = gallery_loader
        self.query_loader = query_loader
        self.criterion = nn.CrossEntropyLoss(reduction = 'none')


    @torch.no_grad()
    def eval(self, model_list: list, device: torch.device = None, **kwargs) -> Dict:
        
        num_models = len(model_list)
        model_device = next(model_list[0].parameters()).device
        if device is None:
            device = self.device
        for m in model_list:
            m.eval()
            m.to(device)

        if type(self.test_loader.dataset) == DatasetSplit:
            C = len(self.test_loader.dataset.dataset.classes)
        else:
            C = len(self.test_loader.dataset.classes)
            
        out_cka_matrix = None
        num_iters = 0

        with torch.no_grad():
            # for images, labels in self.loaders["test"]:
            for idx, (images, labels) in enumerate(self.test_loader):
                
                images, labels = images.to(device), labels.to(device)
                batch_size = images.size(0)

                feature_list = [m(images)["layer4"].data.view(batch_size, -1) for m in model_list]
                feats = torch.stack(feature_list, dim=0)    # (num_models, num_batch, num_feat)
                cka_matrix = self.linear_CKA(feats)
                
                if out_cka_matrix is None:
                    out_cka_matrix = cka_matrix
                else:
                    out_cka_matrix += cka_matrix
                num_iters += 1
        
        out_cka_matrix = (out_cka_matrix / num_iters).cpu().numpy()
        
        for m in model_list:
            m.eval()

        results = {
            "cka_matrix": out_cka_matrix,
        }
        
        return results
    
    def gram(self, X):
        # X: (num_models, num_batch, num_feats)
        return torch.matmul(X, X.transpose(2, 1))
    
    def centering(self, K):
        num_models, num_batch, _ = K.size()   # (num_models, num_batch, num_batch)
        device = K.device
        unit = K.new_ones((num_models, num_batch, num_batch))
        I = torch.eye(num_batch, device=device).unsqueeze(0)
        H = I - unit / num_batch    # (num_models, num_batch, num_batch)
        return torch.matmul(torch.matmul(H, K), H) # (num_models, num_batch, num_batch)

    def cross_HSIC(self, centered_L):
        # centered_K, centered_L: (num_models, num_batch, num_batch)
        centered_K = centered_L.unsqueeze(1)    # (num_models, 1, num_batch, num_batch)
        return (centered_K * centered_L).sum(-1).sum(-1)  # (num_models, num_models)

    def linear_CKA(self, X):
        # X: (num_models, num_batch, num_feats)
        K = self.centering(self.gram(X))    # (num_models, num_batch, num_batch)
        
        cross_hsic = self.cross_HSIC(K)   # (num_models, num_models)
        self_hsic = torch.sqrt(torch.diagonal(cross_hsic)).unsqueeze(-1)  # (num_models, 1)
        cross_var = torch.matmul(self_hsic, self_hsic.T) + 1e-8 # (num_models, num_models)

        return cross_hsic / cross_var
