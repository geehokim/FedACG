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



@EVALER_REGISTRY.register()
class Evaler():

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
    def eval(self, model: nn.Module, epoch: int, device: torch.device = None, **kwargs) -> Dict:

        model.eval()
        model_device = next(model.parameters()).device
        if device is None:
            device = self.device
        model.to(device)
        loss, correct, total = 0, 0, 0

        if type(self.test_loader.dataset) == DatasetSplit:
            C = len(self.test_loader.dataset.dataset.classes)
        else:
            C = len(self.test_loader.dataset.classes)

        class_loss, class_correct, class_total = torch.zeros(C), torch.zeros(C), torch.zeros(C)

        logits_all, labels_all = [], []


        with torch.no_grad():
            # for images, labels in self.loaders["test"]:
            for idx, (images, labels) in enumerate(self.test_loader):
                images, labels = images.to(device), labels.to(device)

                results = model(images)
                _, predicted = torch.max(results["logit"].data, 1) # if errors occur, use ResNet18_base instead of ResNet18_GFLN
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                bin_labels = labels.bincount()
                class_total[:bin_labels.size(0)] += bin_labels.cpu()
                bin_corrects = labels[(predicted == labels)].bincount()
                class_correct[:bin_corrects.size(0)] += bin_corrects.cpu()

                this_loss = self.criterion(results["logit"], labels)
                loss += this_loss.sum().cpu()

                for class_idx, bin_label in enumerate(bin_labels):
                    class_loss[class_idx] += this_loss[(labels.cpu() == class_idx)].sum().cpu()

                logits_all.append(results["logit"].data.cpu())
                labels_all.append(labels.cpu())

        logits_all = torch.cat(logits_all)
        labels_all = torch.cat(labels_all)

        scores = F.softmax(logits_all, 1)

        acc = 100. * correct / float(total)
        class_acc = 100. * class_correct / class_total
        
        loss = loss / float(total)
        class_loss = class_loss / class_total

        model.train()
        results = {
            "acc": acc,
            'class_acc': class_acc,
            'loss': loss,
            'class_loss' : class_loss,
        }
        
        return results
