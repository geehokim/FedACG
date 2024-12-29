#!/usr/bin/env python
# coding: utf-8
import copy
import time

import matplotlib.pyplot as plt
import torch.multiprocessing as mp
import gc

from utils import *
from utils.metrics import evaluate
from models import build_encoder
from typing import Callable, Dict, Tuple, Union, List
from utils.logging_utils import AverageMeter
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

import logging
logger = logging.getLogger(__name__)

from clients.build import CLIENT_REGISTRY

from utils.qunat_function import AQD_update , PAQ_update, WSQ_update, WLQ_update

@CLIENT_REGISTRY.register()
class Client():

    def __init__(self, args, client_index, model=None, loader=None):
        self.args = args
        self.client_index = client_index
        self.model = model
        self.global_model =  copy.deepcopy(model)
        for par in self.global_model.parameters():
            par.requires_grad = False

        self.criterion = nn.CrossEntropyLoss()
        if self.args.client.get('LC'):
            self.FedLC_criterion = FedLC
        self.decorr_criterion = FedDecorrLoss()

        return

    def setup(self, state_dict, device, local_dataset, local_lr, global_epoch, trainer, **kwargs):
        self._update_model(state_dict)
        self._update_global_model(state_dict)
        self.device = device

        if self.args.dataset.num_instances > 0:
            train_sampler = RandomClasswiseSampler(local_dataset, num_instances=self.args.dataset.num_instances)   
        else:
            train_sampler = None

        self.loader =  DataLoader(local_dataset, batch_size=self.args.batch_size, sampler=train_sampler, shuffle=train_sampler is None,
                                   num_workers=self.args.num_workers, pin_memory=self.args.pin_memory)
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=local_lr, momentum=self.args.optimizer.momentum,
                                   weight_decay=self.args.optimizer.wd)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer=self.optimizer, 
                                                     lr_lambda=lambda epoch: self.args.trainer.local_lr_decay ** epoch)
    
        self.trainer = trainer
        self.num_layers = self.model.num_layers
        self.class_counts = np.sort([*local_dataset.class_dict.values()])[::-1]
        if global_epoch == 0:
            logger.info(f"Class counts : {self.class_counts}")

        #For FedLC
        if self.args.client.get('LC'):
            self.class_stats = {
            'ratio': None,
            }
            self.num_classes = len(self.loader.dataset.dataset.classes)
            self.class_stats['ratio'] = torch.zeros(self.num_classes)
            for class_key in local_dataset.class_dict:
                self.class_stats['ratio'][int(class_key)] = local_dataset.class_dict[class_key]
            sorted_key = np.sort([*local_dataset.class_dict.keys()])
            sorted_class_dict = {} 
            for key in sorted_key:  
                sorted_class_dict[key] = local_dataset.class_dict[key]
            self.label_distrib = torch.zeros(len(local_dataset.dataset.classes), device=self.device)
            for key in sorted_class_dict:
                self.label_distrib[int(key)] = sorted_class_dict[key]

        #For FedDyn
        if self.args.client.get('Dyn'):
            self.local_deltas = (kwargs['past_local_deltas'])
            self.user = kwargs['user']
            self.local_delta = copy.deepcopy(self.local_deltas[self.user])

    def _update_model(self, state_dict):
        self.model.load_state_dict(state_dict)

    def _update_global_model(self, state_dict):
        self.global_model.load_state_dict(state_dict)

    def __repr__(self):
        print(f'{self.__class__} {self.client_index}, {"data : "+len(self.loader.dataset) if self.loader else ""}')

    def get_weights(self, epoch=None):

        weights = {
            "cls": self.args.client.ce_loss.weight,
        }
        if self.args.client.get('prox_loss'):
            weights['prox'] = self.args.client.prox_loss.weight

        if self.args.client.get('LC'):
            weights['LC'] = self.args.client.LC.weight

        if self.args.client.get('decorr_loss'):
            weights['decorr'] = self.args.client.decorr_loss.weight

        if self.args.client.get('MLB'):
            weights['MLB_branch_cls'] = self.args.client.MLB.branch_cls_weight
            weights['MLB_branch_kl'] = self.args.client.MLB.branch_kl_weight

        if self.args.client.get('NTD'):
            weights['NTD'] = self.args.client.NTD.weight
            weights["cls"] = 1-self.args.client.NTD.weight

        if self.args.client.get('Dyn'):
            weights['Dyn'] = self.args.client.Dyn.weight
        return weights

    def local_train(self, global_epoch, **kwargs):
        self.global_epoch = global_epoch

        self.model.to(self.device)
        self.global_model.to(self.device)
        scaler = GradScaler()
        start = time.time()
        loss_meter = AverageMeter('Loss', ':.2f')
        time_meter = AverageMeter('BatchTime', ':3.2f')

        self.weights = self.get_weights(epoch=global_epoch)

        if global_epoch % 50 == 0:
            print(self.weights)

        for local_epoch in range(self.args.trainer.local_epochs):
            end = time.time()

            for i, (images, labels) in enumerate(self.loader):
                    
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad(set_to_none=True)

                with autocast(enabled=self.args.use_amp):
                    losses = self._algorithm(images, labels)
                    # for loss_key in losses:
                    #     if loss_key not in self.weights.keys():
                    #         self.weights[loss_key] = 0
                    loss = sum([self.weights[loss_key]*losses[loss_key] for loss_key in losses])

                try:
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                    scaler.step(self.optimizer)
                    scaler.update()

                except Exception as e:
                    print(e)

                loss_meter.update(loss.item(), images.size(0))
                time_meter.update(time.time() - end)
                end = time.time()

            self.scheduler.step()
        
        logger.info(f"[C{self.client_index}] End. Time: {end-start:.4f}s, Loss: {loss_meter.avg:.3f}")

        self.model.to('cpu')
        self.global_model.to('cpu')
        torch.cuda.empty_cache()
        gc.collect()
        
        # AQD
        if self.args.quantizer.uplink:
            if self.args.quantizer.name == "AQD":
                AQD_update(self.model, self.args)
            elif self.args.quantizer.name == "WSQ":
                WSQ_update(self.model, self.args)
            elif self.args.quantizer.name == "PAQ":
                PAQ_update(self.model, self.global_model, self.args)
            elif self.args.quantizer.name == "WLQ":
                WLQ_update(self.model, self.args)
        
        loss_dict = {
            f'loss/{self.args.dataset.name}': loss_meter.avg,
        }

        if self.args.client.get('Dyn'):
            with torch.no_grad():
                fixed_params = {n:p for n,p in self.global_model.named_parameters()}
                for n, p in self.model.named_parameters():
                    self.local_deltas[self.user][n] = (self.local_delta[n] - self.args.client.Dyn.alpha * (p - fixed_params[n]).detach().clone().to('cpu'))
            del fixed_params      

        return self.model.state_dict(), loss_dict

    def _algorithm(self, images, labels, ) -> Dict:
        losses = defaultdict(float)

        results = self.model(images)
        cls_loss = self.criterion(results["logit"], labels)
        losses["cls"] = cls_loss
        ## Weight L2 loss
        if self.args.client.get('prox_loss'):
            prox_loss = 0
            fixed_params = {n:p for n,p in self.global_model.named_parameters()}
            for n, p in self.model.named_parameters():
                prox_loss += ((p-fixed_params[n].detach())**2).sum()  
            losses["prox"] = prox_loss

        #FedLC
        if self.args.client.get('LC'):
            LC_loss = self.FedLC_criterion(self.label_distrib, results["logit"], labels, self.args.client.LC.tau)
            losses["LC"] = LC_loss

        #FedDecorr
        if self.args.client.get('decorr_loss'):
            decorr_loss = self.decorr_criterion(results["feature"])
            losses["decorr"] = decorr_loss
        
        #FedMLB
        if self.args.client.get('MLB'):
            MLB_args = self.args.client.MLB
            cls_branch = []
            kl_branch = []

            for l in range(self.num_layers):
                if l in MLB_args.branch_level:         
                    global_results = self.global_model(results[f"layer{l}"], mlb_level=l+1)
                    cls_branch.append(self.criterion(global_results["logit"], labels))
                    kl_branch.append(KD(global_results["logit"], results["logit"], T=MLB_args.Temp))
            losses["MLB_branch_cls"] = sum(cls_branch)/len(cls_branch)
            losses["MLB_branch_kl"] = sum(kl_branch)/len(kl_branch)
            del global_results

        #FedNTD
        if self.args.client.get("NTD"):
            with torch.no_grad():
                global_results = self.global_model(images)
            batch_size = labels.size(0)
            idxs = torch.arange(results['logit'].size(1)).unsqueeze(0).repeat(batch_size, 1)
            not_true_idx = idxs.to(self.device) != labels.unsqueeze(1)
            not_true_logits = results['logit'][not_true_idx].view(batch_size, results['logit'].size(1) - 1)
            not_true_logits_global = global_results['logit'][not_true_idx].view(batch_size, results['logit'].size(1) - 1)
            losses["NTD"] = KD(not_true_logits_global, not_true_logits, T=self.args.client.NTD.Temp)
            del global_results, not_true_idx, not_true_logits, not_true_logits_global

        #FedDyn
        if self.args.client.get('Dyn'):
            lg_loss = 0
            for n, p in self.model.named_parameters():
                p = torch.flatten(p)
                local_d = self.local_delta[n].detach().clone().to(self.device)
                local_grad = torch.flatten(local_d)
                lg_loss += (p * local_grad.detach()).sum()
            losses["Dyn"] = - lg_loss + 0.5 * self.args.client.Dyn.alpha * prox_loss
            
        del results
        torch.cuda.empty_cache()
        gc.collect()
        return losses
