#!/usr/bin/env python
# coding: utf-8
import copy
import time

import matplotlib.pyplot as plt
import torch.multiprocessing as mp
from sklearn.manifold import TSNE

from utils import *
from utils.metrics import evaluate
from models import build_encoder
from typing import Callable, Dict, Tuple, Union, List

import wandb

from servers.build import SERVER_REGISTRY

@SERVER_REGISTRY.register()
class Server():

    def __init__(self, args):
        self.args = args
        return
    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, epoch=None):
        C = len(client_ids)
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key])/C
        return local_weights

@SERVER_REGISTRY.register()
class AnalizeServer():

    def __init__(self, args):
        self.args = args
        return
    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, epoch):
        C = len(client_ids)
        temp = {'scale/mean/conv1': [],
                'scale/mean/layer1': [],
                'scale/mean/layer2': [],
                'scale/mean/layer3': [],
                'scale/mean/layer4': [],
                'scale/std/conv1': [],
                'scale/std/layer1': [],
                'scale/std/layer2': [],
                'scale/std/layer3': [],
                'scale/std/layer4': [],
                'cos/mean/conv1': [],
                'cos/mean/layer1': [],
                'cos/mean/layer2': [],
                'cos/mean/layer3': [],
                'cos/mean/layer4': [],
                'cos/std/conv1': [],
                'cos/std/layer1': [],
                'cos/std/layer2': [],
                'cos/std/layer3': [],
                'cos/std/layer4': [],}
        from torch.nn import CosineSimilarity
        cos = CosineSimilarity(dim=1, eps=1e-10)
        for param_key in local_weights:
            if 'conv' in param_key:
                weight_list = local_weights[param_key]
                delta_list = local_deltas[param_key]
                for (w, d) in zip(weight_list, delta_list):
                    o, i, h_, w_ = w.size()
                    w_prev = w - d
                    # mean/std of cos(prev, curr)
                    # mean/std of std of curr
                    sim = cos(w.view(o, i * h_ * w_), w_prev.view(o, i *  h_ * w_)).abs()
                    sim_mean = sim.mean()
                    sim_var = sim.var()
                    
                    std = w.view(o, -1).std(dim=1)
                    std_mean = std.mean()
                    std_var = std.var()
                    
                    w_mean = w.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)        
                    w = (w - w_mean).view(o, -1)
                    
                    if param_key == 'conv1.weight':
                        temp['scale/mean/conv1'].append(std_mean.item())
                        temp['scale/std/conv1'].append(std_var.item())
                        temp['cos/mean/conv1'].append(sim_mean.item())
                        temp['cos/std/conv1'].append(sim_var.item())
                    elif 'layer1' in param_key:
                        temp['scale/mean/layer1'].append(std_mean.item())
                        temp['scale/std/layer1'].append(std_var.item())
                        temp['cos/mean/layer1'].append(sim_mean.item())
                        temp['cos/std/layer1'].append(sim_var.item())
                    elif 'layer2' in param_key:
                        temp['scale/mean/layer2'].append(std_mean.item())
                        temp['scale/std/layer2'].append(std_var.item())
                        temp['cos/mean/layer2'].append(sim_mean.item())
                        temp['cos/std/layer2'].append(sim_var.item())
                    elif 'layer3' in param_key:
                        temp['scale/mean/layer3'].append(std_mean.item())
                        temp['scale/std/layer3'].append(std_var.item())
                        temp['cos/mean/layer3'].append(sim_mean.item())
                        temp['cos/std/layer3'].append(sim_var.item())
                    elif 'layer4' in param_key:
                        temp['scale/mean/layer4'].append(std_mean.item())
                        temp['scale/std/layer4'].append(std_var.item())
                        temp['cos/mean/layer4'].append(sim_mean.item())
                        temp['cos/std/layer4'].append(sim_var.item())
                             
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key])/C
        
        for name in temp:
            temp[name] = np.mean(temp[name])
            if 'std' in name:
                temp[name] = np.sqrt(temp[name])
        
        # print(temp)
            
        wandb.log(temp, step=epoch)
        # print(temp_total)

        return local_weights
    

@SERVER_REGISTRY.register()
class ServerM(Server):    
    
    def set_momentum(self, model):

        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])

        self.global_delta = global_delta
        self.global_momentum = global_momentum

    @torch.no_grad()
    def FedACG_lookahead(self, model):
        sending_model_dict = copy.deepcopy(model.state_dict())
        for key in self.global_momentum.keys():
            if 'num_batches_tracked' in key:
                sending_model_dict[key] = self.global_momentum[key]
            else:
                sending_model_dict[key] += self.args.server.momentum * self.global_momentum[key]

        model.load_state_dict(sending_model_dict)
        return copy.deepcopy(model)
    

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, epoch=None):
        C = len(client_ids)
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key])/C
            
        if self.args.server.momentum>0:

            if not self.args.server.get('FedACG'): 
                for param_key in local_weights:               
                    local_weights[param_key] += self.args.server.momentum * self.global_momentum[param_key]
                    
            for param_key in local_deltas:
                self.global_delta[param_key] = sum(local_deltas[param_key])/C
                self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + self.global_delta[param_key]

        return local_weights


@SERVER_REGISTRY.register()
class ServerAdam(Server):    
    
    def set_momentum(self, model):

        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])

        global_v = copy.deepcopy(model.state_dict())
        for key in global_v.keys():
            global_v[key] = torch.zeros_like(global_v[key]) + (self.args.server.tau * self.args.server.tau)

        self.global_delta = global_delta
        self.global_momentum = global_momentum
        self.global_v = global_v
    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, epoch=None):
        C = len(client_ids)
        server_lr = self.args.trainer.global_lr
        
        for param_key in local_deltas:
            self.global_delta[param_key] = sum(local_deltas[param_key])/C
            self.global_momentum[param_key] = self.args.server.momentum * self.global_momentum[param_key] + (1-self.args.server.momentum) * self.global_delta[param_key]
            self.global_v[param_key] = self.args.server.beta * self.global_v[param_key] + (1-self.args.server.beta) * (self.global_delta[param_key] * self.global_delta[param_key])

        for param_key in model_dict.keys():
            model_dict[param_key] += server_lr *  self.global_momentum[param_key] / ( (self.global_v[param_key]**0.5) + self.args.server.tau)
            
        return model_dict

@SERVER_REGISTRY.register()
class ServerDyn(Server):    
    
    def set_momentum(self, model):
        #global_momentum is h^t in FedDyn paper
        global_delta = copy.deepcopy(model.state_dict())
        for key in global_delta.keys():
            global_delta[key] = torch.zeros_like(global_delta[key])

        global_momentum = copy.deepcopy(model.state_dict())
        for key in global_momentum.keys():
            global_momentum[key] = torch.zeros_like(global_momentum[key])

        self.global_delta = global_delta
        self.global_momentum = global_momentum

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr, epoch=None):
        C = len(client_ids)
        for param_key in self.global_momentum:
            self.global_momentum[param_key] -= self.args.client.Dyn.alpha / self.args.trainer.num_clients * sum(local_deltas[param_key])
            local_weights[param_key] = sum(local_weights[param_key])/C - 1/self.args.client.Dyn.alpha * self.global_momentum[param_key]
        return local_weights
