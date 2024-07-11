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


from servers.build import SERVER_REGISTRY

@SERVER_REGISTRY.register()
class Server():

    def __init__(self, args):
        self.args = args
        return
    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        C = len(client_ids)
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key])/C
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
            sending_model_dict[key] += self.args.server.momentum * self.global_momentum[key]

        model.load_state_dict(sending_model_dict)
        return copy.deepcopy(model)
    

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
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
    
    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
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

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        C = len(client_ids)
        for param_key in self.global_momentum:
            self.global_momentum[param_key] -= self.args.client.Dyn.alpha / self.args.trainer.num_clients * sum(local_deltas[param_key])
            local_weights[param_key] = sum(local_weights[param_key])/C - 1/self.args.client.Dyn.alpha * self.global_momentum[param_key]
        return local_weights
    
    
@SERVER_REGISTRY.register()
class ServerWSV1(Server):    

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        # Ver. 1
        for param_key in local_weights.keys():
            if ('conv' in param_key) or ('downsample.0' in param_key):
                weight_stack = torch.stack(local_weights[param_key])
                local_weights[param_key] = self._standardize_repair_each_client_conv(weight_stack)
                    
            if 'fc.weight' in param_key:
                weight_stack = torch.stack(local_weights[param_key])
                local_weights[param_key] = self._standardize_repair_each_client_linear(weight_stack)

        C = len(client_ids)
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key])/C

        return local_weights
        
    def _standardize_repair_each_client_conv(self, conv_weight_stack): # conv_weight_stack: stacked tensor
        weight_each_mean = conv_weight_stack.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight_each = conv_weight_stack - weight_each_mean
        weight_each_mean = weight_each_mean.mean(dim=0, keepdim=True)
        weight_total = conv_weight_stack - weight_each_mean
        nc = conv_weight_stack.size(0)
        c_out = conv_weight_stack.size(1)
        std_each = weight_each.view(nc, c_out, -1).std(dim=-1).view(nc, c_out, 1, 1, 1) + 1e-5
        std_total = weight_total.transpose(0, 1).contiguous().view(c_out, -1).std(dim=-1).view(1, c_out, 1, 1, 1) + 1e-5
        return weight_each / std_each.expand_as(weight_each) * std_total.expand_as(weight_each) + weight_each_mean.expand_as(weight_each)

    def _standardize_repair_each_client_linear(self, linear_weight_stack): # conv_weight_stack: stacked tensor
        weight_each_mean = linear_weight_stack.mean(dim=2, keepdim=True)
        weight_each = linear_weight_stack - weight_each_mean
        weight_each_mean = weight_each_mean.mean(dim=0, keepdim=True)
        weight_total = linear_weight_stack - weight_each_mean
        nc = linear_weight_stack.size(0)
        c_out = linear_weight_stack.size(1)
        std_each = weight_each.view(nc, c_out, -1).std(dim=-1).view(nc, c_out, 1) + 1e-5
        std_total = weight_total.transpose(0, 1).contiguous().view(c_out, -1).std(dim=-1).view(1, c_out, 1) + 1e-5
        return weight_each / std_each.expand_as(weight_each) * std_total.expand_as(weight_each) + weight_each_mean.expand_as(weight_each)


@SERVER_REGISTRY.register()
class ServerWSV2(Server):    

    def aggregate(self, local_weights, local_deltas, client_ids, model_dict, current_lr):
        # Ver. 2
        for param_key in local_weights.keys():
            if ('conv' in param_key) or ('downsample.0' in param_key):
                weight_stack = torch.stack(local_weights[param_key])
                local_weights[param_key] = self._standardize_repair_each_client_conv(weight_stack)

        C = len(client_ids)
        for param_key in local_weights:
            local_weights[param_key] = sum(local_weights[param_key])/C

        return local_weights
        
    def _standardize_each_client_conv(self, conv_weight_stack): # conv_weight_stack: stacked tensor
        weight_each_mean = conv_weight_stack.mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight_each = conv_weight_stack - weight_each_mean
        nc = conv_weight_stack.size(0)
        c_out = conv_weight_stack.size(1)
        std_each = weight_each.view(nc, c_out, -1).std(dim=-1).view(nc, c_out, 1, 1, 1) + 1e-5
        return weight_each / std_each.expand_as(weight_each)
