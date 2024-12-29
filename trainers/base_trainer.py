from pathlib import Path
from typing import Callable, Dict, Tuple, Union, List, Type, Any
from argparse import Namespace
from collections import defaultdict

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import tqdm
import wandb
import gc
import psutil

import pickle, os
import numpy as np

import logging
logger = logging.getLogger(__name__)

import time, io, copy

from trainers.build import TRAINER_REGISTRY

from servers import Server
from clients import Client

from utils import DatasetSplit, DatasetSplitSubset, get_dataset
from utils.logging_utils import AverageMeter

from torch.utils.data import DataLoader

from utils import terminate_processes, initalize_random_seed, save_checkpoint
from omegaconf import DictConfig,OmegaConf


#from netcal.metrics import ECE
import matplotlib.pyplot as plt

from utils.qunat_function import AQD_update, WSQ_update, WLQ_update


@TRAINER_REGISTRY.register()
class Trainer():

    def __init__(self,
                 model: nn.Module,
                 client_type: Type,
                 server: Server,
                 evaler_type: Type,
                 datasets: Dict,
                 device: torch.device,
                 args: DictConfig,
                 multiprocessing: Dict = None,
                 **kwargs) -> None:

        self.args = args
        self.device = device
        self.model = model

        self.checkpoint_path = Path(self.args.checkpoint_path)
        mode = self.args.split.mode 
        if self.args.split.mode == 'dirichlet':
            mode += str(self.args.split.alpha)
        self.exp_path = self.checkpoint_path / self.args.dataset.name / mode / self.args.exp_name
        logger.info(f"Exp path : {self.exp_path}")

        ### training config
        trainer_args = self.args.trainer
        self.num_clients = trainer_args.num_clients
        self.participation_rate = trainer_args.participation_rate
        self.global_rounds = trainer_args.global_rounds
        self.lr = trainer_args.local_lr
        self.local_lr_decay = trainer_args.local_lr_decay

        self.clients: List[Client] = [client_type(self.args, client_index=c, model=copy.deepcopy(self.model)) for c in range(self.args.trainer.num_clients)]
        self.server = server
        if self.args.server.momentum > 0 or self.args.client.get('Dyn'):
            self.server.set_momentum(self.model)

        self.datasets = datasets
        self.local_dataset_split_ids = get_dataset(self.args, self.datasets['train'], mode=self.args.split.mode)

        test_loader = DataLoader(self.datasets["test"],
                                batch_size=args.evaler.batch_size if args.evaler.batch_size > 0 else args.batch_size,
                                shuffle=False, num_workers=args.num_workers)
        eval_device = self.device if not self.args.multiprocessing else torch.device(f'cuda:{self.args.main_gpu}')
        eval_params = {
            "test_loader": test_loader,
            "device": eval_device,
            "args": args,
        }
        self.eval_params = eval_params
        self.eval_device = eval_device
        self.evaler = evaler_type(**eval_params)
        logger.info(f"Trainer: {self.__class__}, client: {client_type}, server: {server.__class__}, evaler: {evaler_type}")

        self.start_round = 0
        if self.args.get('load_model_path'):
            self.load_model()

        if self.args.client.get('Dyn'):
            local_g = copy.deepcopy(self.model.state_dict())
            for key in local_g.keys():
                local_g[key] = torch.zeros_like(local_g[key]).to('cpu')
            self.past_local_deltas = {net_i: copy.deepcopy(local_g) for net_i in range(self.num_clients)}


    def local_update(self, device, task_queue, result_queue):
        if self.args.multiprocessing:
            torch.cuda.set_device(device)
            initalize_random_seed(self.args)

        while True:
            task = task_queue.get()
            
            if task is None:
                break
            
            client = self.clients[task['client_idx']]

            local_dataset = DatasetSplitSubset(
                self.datasets['train'],
                idxs=self.local_dataset_split_ids[task['client_idx']],
                subset_classes=self.args.dataset.get('subset_classes'),
                )

            setup_inputs = {
                'state_dict': task['state_dict'],
                'device': device,
                'local_dataset': local_dataset,
                'local_lr': task['local_lr'],
                'global_epoch': task['global_epoch'],
                'trainer': self,
            }

            if self.args.client.get('Dyn'):
                setup_inputs['past_local_deltas'] = self.past_local_deltas
                setup_inputs['user'] = task['client_idx']

            client.setup(**setup_inputs)
            # Local Training
            local_model, local_loss_dict = client.local_train(global_epoch=task['global_epoch'])
            result_queue.put((local_model, local_loss_dict))

            if not self.args.multiprocessing:
                break


    def train(self) -> Dict:

        result_queue = mp.Manager().Queue()

        M = max(int(self.participation_rate * self.num_clients), 1)

        if self.args.multiprocessing:
            ngpus_per_node = torch.cuda.device_count()
            task_queues = [mp.Queue() for _ in range(M)]
            processes = [mp.get_context('spawn').Process(target=self.local_update, args=(
                i % ngpus_per_node, task_queues[i], result_queue)) for i in range(M)]
            # target: target client process
            # args = (GPU ID, client's task queue, server queue)

            # start all processes
            for p in processes:
                p.start()
                
        for epoch in range(self.start_round, self.global_rounds):
            
            self.lr_update(epoch=epoch)
            current_lr = self.lr

            # AQD
            if self.args.quantizer.downlink:
                if self.args.quantizer.name == "AQD":
                    AQD_update(self.model, self.args)
                elif self.args.quantizer.name == "WSQ":
                    WSQ_update(self.model, self.args)
                elif self.args.quantizer.name == "WLQ":
                    WLQ_update(self.model, self.args)

            # Global model
            global_state_dict = copy.deepcopy(self.model.state_dict())
            
            # Select clients
            if self.participation_rate < 1.:
                selected_client_ids = np.random.choice(range(self.num_clients), M, replace=False)
            else:
                selected_client_ids = range(len(self.clients))
            logger.info(f"Global epoch {epoch}, Selected client : {selected_client_ids}")

            local_weights = defaultdict(list)
            local_loss_dicts = defaultdict(list)
            local_deltas = defaultdict(list)

            # FedACG lookahead momentum
            if self.args.server.get('FedACG'):
                assert(self.args.server.momentum > 0)
                self.model= copy.deepcopy(self.server.FedACG_lookahead(copy.deepcopy(self.model)))
                global_state_dict = copy.deepcopy(self.model.state_dict())
            
            # Client-side
            start = time.time()
            for i, client_idx in enumerate(selected_client_ids):
                task_queue_input = {
                    'state_dict': self.model.state_dict(),
                    'client_idx': client_idx,
                    'local_lr': current_lr,
                    'global_epoch': epoch,
                }
                if self.args.multiprocessing:
                    task_queues[i].put(task_queue_input)
                else:
                    task_queue = mp.Queue()
                    task_queue.put(task_queue_input)
                    self.local_update(self.device, task_queue, result_queue)

                    local_state_dict, local_loss_dict = result_queue.get()
                    for loss_key in local_loss_dict:
                        local_loss_dicts[loss_key].append(local_loss_dict[loss_key])

                    for param_key in local_state_dict:
                        local_weights[param_key].append(local_state_dict[param_key])
                        local_deltas[param_key].append(local_state_dict[param_key] - global_state_dict[param_key])

            if self.args.multiprocessing:
                for _ in range(len(selected_client_ids)):
                    # Retrieve results from the queue
                    result = result_queue.get()
                    local_state_dict, local_loss_dict = result
                    for loss_key in local_loss_dict:
                        local_loss_dicts[loss_key].append(local_loss_dict[loss_key])

                    # If you want to save gpu memory, make sure that weights are not allocated to GPU
                    for param_key in local_state_dict:
                        local_weights[param_key].append(local_state_dict[param_key])
                        local_deltas[param_key].append(local_state_dict[param_key] - global_state_dict[param_key])
            
            logger.info(f"Global epoch {epoch}, Train End. Total Time: {time.time() - start:.2f}s")

            updated_global_state_dict = self.server.aggregate(local_weights, local_deltas,
                                                            selected_client_ids, copy.deepcopy(global_state_dict), current_lr, 
                                                            epoch=epoch if self.args.server.get('AnalizeServer') else None)

            self.model.load_state_dict(updated_global_state_dict)

            if self.args.eval.freq > 0 and epoch % self.args.eval.freq == 0:
                self.evaluate(epoch=epoch)

            if (self.args.save_freq > 0 and (epoch + 1) % self.args.save_freq == 0) or (epoch + 1 == self.args.trainer.global_rounds):
                self.save_model(epoch=epoch)

            # Logging
            wandb_dict = {loss_key: np.mean(local_loss_dicts[loss_key]) for loss_key in local_loss_dicts}
            wandb_dict['lr'] = self.lr

            self.wandb_log(wandb_dict, step=epoch)

            # Memory clean up
            del local_weights, local_loss_dicts, local_deltas
            torch.cuda.empty_cache()
            gc.collect()

        if self.args.multiprocessing:
            # Terminate Processes
            terminate_processes(task_queues, processes)

        return

    def lr_update(self, epoch: int) -> None:
        if self.global_rounds == 1000:
            exponent = epoch
        elif self.global_rounds < 1000:
            exponent = epoch * (1000 // self.global_rounds)
        else:
            exponent = epoch // (self.global_rounds // 1000)
        
        self.lr = self.args.trainer.local_lr * (self.local_lr_decay) ** (exponent)
        return

    def save_model(self, epoch: int = -1, suffix: str = '') -> None:
        
        model_path = self.exp_path / self.args.output_model_path
        if not model_path.parent.exists():
            model_path.parent.mkdir(parents=True, exist_ok=True)

        if epoch < self.args.trainer.global_rounds - 1:
            model_path = Path(f"{model_path}.e{epoch+1}")

        if suffix:
            model_path = Path(f"{model_path}.{suffix}")
        
        save_checkpoint(self.model, model_path, epoch, save_torch=True, use_breakpoint=False)      
        print(f'Saved model at {model_path}')  
        return
    

    def load_model(self) -> None:
        if self.args.get('load_model_path'):
            saved_dict = torch.load(self.args.load_model_path)
            self.model.load_state_dict(saved_dict['model_state_dict'], strict=False)
            self.start_round = saved_dict["epoch"]+1
            logger.warning(f'Load model from {self.args.load_model_path}, epoch {saved_dict["epoch"]}')
            
        return

    def wandb_log(self, log: Dict, step: int = None):
        if self.args.wandb:
            wandb.log(log, step=step)

    def validate(self, epoch: int, ) -> Dict:
        return

    def evaluate(self, epoch: int) -> Dict:

        results = self.evaler.eval(model=copy.deepcopy(self.model), epoch=epoch)
        acc = results["acc"]

        wandb_dict = {
            f"acc/{self.args.dataset.name}": acc,
            }

        logger.warning(f'[Epoch {epoch}] Test Accuracy: {acc:.2f}%')

        plt.close()
        
        self.wandb_log(wandb_dict, step=epoch)
        return {
            "acc": acc
        }


@TRAINER_REGISTRY.register()
class CKATrainer(Trainer):

    def train(self) -> Dict:

        result_queue = mp.Manager().Queue()

        M = max(int(self.participation_rate * self.num_clients), 1)

        if self.args.multiprocessing:
            ngpus_per_node = torch.cuda.device_count()
            task_queues = [mp.Queue() for _ in range(M)]
            processes = [mp.get_context('spawn').Process(target=self.local_update, args=(
                i % ngpus_per_node, task_queues[i], result_queue)) for i in range(M)]

            # start all processes
            for p in processes:
                p.start()
                
        # # FedWS lookahead init
        # if self.args.server.get('FedWS'):
        #     self.model= copy.deepcopy(self.server.FedWS_init_norm(copy.deepcopy(self.model)))

        for epoch in range(self.start_round, self.global_rounds):
            
            self.lr_update(epoch=epoch)

            global_state_dict = copy.deepcopy(self.model.state_dict())
            
            # Select clients
            if self.participation_rate < 1.:
                selected_client_ids = np.random.choice(range(self.num_clients), M, replace=False)
            else:
                selected_client_ids = range(len(self.clients))
            logger.info(f"Global epoch {epoch}, Selected client : {selected_client_ids}")

            current_lr = self.lr

            local_weights = defaultdict(list)
            local_loss_dicts = defaultdict(list)
            local_deltas = defaultdict(list)

            local_models = []

            # FedACG lookahead momentum
            if self.args.server.get('FedACG'):
                assert(self.args.server.momentum > 0)
                self.model= copy.deepcopy(self.server.FedACG_lookahead(copy.deepcopy(self.model)))
                global_state_dict = copy.deepcopy(self.model.state_dict())

            # Client-side
            start = time.time()
            for i, client_idx in enumerate(selected_client_ids):
                task_queue_input = {
                    'state_dict': self.model.state_dict(),
                    'client_idx': client_idx,
                    'local_lr': current_lr,
                    'global_epoch': epoch,
                }
                if self.args.multiprocessing:
                    task_queues[i].put(task_queue_input)
                else:
                    task_queue = mp.Queue()
                    task_queue.put(task_queue_input)
                    self.local_update(self.device, task_queue, result_queue)

                    local_state_dict, local_loss_dict = result_queue.get()
                    for loss_key in local_loss_dict:
                        local_loss_dicts[loss_key].append(local_loss_dict[loss_key])

                    local_models.append(local_state_dict)

                    for param_key in local_state_dict:
                        local_weights[param_key].append(local_state_dict[param_key])
                        local_deltas[param_key].append(local_state_dict[param_key] - global_state_dict[param_key])

            if self.args.multiprocessing:
                for _ in range(len(selected_client_ids)):
                    # Retrieve results from the queue
                    result = result_queue.get()
                    local_state_dict, local_loss_dict = result
                    for loss_key in local_loss_dict:
                        local_loss_dicts[loss_key].append(local_loss_dict[loss_key])

                    local_models.append(local_state_dict)

                    # If you want to save gpu memory, make sure that weights are not allocated to GPU
                    for param_key in local_state_dict:
                        local_weights[param_key].append(local_state_dict[param_key])
                        local_deltas[param_key].append(local_state_dict[param_key] - global_state_dict[param_key])

            logger.info(f"Global epoch {epoch}, Train End. Total Time: {time.time() - start:.2f}s")
            
            if ((epoch + 1) % 100 == 0 or epoch == 0):
                cka_mat = self.evaluate(local_models)
                wandb.log({"CKA": np.mean(cka_mat)}, step=epoch)
                logger.info(cka_mat)

            # Server-side
            updated_global_state_dict = self.server.aggregate(local_weights, local_deltas,
                                                            selected_client_ids, copy.deepcopy(global_state_dict), current_lr, 
                                                            epoch=epoch if self.args.server.get('AnalizeServer') else None)
            
            self.model.load_state_dict(updated_global_state_dict)
            gc.collect()

        if self.args.multiprocessing:
            # Terminate Processes
            terminate_processes(task_queues, processes)

        return

    def evaluate(self, local_models: list) -> Dict:
        local_model_list = [copy.deepcopy(self.model) for _ in range(len(local_models))]
        for i, state_dict in enumerate(local_models):
            local_model_list[i].load_state_dict(state_dict)
        results = self.evaler.eval(local_model_list)
        cka_mat = results["cka_matrix"]

        return cka_mat
