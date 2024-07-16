CUDA_VISIBLE_DEVICES=0,1 \
python federated_train.py multiprocessing=True main_gpu=0 client=Dyn server=FedDyn exp_name=FedWSV2_DynDyn dataset=cifar100 trainer.num_clients=100 split.alpha=0.3 trainer.participation_rate=0.05 wandb=True model=resnet18_WS
