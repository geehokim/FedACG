CUDA_VISIBLE_DEVICES=0,1 \
python federated_train.py multiprocessing=True main_gpu=0 server=FedACG client=ACG exp_name=FedWSV2_proxACG dataset=tinyimagenet trainer.num_clients=100 split.alpha=0.3 trainer.participation_rate=0.05 batch_size=100 wandb=True model=resnet18_WS trainer.local_lr_decay=0.995
