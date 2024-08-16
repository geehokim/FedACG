CUDA_VISIBLE_DEVICES=0 \
python federated_train.py client=base server=base exp_name=FedAvgBN dataset=cifar100 trainer.num_clients=100 split.alpha=0.3 trainer.participation_rate=0.05 wandb=True trainer.local_lr_decay=0.995 model=resnet18_BN
