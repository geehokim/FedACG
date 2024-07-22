CUDA_VISIBLE_DEVICES=0 \
python federated_train.py client=Decorr server=base exp_name=FedDecorr dataset=cifar100 trainer.num_clients=100 split.alpha=0.3 trainer.participation_rate=0.05 batch_size=50 wandb=True
