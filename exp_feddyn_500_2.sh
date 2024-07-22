CUDA_VISIBLE_DEVICES=0 \
python federated_train.py client=Dyn server=FedDyn exp_name=FedDyn_500_2 dataset=cifar100 trainer.num_clients=500 split.alpha=0.3 trainer.participation_rate=0.02 batch_size=10 wandb=True
