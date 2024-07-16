CUDA_VISIBLE_DEVICES=0 \
python federated_train.py server=FedACG client=ACG exp_name=FedACG dataset=tinyimagenet trainer.num_clients=500 split.alpha=0.3 trainer.participation_rate=0.02 batch_size=20 wandb=True trainer.local_lr_decay=0.995
