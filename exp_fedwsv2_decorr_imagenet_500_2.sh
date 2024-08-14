CUDA_VISIBLE_DEVICES=0 \
python federated_train.py client=Decorr server=base exp_name=FedWSV2_Decorr_500_2 dataset=tinyimagenet trainer.num_clients=500 split.alpha=0.3 trainer.participation_rate=0.02 batch_size=20 wandb=True model=resnet18_WS 
