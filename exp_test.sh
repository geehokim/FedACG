CUDA_VISIBLE_DEVICES=0
DATASET=cifar100
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
LR_DECAY=0.995

<<<<<<< HEAD
python3 federated_train.py server=base client=base exp_name=anble dataset=${DATASET} project=test \
=======
python3 federated_train.py multiprocessing=True main_gpu=0 server=FedACG client=base exp_name=FedAVGGNbase_rho1 dataset=${DATASET} project=test \
>>>>>>> 877819a8a355e05dcbccbddcf13fb7b723929925
trainer.num_clients=100 split.alpha=0.3 trainer.participation_rate=0.05 \
batch_size=${BATCH_SIZE} wandb=False trainer.local_lr_decay=${LR_DECAY} model=resnet18_WS \
trainer.local_epochs=5 trainer.global_rounds=1000
# multiprocessing=True main_gpu=0