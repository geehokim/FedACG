CUDA_VISIBLE_DEVICES=0
DATASET=tinyimagenet
BATCH_SIZE=100
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=20
fi 
LR_DECAY=0.998

python3 federated_train.py server=base client=base exp_name=FedAVG dataset=${DATASET} project=FedWS500 \
trainer.num_clients=500 split.alpha=0.3 trainer.participation_rate=0.02 \
batch_size=${BATCH_SIZE} wandb=True trainer.local_lr_decay=${LR_DECAY}
