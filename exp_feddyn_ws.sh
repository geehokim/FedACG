CUDA_VISIBLE_DEVICES=5,7
DATASET=tinyimagenet
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi
LR_DECAY=0.998

python3 federated_train.py server=FedDyn client=Dyn exp_name=FedDynWS dataset=${DATASET} trainer.num_clients=100 split.alpha=0.3 trainer.participation_rate=0.05 batch_size=${BATCH_SIZE} wandb=True trainer.local_lr_decay=${LR_DECAY} model=resnet18_WS
