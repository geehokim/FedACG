CUDA_VISIBLE_DEVICES=0
DATASET=cifar100
BATCH_SIZE=50
DECAY=0.998
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
    DECAY=0.995
fi 
ALPHA=0.1

python3 federated_train.py client=ACG server=FedACG exp_name=FedACGWS_"$ALPHA" \
dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
batch_size=${BATCH_SIZE} wandb=True trainer.local_lr_decay=${DECAY} model=resnet18_WS \
# split.mode=iid
