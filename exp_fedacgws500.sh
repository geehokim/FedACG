CUDA_VISIBLE_DEVICES=0
DATASET=cifar100
BATCH_SIZE=10
DECAY=0.995
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=20
    DECAY=0.998
fi 
ALPHA=0.3

python3 federated_train.py client=ACG server=FedACG exp_name=FedACGWSV2_"$ALPHA" \
dataset=${DATASET} trainer.num_clients=500 split.alpha=${ALPHA} trainer.participation_rate=0.02 \
batch_size=${BATCH_SIZE} wandb=True trainer.local_lr_decay=${DECAY} model=resnet18_WS project="FedWS_2_500" \
# split.mode=iid

