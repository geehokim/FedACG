CUDA_VISIBLE_DEVICES=0
DATASET=cifar100
BATCH_SIZE=50
DECAY=0.995
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
    DECAY=0.998
fi 
ALPHA=0.6

python3 federated_train.py client=ACG server=FedACG exp_name=FedACGWSV2_IID \
dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
batch_size=${BATCH_SIZE} wandb=True trainer.local_lr_decay=${DECAY} model=resnet18_WS project="FedWS_5_100" \
split.mode=iid
