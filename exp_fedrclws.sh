CUDA_VISIBLE_DEVICES=0
DATASET=tinyimagenet
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.1

python3 federated_train.py client=fedrcl server=base exp_name=FedRCLWS_"$ALPHA" \
dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
batch_size=${BATCH_SIZE} wandb=True model=resnet18_WS \
# split.mode=iid
