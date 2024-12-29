CUDA_VISIBLE_DEVICES=0
DATASET=cifar100
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.3
NBITS=1

python3 federated_train.py client=base server=base visible_devices=\'0\' exp_name=FedAvgWSQP_"$ALPHA"_"B$NBITS" \
dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
quantizer=WSQ quantizer.wt_bit=${NBITS} quantizer.wt_clip_prob=0.05 \
batch_size=${BATCH_SIZE} wandb=True model=resnet18_WS project="FedWS_5_100" \
# split.mode=iid
