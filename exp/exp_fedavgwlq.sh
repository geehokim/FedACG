CUDA_VISIBLE_DEVICES=0
DATASET=cifar100
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.3
NBITS=4

python3 federated_train.py client=base server=base visible_devices=\'3\' exp_name=FedAvgWLQ_"$ALPHA"_"B$NBITS" \
dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
quantizer=WLQ quantizer.wt_bit=${NBITS} \
batch_size=${BATCH_SIZE} wandb=True model=resnet18_WS project="FedWS_5_100" \
# split.mode=iid
