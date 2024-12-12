CUDA_VISIBLE_DEVICES=0,1
DATASET=cifar100
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.3

python3 federated_train.py multiprocessing=True main_gpu=0 client=MLB server=base exp_name=FedMLB_"$ALPHA" \
dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.02 \
batch_size=${BATCH_SIZE} wandb=True model=resnet18_MLB project="FedWS_2_100" \
# split.mode=iidd