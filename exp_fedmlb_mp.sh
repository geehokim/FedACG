CUDA_VISIBLE_DEVICES=0,1
DATASET=tinyimagenet
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.05

python3 federated_train.py multiprocessing=True main_gpu=0 client=MLB server=FedAvgM exp_name=FedMLB_"$ALPHA" \
dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
batch_size=${BATCH_SIZE} wandb=True model=resnet18_MLB \
# split.mode=iidd