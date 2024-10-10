CUDA_VISIBLE_DEVICES=0
DATASET=cifar10
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 
ALPHA=0.3
RHO=0.1

python3 federated_train.py client=base server=base exp_name=FedAvg_IID_"$RHO" \
dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
batch_size=${BATCH_SIZE} wandb=True model=resnet18_WS model.rho=${RHO} project="ablations" \
split.mode=iid
