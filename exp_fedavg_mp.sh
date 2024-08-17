CUDA_VISIBLE_DEVICES=0,1
DATASET=tinyimagenet
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
fi 

python federated_train.py multiprocessing=True main_gpu=0 client=base server=base exp_name=FedAvg dataset=${DATASET} trainer.num_clients=100 split.alpha=0.3 trainer.participation_rate=0.05 batch_size=${BATCH_SIZE} wandb=True