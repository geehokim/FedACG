CUDA_VISIBLE_DEVICES=0
DATASET=cifar100
BATCH_SIZE=10
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=20
fi 
LR_DECAY=0.998

python3 federated_train.py server=base client=base exp_name=FedAVG dataset=${DATASET} project=test \
trainer.num_clients=500 split.alpha=0.3 trainer.participation_rate=0.02 \
batch_size=${BATCH_SIZE} wandb=True trainer.local_lr_decay=${LR_DECAY} model=resnet18 \
trainer.local_epochs=5 trainer.global_rounds=1000