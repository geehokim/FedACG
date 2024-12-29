CUDA_VISIBLE_DEVICES=0
DATASET=cifar100
BATCH_SIZE=50
DECAY=0.995
if [ ${DATASET} = "tinyimagenet" ];then
    BATCH_SIZE=100
    DECAY=0.998
fi 
ALPHA=0.3
TOTAL_EPOCHS=5000
LOCAL_EPOCHS=5
GLOBAL_ROUNDS=$(($TOTAL_EPOCHS / $LOCAL_EPOCHS))

python3 federated_train.py client=ACG server=FedACG exp_name=FedACGWS_L"$LOCAL_EPOCHS" \
dataset=${DATASET} trainer.num_clients=100 split.alpha=${ALPHA} trainer.participation_rate=0.05 \
trainer.local_epochs=${LOCAL_EPOCHS} trainer.global_rounds=${GLOBAL_ROUNDS} \
batch_size=${BATCH_SIZE} wandb=True trainer.local_lr_decay=${DECAY} model=resnet18_WS project="ablations" \

