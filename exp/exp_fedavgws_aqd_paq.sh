CUDA_VISIBLE_DEVICES=0

DATASET=cifar100
BATCH_SIZE=50
if [ ${DATASET} = "tinyimagenet" ]; then
    BATCH_SIZE=100
fi

ALPHA=0.3
PARTICIPATION_RATE=0.02
QUANTIZERS=("PAQ" "AQD")
WT_BIT=1

for QUANTIZER in "${QUANTIZERS[@]}"
do
    echo "Running experiment with quantizer: $QUANTIZER"
    python federated_train.py client=base server=base quantizer=${QUANTIZER} \
    exp_name=FedAvgWS_${QUANTIZER}_${ALPHA}_${WT_BIT}bit dataset=${DATASET} trainer.num_clients=100 \
    split.alpha=${ALPHA} trainer.participation_rate=${PARTICIPATION_RATE} \
    batch_size=${BATCH_SIZE} wandb=True model=resnet18_WS model.wt_bit=${WT_BIT} \
    project="FedACG_WS_${ALPHA}_${PARTICIPATION_RATE}_Quant"
done