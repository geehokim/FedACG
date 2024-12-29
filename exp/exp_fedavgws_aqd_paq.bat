@echo off

cd ..\

set CUDA_VISIBLE_DEVICES=0

set DATASET=cifar100
set BATCH_SIZE=50
if "%DATASET%" == "tinyimagenet" (
    set BATCH_SIZE=100
)

set ALPHA=0.3
set PARTICIPATION_RATE=0.02
set WT_BIT=1
set QUANTIZERS=PAQ AQD

for %%Q in (%QUANTIZERS%) do (
    echo Running experiment with quantizer: %%Q
    python federated_train.py client=base server=base quantizer=%%Q ^
    exp_name=FedAvgWS_%%Q_%ALPHA%_%WT_BIT%bit dataset=%DATASET% trainer.num_clients=100 ^
    split.alpha=%ALPHA% trainer.participation_rate=%PARTICIPATION_RATE% ^
    batch_size=%BATCH_SIZE% wandb=True model=resnet18_WS quantizer.wt_bit=%WT_BIT% ^
    project="FedACG_WS_%ALPHA%_%PARTICIPATION_RATE%_Quant"
)
