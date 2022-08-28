DATA_ROOT=/data/tsq/dialog_sum/CSDS/glm
CHECKPOINT_PATH="/data/tsq/glm"
SAVE_PATH=/data/tsq/dialog_sum/CSDS/glm/finetune_checkpoints
DATESTR=$(date +"%m-%d-%H-%M")
TASK_NAME="cnn_dm_original"
MODEL_TYPE="blocklm-large-chinese"
EXPERIMENT_NAME=${MODEL_TYPE}-csds
DATA_PATH="${DATA_ROOT}"


source $1    # Model
source $2    # Task

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=8
HOST_FILE_PATH="./hostfile"
MP_SIZE=1
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

OPTIONS_NCCL="NCCL_DEBUG=info NCCL_IB_DISABLE=0 NCCL_NET_GDR_LEVEL=2"

EXPERIMENT_NAME=${EXPERIMENT_NAME}_${DATESTR}
mkdir logs
DISTRIBUTED_ARGS=python
run_cmd="${DISTRIBUTED_ARGS} finetune_glm.py \
       --finetune \
       --experiment-name ${EXPERIMENT_NAME} \
       --task ${TASK_NAME} \
       --data-dir ${DATA_PATH} \
       --save ${CHECKPOINT_PATH} \
       --checkpoint-activations \
       --num-workers 1 \
       --no-load-lr-scheduler \
       $MODEL_ARGS \
       $TRAIN_ARGS \
       $COMMON_ARGS \
       $TASK_ARGS \
       --fp16 \
       --model-parallel-size ${MP_SIZE} \
       --overwrite \
       2>&1 | tee logs/log-${EXPERIMENT_NAME}.txt"

echo ${run_cmd}
eval ${run_cmd}
