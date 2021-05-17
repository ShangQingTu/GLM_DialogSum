source config_tasks/model_blocklm_1.25.sh
source $1
CHECKPOINT_PATH="/dataset/c07bd62b/finetune_checkpoints"
DATESTR=$(date +"%m-%d-%H-%M")

export NCCL_DEBUG=info
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
export PATH="/opt/conda/bin:$PATH"

NUM_WORKERS=1
NUM_GPUS_PER_WORKER=4
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
HOST_FILE_PATH="/root/code/config/hostfile"

mkdir logs
deepspeed --master_port ${MASTER_PORT} --num_nodes ${NUM_WORKERS} --num_gpus ${NUM_GPUS_PER_WORKER} --hostfile ${HOST_FILE_PATH} finetune_gpt2.py \
       --finetune \
       --experiment-name ${EXPERIMENT_NAME} \
       --task ${TASK_NAME} \
       --data-dir ${DATA_PATH} \
       --save ${CHECKPOINT_PATH} \
       --checkpoint-activations \
       $MODEL_ARGS \
       $TRAIN_ARGS \
       $COMMON_ARGS \
       $TASK_ARGS \
       2>&1 | tee logs/log-${DATESTR}.txt