#!/bin/bash
export HF_HOME=/workspace
export OMP_NUM_THREADS=9
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=0
NUM_PROCESSES=${WORLD_SIZE}

source /workspace/polar/bin/activate

ADAPTER_NAME=$1
TASK_NAME=$2
BASE_MODEL=$3
RUN_DIR=${ADAPTER_NAME}/eval_docker_${TASK_NAME}
mkdir -p $RUN_DIR
SEED=0
accelerate launch --gpu_ids ${CUDA_VISIBLE_DEVICES} --num_processes ${NUM_PROCESSES} -m lm_eval \
    --model hf \
    --model_args "pretrained=${BASE_MODEL},peft=${ADAPTER_NAME}" \
    --tasks $TASK_NAME \
    --output_path ${RUN_DIR} \
    --batch_size 2 \
    --seed ${SEED} \
    --num_fewshot 0 \
    --write_out \
    --log_samples \
    --trust_remote_code 2>&1 | tee "${RUN_DIR}/eval_${TASK_NAME}.log"
