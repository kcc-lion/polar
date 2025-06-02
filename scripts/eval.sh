#!/bin/bash

export NCCL_NET=Socket
export NCCL_IBEXT_DISABLE=1
export HF_HOME=/local/home/$USER/.cache/huggingface/
export OMP_NUM_THREADS=9
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=6
NUM_PROCESSES=${WORLD_SIZE}

ADAPTER_NAME=$1
TASK_NAME=$2
BASE_MODEL=$3
RUN_DIR=${ADAPTER_NAME}/eval_${TASK_NAME}
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
