#!/bin/bash

export HF_HOME=/local/home/$USER/.cache/huggingface/
export OMP_NUM_THREADS=9
export WORLD_SIZE=1
export CUDA_VISIBLE_DEVICES=1
export SCRATCH=/local/home/$USER

export NCCL_NET=Socket
export NCCL_IBEXT_DISABLE=1

NUM_PROCESSES=${WORLD_SIZE}

ADAPTER_NAME=$1 # polar, lora, dora
LEARNING_RATE=$2 # 4e-3, 4e-4
REGULARIZATION_LAMBDA=$3 # 5e-3
NUM_EPOCHS=$4 # 5
RANK=$5 # 4, 8, 16, 32
DATASET_DIR=$6 # boolq piqa social_iqa hellaswag winogrande arc_easy arc_challenge openbookqa
SEED=0
ACTIVATE_PROFILING=False
WANDB_PROJECT=polar_experiments
WANDB_MODE=disabled
PARAMETERIZE_S=identity
GRADIENT_TYPE=landing
INIT_STRATEGY=default
DTYPE=fp16
export WANDB_MODE=disabled
BASE_MODEL=meta-llama/Llama-2-7b-hf
LR_SCHEDULER_TYPE=cosine
CUTOFF_LEN=512
LORA_ALPHA=32
CURRENT_DATE=$(date +"%Y%m%dT%H%M%S%3N")
echo $CURRENT_DATE
echo $ADAPTER_NAME
RUN_NAME=${CURRENT_DATE}_${ADAPTER_NAME}_${DATASET_DIR}_${LEARNING_RATE}_${REGULARIZATION_LAMBDA}_${RANK}
RUN_DIR=$SCRATCH/${WANDB_PROJECT}/${RUN_NAME}
mkdir -p $RUN_DIR
WANDB_RUN_NAME=${RUN_NAME}

python -m torch.distributed.run --nproc_per_node=${NUM_PROCESSES} --master-port=29510 $HOME/repos/landing-lora/finetune_commonsense.py \
	--base_model ${BASE_MODEL} \
	--data_path ${DATASET_DIR} \
	--output_dir $RUN_DIR \
    --adapter_name ${ADAPTER_NAME} \
	--activate_profiling ${ACTIVATE_PROFILING} \
	--batch_size 128 \
	--dtype ${DTYPE} \
	--micro_batch_size 6 \
	--num_epochs ${NUM_EPOCHS} \
	--learning_rate $LEARNING_RATE \
    --regularization_lambda $REGULARIZATION_LAMBDA \
	--lora_r ${RANK} \
	--lora_alpha ${LORA_ALPHA} \
	--cutoff_len ${CUTOFF_LEN} \
	--lr_scheduler_type ${LR_SCHEDULER_TYPE} \
	--val_set_size 0 \
	--init_lora_weights ${INIT_STRATEGY} \
	--lora_target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
	--adapter_name $ADAPTER_NAME \
	--parameterize_S ${PARAMETERIZE_S} \
	--gradient_type ${GRADIENT_TYPE} \
    --wandb_project ${WANDB_PROJECT} \
    --seed ${SEED} \
    --wandb_run_name ${WANDB_RUN_NAME} 2>&1 | tee "${RUN_DIR}/train.log"
