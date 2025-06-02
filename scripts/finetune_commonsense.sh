#!/bin/bash
#SBATCH --job-name=commonsense_landing
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=03:30:00
#SBATCH --account=a-a11
#SBATCH --environment=landing-lora
#SBATCH --container-workdir=/users/$USER/repos/landing-lora
export HF_HOME=$SCRATCH/huggingface
#export HF_HOME=/local/home/$USER/.cache/huggingface/
export OMP_NUM_THREADS=9
export WORLD_SIZE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

export NCCL_NET=Socket
export NCCL_IBEXT_DISABLE=1

NUM_PROCESSES=${WORLD_SIZE}
source ${HOME}/venv-sola/bin/activate

ADAPTER_NAME=$1 # LoRA, polar
LEARNING_RATE=$2
REGULARIZATION_LAMBDA=$3
NUM_EPOCHS=$4
RANK=$5
DATASET_DIR=$6 
SEED=$7
WANDB_PROJECT=$8
WANDB_MODE=$9
PARAMETERIZE_S=${10}
GRADIENT_TYPE=${11}
INIT_STRATEGY=${12}
DTYPE=${13}
export WANDB_MODE=${WANDB_MODE}
BASE_MODEL=meta-llama/Llama-2-7b-hf
LR_SCHEDULER_TYPE=cosine
CUTOFF_LEN=512
LORA_ALPHA=32
CURRENT_DATE=$(date +"%Y%m%dT%H%M%S%3N")
echo $CURRENT_DATE
echo $ADAPTER_NAME
RUN_NAME=${SLURM_JOB_ID}_${ADAPTER_NAME}_${DATASET_DIR}_${LEARNING_RATE}_${REGULARIZATION_LAMBDA}_${RANK}
RUN_DIR=$SCRATCH/${WANDB_PROJECT}/${RUN_NAME}
mkdir -p $RUN_DIR
WANDB_RUN_NAME=${RUN_NAME}

python -m torch.distributed.run --nproc_per_node=${NUM_PROCESSES} $HOME/repos/landing-lora/finetune_commonsense.py \
	--base_model ${BASE_MODEL} \
	--data_path ${DATASET_DIR} \
	--output_dir $RUN_DIR \
    --adapter_name ${ADAPTER_NAME} \
	--batch_size 128 \
	--dtype ${DTYPE} \
	--micro_batch_size 8 \
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
