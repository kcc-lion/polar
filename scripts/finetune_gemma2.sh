#!/bin/bash
#SBATCH --job-name=commonsense_landing
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=288
#SBATCH --time=23:30:00
#SBATCH --account=a-a11
#SBATCH --environment=landing-lora
#SBATCH --container-workdir=/users/$USER

#export HF_HOME=$SCRATCH/huggingface
export HF_HOME=/local/home/$USER/.cache/huggingface/
export WANDB_MODE=online
# Don't forget: Change n_proc_per_node below
export CUDA_VISIBLE_DEVICES=2
export OMP_NUM_THREADS=8
#nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
#nodes_array=($nodes)
#head_node=${nodes_array[0]}
SCRATCH=/local/home/$USER
#source ${HOME}/venv-sola/bin/activate

ADAPTER_NAME=$1 # LoRA, PoLAR
LEARNING_RATE=$2
REGULARIZATION_LAMBDA=$3
NUM_EPOCHS=$4
RANK=$5
DATASET_DIR=$6
SEED=$7
WANDB_PROJECT=$8
WANDB_MODE=$9
export WANDB_MODE=${WANDB_MODE}
PARAMETERIZE_S=${10}
GRADIENT_TYPE=${11}
INITIALIZATION=${12}
BASE_MODEL=google/gemma-2-2b
CUTOFF_LEN=512
CURRENT_DATE=$(date +"%Y%m%dT%H%M%S%3N")
RUN_NAME=${CURRENT_DATE}_${ADAPTER_NAME}_${DATASET_DIR}_${LEARNING_RATE}_${REGULARIZATION_LAMBDA}_${RANK}
RUN_DIR=$SCRATCH/${WANDB_PROJECT}/${RUN_NAME}
mkdir -p $RUN_DIR
WANDB_RUN_NAME=${RUN_NAME}
echo "Starting job at $CURRENT_DATE"
echo "Run Name: ${RUN_NAME}"
echo "Output Dir: ${RUN_DIR}"

python -m torch.distributed.run --nproc_per_node=1 $HOME/repos/landing-lora/finetune_commonsense.py \
        --base_model ${BASE_MODEL} \
        --data_path ${DATASET_DIR} \
        --output_dir $RUN_DIR \
        --adapter_name ${ADAPTER_NAME} \
        --batch_size 128 \
        --micro_batch_size 16 \
        --num_epochs ${NUM_EPOCHS} \
        --learning_rate $LEARNING_RATE \
        --regularization_lambda $REGULARIZATION_LAMBDA \
        --init_lora_weights ${INITIALIZATION} \
        --lora_r ${RANK} \
        --lora_alpha 32 \
        --cutoff_len $CUTOFF_LEN \
        --val_set_size 0 \
        --lora_target_modules '["q_proj", "k_proj", "v_proj", "up_proj", "down_proj"]' \
        --adapter_name $ADAPTER_NAME \
        --parameterize_S ${PARAMETERIZE_S} \
        --gradient_type ${GRADIENT_TYPE} \
        --wandb_project ${WANDB_PROJECT} \
        --seed ${SEED} \
        --do_eval True \
        --wandb_run_name ${WANDB_RUN_NAME}

