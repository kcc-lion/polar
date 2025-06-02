#!/bin/bash
#SBATCH --job-name=commonsense_landing
#SBATCH --nodes=3
#SBATCH --ntasks=3
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --account=a-a11
#SBATCH --environment=landing-lora-2503
#SBATCH --container-workdir=/users/$USER

export HF_HOME=$SCRATCH/huggingface
export WANDB_MODE=online
export OMP_NUM_THREADS=9

# Below needed because of intermittent error: 
# ncclInvalidUsage: This usually reflects invalid usage of NCCL library.
# Error: network AWS Libfabric not found.
export NCCL_NET=Socket
export NCCL_IBEXT_DISABLE=1

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}

source ${HOME}/venv-polar/bin/activate

ADAPTER_NAME=$1 # LoRA, polar
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
BASE_MODEL=${11}
GRADIENT_TYPE=landing
CUTOFF_LEN=512
CURRENT_DATE=$(date +"%Y%m%dT%H%M%S%3N")
#RUN_NAME=${CURRENT_DATE}_${ADAPTER_NAME}_${DATASET_DIR}_${LEARNING_RATE}_${REGULARIZATION_LAMBDA}_${RANK}
RUN_NAME=${SLURM_JOB_ID}_${ADAPTER_NAME}_${DATASET_DIR}_${LEARNING_RATE}_${REGULARIZATION_LAMBDA}_${RANK}
RUN_DIR=$SCRATCH/${WANDB_PROJECT}/${RUN_NAME}
mkdir -p $RUN_DIR
WANDB_RUN_NAME=${RUN_NAME}
echo "Starting job at $CURRENT_DATE"
echo "Run Name: ${RUN_NAME}"
echo "Output Dir: ${RUN_DIR}"
echo "SLURM_NNODES: ${SLURM_NNODES}"
echo "SLURM_JOBID ${SLURM_JOBID}"

srun --export=ALL --container-workdir=$PWD bash -c "cd ${HOME}/repos/landing-lora && source ${HOME}/venv-polar/bin/activate && python -m torch.distributed.run --nproc_per_node=4 --rdzv-id=${SLURM_JOBID} --rdzv_backend=c10d --rdzv-endpoint="$head_node:44500" --nnodes=${SLURM_NNODES} $HOME/repos/landing-lora/finetune_commonsense.py \
        --base_model ${BASE_MODEL} \
        --data_path ${DATASET_DIR} \
        --output_dir $RUN_DIR \
        --adapter_name ${ADAPTER_NAME} \
        --batch_size 128 \
        --micro_batch_size 2 \
        --num_epochs ${NUM_EPOCHS} \
        --learning_rate $LEARNING_RATE \
        --regularization_lambda $REGULARIZATION_LAMBDA \
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
        --do_eval False \
        --wandb_run_name ${WANDB_RUN_NAME}"