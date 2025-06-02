#!/bin/bash
#SBATCH --job-name=commonsense_landing
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH --account=a-a11
#SBATCH --environment=landing-lora-2503
#SBATCH --container-workdir=/users/$USER

# Below needed because of intermittent error: 
# ncclInvalidUsage: This usually reflects invalid usage of NCCL library.
# Error: network AWS Libfabric not found.
export NCCL_NET=Socket
export NCCL_IBEXT_DISABLE=1
#export HF_HOME=/local/home/$USER/.cache/huggingface/
export HF_HOME=$SCRATCH/huggingface
export OMP_NUM_THREADS=9
export WORLD_SIZE=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_PROCESSES=${WORLD_SIZE}
WORKING_DIR=$SCRATCH
#WORKING_DIR=/home/$USER/repos/landing-lora
cd ${WORKING_DIR}
source ${HOME}/venv-polar/bin/activate



ADAPTER_NAME=$1
TASK_NAME=$2
BASE_MODEL=$3
RUN_DIR=${ADAPTER_NAME}/eval_${TASK_NAME}
mkdir -p $RUN_DIR
SEED=0
#,load_in_4bit=True,bnb_4bit_compute_dtype=bfloat16,bnb_4bit_quant_type=nf4,bnb_4bit_use_double_quant=True
accelerate launch --multi_gpu --gpu_ids ${CUDA_VISIBLE_DEVICES} --num_processes ${NUM_PROCESSES} -m lm_eval \
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
