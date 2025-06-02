#!/bin/bash
#SBATCH --job-name=landing_lora
#SBATCH --ntasks=4
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=9
#SBATCH --gpus-per-task=1
#SBATCH --time=10:30:00
#SBATCH --account=a-a11
#SBATCH --environment=landing-lora
#SBATCH --container-workdir=/users/$USER

export HF_HOME=$SCRATCH/huggingface
export WORLD_SIZE=1
export MASTER_ADDR=localhost
MASTER_PORT1=12355
MASTER_PORT2=12356
MASTER_PORT3=12357
MASTER_PORT4=12358


WANDB_MODE=online
ADAPTER_NAME=$1
LEARNING_RATE=$2
REGULARIZATION_LAMBDA=$3
NUM_TRAIN_EPOCHS=$4
LORA_RANK=$5
TASK_NAME=$6
WANDB_MODE=$7
WANDB_PROJECT=$8
PARAMETRIZE_S=$9
GRADIENT_TYPE=landing
USE_DORA=False
INIT_LORA_WEIGHTS=default
MODEL_NAME=microsoft/deberta-v3-base
LORA_ALPHA=8
MAX_SEQ_LENGTH=128
BATCH_SIZE=32
CLS_LEARNING_RATE=$LEARNING_RATE
LOGGING_STEPS=300


SEED1=0
SEED2=1
SEED3=2
SEED4=3
RUN_NAME1=$(date +"%Y%m%dT%H%M%S%3N")
RESULTS_DIR1=$SCRATCH/$WANDB_PROJECT/${RUN_NAME1}_${SLURM_JOB_ID}_${ADAPTER_NAME}_${REGULARIZATION_LAMBDA}_${LEARNING_RATE}
sleep 5
RUN_NAME2=$(date +"%Y%m%dT%H%M%S%3N")
RESULTS_DIR2=$SCRATCH/$WANDB_PROJECT/${RUN_NAME2}_${SLURM_JOB_ID}_${ADAPTER_NAME}_${REGULARIZATION_LAMBDA}_${LEARNING_RATE}
sleep 5
RUN_NAME3=$(date +"%Y%m%dT%H%M%S%3N")
RESULTS_DIR3=$SCRATCH/$WANDB_PROJECT/${RUN_NAME3}_${SLURM_JOB_ID}_${ADAPTER_NAME}_${REGULARIZATION_LAMBDA}_${LEARNING_RATE}
sleep 5
RUN_NAME4=$(date +"%Y%m%dT%H%M%S%3N")
RESULTS_DIR4=$SCRATCH/$WANDB_PROJECT/${RUN_NAME4}_${SLURM_JOB_ID}_${ADAPTER_NAME}_${REGULARIZATION_LAMBDA}_${LEARNING_RATE}

echo $PWD
# Launch 4 independent training jobs, each assigned to a different GPU
srun --exclusive --export=ALL,MASTER_PORT=$MASTER_PORT1 --ntasks=1 --mem=75000 bash -c "source ${HOME}/venv-sola/bin/activate && WANDB_MODE=${WANDB_MODE} WANDB_PROJECT=${WANDB_PROJECT} python $HOME/repos/landing-lora/run_glue.py --adapter_name $ADAPTER_NAME --regularization_lambda $REGULARIZATION_LAMBDA --parametrize_S $PARAMETRIZE_S --gradient_type $GRADIENT_TYPE --use_dora $USE_DORA --run_name $RUN_NAME1 --init_lora_weights $INIT_LORA_WEIGHTS --model_name_or_path $MODEL_NAME --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA --task_name $TASK_NAME --do_train --do_eval --seed $SEED1 --max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE --cls_learning_rate $CLS_LEARNING_RATE --num_train_epochs $NUM_TRAIN_EPOCHS --save_steps 500 --save_strategy no --evaluation_strategy epoch --logging_steps $LOGGING_STEPS --overwrite_output_dir --output_dir $RESULTS_DIR1" &
srun --exclusive --export=ALL,MASTER_PORT=$MASTER_PORT2 --ntasks=1 --mem=75000 bash -c "source ${HOME}/venv-sola/bin/activate && WANDB_MODE=${WANDB_MODE} WANDB_PROJECT=${WANDB_PROJECT} python $HOME/repos/landing-lora/run_glue.py --adapter_name $ADAPTER_NAME --regularization_lambda $REGULARIZATION_LAMBDA --parametrize_S $PARAMETRIZE_S --gradient_type $GRADIENT_TYPE --use_dora $USE_DORA --run_name $RUN_NAME2 --init_lora_weights $INIT_LORA_WEIGHTS --model_name_or_path $MODEL_NAME --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA --task_name $TASK_NAME --do_train --do_eval --seed $SEED2 --max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE --cls_learning_rate $CLS_LEARNING_RATE --num_train_epochs $NUM_TRAIN_EPOCHS --save_steps 500 --save_strategy no --evaluation_strategy epoch --logging_steps $LOGGING_STEPS --overwrite_output_dir --output_dir $RESULTS_DIR2" &
srun --exclusive --export=ALL,MASTER_PORT=$MASTER_PORT3 --ntasks=1 --mem=75000 bash -c "source ${HOME}/venv-sola/bin/activate && WANDB_MODE=${WANDB_MODE} WANDB_PROJECT=${WANDB_PROJECT} python $HOME/repos/landing-lora/run_glue.py --adapter_name $ADAPTER_NAME --regularization_lambda $REGULARIZATION_LAMBDA --parametrize_S $PARAMETRIZE_S --gradient_type $GRADIENT_TYPE --use_dora $USE_DORA --run_name $RUN_NAME3 --init_lora_weights $INIT_LORA_WEIGHTS --model_name_or_path $MODEL_NAME --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA --task_name $TASK_NAME --do_train --do_eval --seed $SEED3 --max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE --cls_learning_rate $CLS_LEARNING_RATE --num_train_epochs $NUM_TRAIN_EPOCHS --save_steps 500 --save_strategy no --evaluation_strategy epoch --logging_steps $LOGGING_STEPS --overwrite_output_dir --output_dir $RESULTS_DIR3" &
srun --exclusive --export=ALL,MASTER_PORT=$MASTER_PORT4 --ntasks=1 --mem=75000 bash -c "source ${HOME}/venv-sola/bin/activate && WANDB_MODE=${WANDB_MODE} WANDB_PROJECT=${WANDB_PROJECT} python $HOME/repos/landing-lora/run_glue.py --adapter_name $ADAPTER_NAME --regularization_lambda $REGULARIZATION_LAMBDA --parametrize_S $PARAMETRIZE_S --gradient_type $GRADIENT_TYPE --use_dora $USE_DORA --run_name $RUN_NAME4 --init_lora_weights $INIT_LORA_WEIGHTS --model_name_or_path $MODEL_NAME --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA --task_name $TASK_NAME --do_train --do_eval --seed $SEED4 --max_seq_length $MAX_SEQ_LENGTH --per_device_train_batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE --cls_learning_rate $CLS_LEARNING_RATE --num_train_epochs $NUM_TRAIN_EPOCHS --save_steps 500 --save_strategy no --evaluation_strategy epoch --logging_steps $LOGGING_STEPS --overwrite_output_dir --output_dir $RESULTS_DIR4" &
wait