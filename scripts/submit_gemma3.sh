TASKS=( boolq )
#TASKS=( boolq piqa social_iqa hellaswag winogrande arc_easy arc_challenge openbookqa )
ADAPTER_NAMES=( polar lora )
NUM_EPOCHS=5
LEARNING_RATES=( 4e-3 4e-4 )
REGULARIZATION_LAMBDAS=( 5e-3 )
RANKS=( 32 )
SEED=0
WANDB_PROJECT=0423_gemma3_commonsense
WANDB_MODE=online
EXP_DIR=$SCRATCH/${WANDB_PROJECT}
mkdir -p $EXP_DIR
for RANK in ${RANKS[@]}; do
    for TASK in ${TASKS[@]}; do
        for REGULARIZATION_LAMBDA in ${REGULARIZATION_LAMBDAS[@]}; do
            for LEARNING_RATE in ${LEARNING_RATES[@]}; do
                for ADAPTER_NAME in ${ADAPTER_NAMES[@]}; do
                    sbatch --output=${EXP_DIR}/slurm-%j.out scripts/finetune_gemma3.sh ${ADAPTER_NAME} ${LEARNING_RATE} ${REGULARIZATION_LAMBDA} $NUM_EPOCHS $RANK $TASK $SEED $WANDB_PROJECT $WANDB_MODE
                done
            done
        done
    done
done
