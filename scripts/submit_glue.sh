TASKS=( cola mnli mrpc qnli qqp rte sst2 stsb )
ADAPTER_NAMES=( polar lora )
NUM_EPOCHS_PER_TASK=( 25 7 30 5 7 50 25 25 )
LEARNING_RATES=( 4e-4 8e-4 1e-3 )
REGULARIZATION_LAMBDAS=( 1.0 )
RANK=8
WANDB_PROJECT=0311_glue
WANDB_MODE=offline
PARAMETERIZE_S=identity
EXP_DIR=$SCRATCH/${WANDB_PROJECT}
mkdir -p $EXP_DIR
for i in ${!TASKS[@]}; do
    TASK=${TASKS[$i]}
    NUM_EPOCHS=${NUM_EPOCHS_PER_TASK[$i]}
    echo "TASK: ${TASK}, NUM_EPOCHS: ${NUM_EPOCHS}"
    for ADAPTER_NAME in ${ADAPTER_NAMES[@]}; do
        for LEARNING_RATE in ${LEARNING_RATES[@]}; do
            for REGULARIZATION_LAMBDA in ${REGULARIZATION_LAMBDAS[@]}; do
                sbatch --output=${EXP_DIR}/slurm-%j.out scripts/run_glue_cluster.sh ${ADAPTER_NAME} ${LEARNING_RATE} ${REGULARIZATION_LAMBDA} $NUM_EPOCHS $RANK $TASK $WANDB_MODE $WANDB_PROJECT $PARAMETERIZE_S
            done
        done
    done
done