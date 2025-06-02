TASKS=( metamathqa )
ADAPTER_NAMES=( polar lora )
NUM_EPOCHS=2
LEARNING_RATES=( 4e-3 8e-4 )
REGULARIZATION_LAMBDAS=( 5e-3 )
RANKS=( 16 )
SEED=0
WANDB_PROJECT=0424_gemma3_meta
BASE_MODEL=google/gemma-3-27b-pt
WANDB_MODE=online
PARAMETERIZE_S_GRID=( identity )
EXP_DIR=$SCRATCH/${WANDB_PROJECT}
mkdir -p $EXP_DIR
for RANK in ${RANKS[@]}; do
    for TASK in ${TASKS[@]}; do
        for REGULARIZATION_LAMBDA in ${REGULARIZATION_LAMBDAS[@]}; do
            for LEARNING_RATE in ${LEARNING_RATES[@]}; do
                for ADAPTER_NAME in ${ADAPTER_NAMES[@]}; do
                    for PARAMETERIZE_S in ${PARAMETERIZE_S_GRID[@]}; do
                        ID=$(sbatch --output=${EXP_DIR}/slurm-%j.out --parsable scripts/finetune_gemma3_meta.sh ${ADAPTER_NAME} ${LEARNING_RATE} ${REGULARIZATION_LAMBDA} $NUM_EPOCHS $RANK $TASK $SEED $WANDB_PROJECT $WANDB_MODE $PARAMETERIZE_S $BASE_MODEL)
                        sbatch --output=${EXP_DIR}/slurm-%j.out --dependency=afterok:${ID} --kill-on-invalid-dep=yes scripts/eval_math.sh $EXP_DIR/${ID}_${ADAPTER_NAME}_${TASK}_${LEARNING_RATE}_${REGULARIZATION_LAMBDA}_${RANK} hendrycks_math $BASE_MODEL
                        sbatch --output=${EXP_DIR}/slurm-%j.out --dependency=afterok:${ID} --kill-on-invalid-dep=yes scripts/eval_math.sh $EXP_DIR/${ID}_${ADAPTER_NAME}_${TASK}_${LEARNING_RATE}_${REGULARIZATION_LAMBDA}_${RANK} gsm8k $BASE_MODEL
                    done
                done
            done
        done
    done
done