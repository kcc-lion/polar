TASKS=( boolq piqa social_iqa hellaswag winogrande arc_easy arc_challenge openbookqa )
ADAPTER_NAMES=( polar )
NUM_EPOCHS=5
LEARNING_RATES=( 4e-3 )
REGULARIZATION_LAMBDAS=( 5e-3 )
RANKS=( 32 )
SEED=0
DTYPE=fp16
WANDB_PROJECT=0507_sr_dynamics
WANDB_MODE=online
PARAMETERIZE_S_GRID=( identity )
GRADIENT_TYPES=( landing )
#default, symmetric_gaussian
INIT_STRATEGY=default
EXP_DIR=$SCRATCH/${WANDB_PROJECT}
mkdir -p $EXP_DIR
for RANK in ${RANKS[@]}; do
    for TASK in ${TASKS[@]}; do
        for REGULARIZATION_LAMBDA in ${REGULARIZATION_LAMBDAS[@]}; do
            for LEARNING_RATE in ${LEARNING_RATES[@]}; do
                for ADAPTER_NAME in ${ADAPTER_NAMES[@]}; do
                    for PARAMETERIZE_S in ${PARAMETERIZE_S_GRID[@]}; do
                        for GRADIENT_TYPE in ${GRADIENT_TYPES[@]}; do
                            sbatch --output=${EXP_DIR}/slurm-%j.out --parsable scripts/finetune_commonsense.sh ${ADAPTER_NAME} ${LEARNING_RATE} ${REGULARIZATION_LAMBDA} $NUM_EPOCHS $RANK $TASK $SEED $WANDB_PROJECT $WANDB_MODE $PARAMETERIZE_S $GRADIENT_TYPE $INIT_STRATEGY $DTYPE
                            #ID=$(sbatch --output=${EXP_DIR}/slurm-%j.out --parsable scripts/finetune_commonsense.sh ${ADAPTER_NAME} ${LEARNING_RATE} ${REGULARIZATION_LAMBDA} $NUM_EPOCHS $RANK $TASK $SEED $WANDB_PROJECT $WANDB_MODE $PARAMETERIZE_S $GRADIENT_TYPE $INIT_STRATEGY)
                            #sbatch --output=${EXP_DIR}/slurm-%j.out --dependency=afterok:${ID} --kill-on-invalid-dep=yes scripts/eval_math.sh $EXP_DIR/${ID}_${ADAPTER_NAME}_${TASK}_${LEARNING_RATE}_${REGULARIZATION_LAMBDA}_${RANK} minerva_math meta-llama/Llama-2-7b-hf
                        done
                    done
                done
            done
        done
    done
done