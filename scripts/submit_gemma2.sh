TASKS=( boolq  )
ADAPTER_NAMES=( polar lora )
NUM_EPOCHS=5
LEARNING_RATES=( 4e-3 )
REGULARIZATION_LAMBDAS=( 0.05 )
RANKS=( 8 )
SEED=0
WANDB_PROJECT=0408_compile_single_gpu
WANDB_MODE=online
PARAMETERIZE_S_GRID=( identity )
GRADIENT_TYPES=( landing )
INITIALIZATIONS=( default )
SCRATCH=/local/home/$USER
EXP_DIR=$SCRATCH/${WANDB_PROJECT}
mkdir -p $EXP_DIR
for RANK in ${RANKS[@]}; do
    for TASK in ${TASKS[@]}; do
        for REGULARIZATION_LAMBDA in ${REGULARIZATION_LAMBDAS[@]}; do
            for LEARNING_RATE in ${LEARNING_RATES[@]}; do
                for ADAPTER_NAME in ${ADAPTER_NAMES[@]}; do
                    for i in ${!PARAMETERIZE_S_GRID[@]}; do
                        PARAMETERIZE_S=${PARAMETERIZE_S_GRID[$i]}
                        GRADIENT_TYPE=${GRADIENT_TYPES[$i]}
                        INITIALIZATION=${INITIALIZATIONS[$i]}
                        echo "TASK: ${TASK}, GRAD: ${GRADIENT_TYPE}, PARA: ${PARAMETERIZE_S}, Init ${INITIALIZATION}"
                        bash scripts/finetune_gemma_local.sh ${ADAPTER_NAME} ${LEARNING_RATE} ${REGULARIZATION_LAMBDA} $NUM_EPOCHS $RANK $TASK $SEED $WANDB_PROJECT $WANDB_MODE $PARAMETERIZE_S $GRADIENT_TYPE $INITIALIZATION
                    done
                done
            done
        done
    done
done
