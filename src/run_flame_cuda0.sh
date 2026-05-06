export CUDA_VISIBLE_DEVICES=0
SEEDS=(2 42 453)
EXPERTS=(5)

RESULTS_DIR='/cis/home/xhan56/code/clinical-highmmt/src/results'
TASK='pheno'
GATING='laplace'
BALANCE_LOSS_COEF='1.0'
ALPHA='const_0.0'
MOD_DROP_RATE='0.0'

for NUM_EXPERTS in "${EXPERTS[@]}"; do
    RESULT_DIR="${RESULTS_DIR}/fusemoe/singletask/${GATING}/${TASK}/mod_drop_rate_${MOD_DROP_RATE}/experts_${NUM_EXPERTS}"
    python aggregate_results.py --result_dir "${RESULT_DIR}"
done