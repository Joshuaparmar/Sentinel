#!/bin/bash

# This script runs evaluation for all models on the OLCF dataset
# for the long_term_forecast task.
# It should be run from the project root directory.
# Example: bash scripts/long_term_forecast/run_all_olcf_eval.sh

echo "Starting OLCF dataset evaluation for all models..."
echo "Current working directory: $(pwd)"
echo "Please ensure you are running this script from the project root directory."

# Common parameters for all model evaluation scripts
export TASK_NAME="long_term_forecast"
# export IS_TRAINING=1
export IS_TRAINING=0
export ROOT_PATH="./data/olcf_task/"
export DATA_PATH="train.csv"
export DATASET="olcf_cluster"
export FEATURES="S"
export TARGET='power'
export SEQ_LEN=60
export LABEL_LEN=30
# export E_LAYERS=2
# export D_LAYERS=1
# export FACTOR=3
export ENC_IN=1
export DEC_IN=1
export C_OUT=1
export DES="Exp_Eval_OLCF_Cluster"
export ITR=1
export PRED_LENS_STR="1 6 60 360" # Space-separated string for PRED_LENS array
# export PRED_LENS_STR="1" # Space-separated string for PRED_LENS array
# export PRED_LENS_STR="360" # Space-separated string for PRED_LENS array
# export PRED_LENS_STR="1 6" # Space-separated string for PRED_LENS array
# export PRED_LENS_STR="6 60 360" # Space-separated string for PRED_LENS array

# Define models to evaluate
models=(
    "Autoformer"
    "Crossformer"
    "DLinear"
    "ETSformer"
    "FEDformer"
    "FiLM"
    "Informer"
    "Mamba"
    "Nonstationary_Transformer"
    "PAttn"
    "PatchTST"
    "Pyraformer"
    "Reformer"
    "TimesNet"
    "TimeXer"
    "Transformer"
    "TSMixer"
)

LOG_DIR="logs/olcf_cluster_eval_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${LOG_DIR}

echo "Logging output to ${LOG_DIR}"

for model_name in "${models[@]}"; do
    script_path="scripts/long_term_forecast/olcf_cluster_script/${model_name}.sh"
    log_file="${LOG_DIR}/${model_name}_eval.log"

    echo "----------------------------------------------------"
    echo "Running evaluation for model: $model_name"
    echo "Log file: ${log_file}"
    echo "----------------------------------------------------"
    
    if [ -f "${script_path}" ]; then
        # Make sure script is executable (optional, can be done once manually)
        # chmod +x "${script_path}"
        # bash "${script_path}" > "${log_file}" 2>&1 &
        # If you want to run sequentially, remove '&' and add error checking:
        bash "${script_path}" > "${log_file}" 2>&1
        if [ $? -ne 0 ]; then
           echo "ERROR: Evaluation failed for model $model_name. Check log: ${log_file}"
        else
           echo "Successfully completed evaluation for model $model_name."
        fi
    else
        echo "WARNING: Script ${script_path} not found!" | tee -a "${LOG_DIR}/master_run.log"
    fi
done

echo ""
echo "All model evaluation scripts have been launched in the background (if using '&')."
echo "Check individual log files in ${LOG_DIR} for progress and results."

# Wait for all background jobs to finish if running in parallel
wait
echo "All OLCF dataset evaluations complete."
