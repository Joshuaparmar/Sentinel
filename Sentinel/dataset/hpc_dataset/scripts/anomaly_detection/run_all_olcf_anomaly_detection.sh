#!/bin/bash

# This script runs evaluation for all models on the OLCF dataset
# for the anomaly_detection task.
# It should be run from the project root directory.
# Example: bash scripts/anomaly_detection/run_all_olcf_anomaly_detection.sh

echo "Starting OLCF dataset anomaly detection for all models..."
echo "Current working directory: $(pwd)"
echo "Please ensure you are running this script from the project root directory."
# Common parameters for all model evaluation scripts
export TASK_NAME="anomaly_detection"
# export IS_TRAINING=1
export IS_TRAINING=0
export ROOT_PATH="./data/olcf_task/"
export DATASET="olcf_anomaly"
export FEATURES="M"
export SEQ_LEN=60
export PRED_LEN=0
export ENC_IN=28 # Adjusted for OLCF dataset
export C_OUT=28  # Adjusted for OLCF dataset
export ANOMALY_RATIO=1
export BATCH_SIZE=2048
export TRAIN_EPOCHS=10

# Define xid and anomaly tolerances to iterate over
xids=(63 74 79)
# xids=(79)
anomaly_tolerances=('1min' '5min' '10min' '30min' '60min')
# anomaly_tolerances=('1min')

# Define models to evaluate (matching MSL anomaly detection scripts)
# models=(
#     "FEDformer"
# )
models=(
    "Autoformer"
    "DLinear"
    "Informer"
    "iTransformer"
    "MICN"
    "Pyraformer"
    "Reformer"
    "TimesNet"
    "Transformer"
)

BASE_LOG_DIR="logs/olcf_anomaly_detection_$(date +%Y%m%d_%H%M%S)"
mkdir -p ${BASE_LOG_DIR}

echo "Logging output to ${BASE_LOG_DIR}"

for xid in "${xids[@]}"; do
    for tolerance in "${anomaly_tolerances[@]}"; do
        export ANOMALY_TOLERANCE="${tolerance}" # Export for sub-scripts
        export XID="${xid}"
        echo "===================================================="
        echo "Running with Anomaly Tolerance: $ANOMALY_TOLERANCE"
        echo "Running with XID: $XID"
        echo "===================================================="

        LOG_DIR="${BASE_LOG_DIR}/tolerance_${tolerance}"
        mkdir -p ${LOG_DIR}

        for model_name in "${models[@]}"; do
            script_path="scripts/anomaly_detection/OLCF/${model_name}.sh"
            log_file="${LOG_DIR}/${model_name}_anomaly_tolerance_${tolerance}_xid_${xid}.log"

            echo "----------------------------------------------------"
            echo "Running anomaly detection for model: $model_name with tolerance: $ANOMALY_TOLERANCE and XID: $XID"
            echo "Log file: ${log_file}"
            echo "----------------------------------------------------"
            
            if [ -f "${script_path}" ]; then
                # Pass ANOMALY_TOLERANCE to the script if it accepts it as an argument,
                # or ensure the script reads it from the environment.
                # Assuming the scripts read it from the environment as it's exported.
                bash "${script_path}" > "${log_file}" 2>&1
                if [ $? -ne 0 ]; then
                echo "ERROR: Anomaly detection failed for model $model_name with tolerance $ANOMALY_TOLERANCE and XID: $XID. Check log: ${log_file}" | tee -a "${BASE_LOG_DIR}/master_run.log"
                else
                echo "Successfully completed anomaly detection for model $model_name with tolerance $ANOMALY_TOLERANCE and XID: $XID."
                fi
            else
                echo "WARNING: Script ${script_path} not found!" | tee -a "${BASE_LOG_DIR}/master_run.log"
            fi
        done
    done
done

echo ""
echo "All model anomaly detection scripts for all tolerances have been launched."
echo "Check individual log files in ${BASE_LOG_DIR} for progress and results."

wait
echo "All OLCF dataset anomaly detection tasks complete."