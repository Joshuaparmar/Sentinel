#!/bin/bash

# This script evaluates Reformer on OLCF dataset for long_term_forecast.
# It is intended to be run from the project root directory.

# export CUDA_VISIBLE_DEVICES=0

# Common parameters are now expected to be in the environment (exported by the calling script)
# TASK_NAME="long_term_forecast"
# IS_TRAINING=1
# ROOT_PATH="./data/olcf_task/"
# DATA_PATH="train.csv"
# DATASET="olcf"
# FEATURES="M"
# SEQ_LEN=60
# LABEL_LEN=30
E_LAYERS=2
D_LAYERS=1
FACTOR=3
# ENC_IN=28
# DEC_IN=28
# C_OUT=28
# DES="Exp_Eval_OLCF"
# ITR=1

MODEL_NAME="Reformer" # This remains specific to this model's script
PRED_LENS=(${PRED_LENS_STR}) # Reconstruct PRED_LENS array from the exported string

echo "===================================================="
echo "Evaluating Model: ${MODEL_NAME} on Dataset: ${DATASET}"
echo "Sequence Length: ${SEQ_LEN}, Label Length: ${LABEL_LEN}"
echo "Prediction Lengths (minutes): 10, 20, 30, 60"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"
echo "===================================================="

for PRED_LEN in "${PRED_LENS[@]}"; do
    MODEL_ID="OLCF_${SEQ_LEN}_${PRED_LEN}"
    echo ""
    echo "------ Evaluating ${MODEL_NAME} with pred_len ${PRED_LEN} ------"
    echo "Model ID: ${MODEL_ID}"
    COMMAND="python -u run.py --task_name ${TASK_NAME} --is_training ${IS_TRAINING} --root_path ${ROOT_PATH} --data_path ${DATA_PATH} --model_id ${MODEL_ID} --model ${MODEL_NAME} --data ${DATASET} --features ${FEATURES} --seq_len ${SEQ_LEN} --label_len ${LABEL_LEN} --pred_len ${PRED_LEN} --e_layers ${E_LAYERS} --d_layers ${D_LAYERS} --factor ${FACTOR} --enc_in ${ENC_IN} --dec_in ${DEC_IN} --c_out ${C_OUT} --des ${DES} --itr ${ITR} --use_multi_gpu"
    echo "Executing command:"
    echo "${COMMAND}"
    ${COMMAND}
    echo "------ Finished evaluation for pred_len ${PRED_LEN} ------"
done

echo "===================================================="
echo "Finished all evaluations for Model: ${MODEL_NAME}"
echo "====================================================="
echo ""