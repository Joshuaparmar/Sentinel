#!/bin/bash
# export CUDA_VISIBLE_DEVICES=0

python -u run.py \
  --task_name $TASK_NAME \
  --is_training $IS_TRAINING \
  --root_path $ROOT_PATH \
  --model_id $DATASET \
  --model Crossformer \
  --data $DATASET \
  --features $FEATURES \
  --seq_len $SEQ_LEN \
  --pred_len $PRED_LEN \
  --d_model 128 \
  --d_ff 128 \
  --e_layers 3 \
  --enc_in $ENC_IN \
  --c_out $C_OUT \
  --anomaly_ratio $ANOMALY_RATIO \
  --batch_size $BATCH_SIZE \
  --train_epochs $TRAIN_EPOCHS\
  --anomaly_tolerance $ANOMALY_TOLERANCE \
  --xid $XID \
  --use_multi_gpu
