#!/bin/bash

MODEL=bert-base-uncased # model path bert-base-uncased / roberta-base/ roberta-large
NUM_CLASS=2
MODEL_NAME=BERT # BERT / RoBERTa
BSZ=16
EPOCH=10
PATIENT=3
LR=1e-5
SEED=42
SEQ_LENS=128
LAYERS=12
HEADS=12
OPTION=train
CUDA="cuda:5"

mkdir log_ft

nohup python main.py \
--bert_path ${MODEL} \
--num_class ${NUM_CLASS} \
--model ${MODEL_NAME} \
--epoch ${EPOCH} \
--layers ${LAYERS} \
--heads ${HEADS} \
--patient ${PATIENT} \
--PTM_learning_rate ${LR} \
--max_len ${SEQ_LENS} \
--option ${OPTION} \
--seed ${SEED} \
--main_cuda ${CUDA} \
--batch_size ${BSZ} > log_ft/log_ft.log 2>&1


