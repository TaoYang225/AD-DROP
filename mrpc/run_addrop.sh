#!/bin/bash

MODEL=bert-base-uncased # model path bert-base-uncased / roberta-base/ roberta-large
NUM_CLASS=2
MODEL_NAME=BERT # BERT / RoBERTa
BSZ=16
EPOCH=50
PATIENT=5
LR=1e-5
SEED=42
SEQ_LENS=100
LAYERS=12
OPTION=train
CUDA="cuda:3"
ATTR=IGA # GA IGA AA RD
MASK_LAYERS="0,1,2,3,4,5,6,7,8,9,10,11"


#mkdir log_${MODEL}_${ATTR}_${SEED}
#
#for P in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
#do
#  for Q in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
#  do
#    nohup python main.py \
#    --bert_path ${MODEL} \
#    --num_class ${NUM_CLASS} \
#    --do_mask \
#    --model ${MODEL_NAME} \
#    --epoch ${EPOCH} \
#    --layers ${LAYERS} \
#    --patient ${PATIENT} \
#    --PTM_learning_rate ${LR} \
#    --max_len ${SEQ_LENS} \
#    --option ${OPTION} \
#    --seed ${SEED} \
#    --main_cuda ${CUDA} \
#    --batch_size ${BSZ} \
#    --attribution ${ATTR} \
#    --p_rate ${P} \
#    --q_rate ${Q} \
#    --mask_layers ${MASK_LAYERS} > log_${MODEL}_${ATTR}_${SEED}/P_${P}_Q_${Q}.log 2>&1
#  done
#done

python main.py \
--bert_path ${MODEL} \
--num_class ${NUM_CLASS} \
--do_mask \
--model ${MODEL_NAME} \
--epoch ${EPOCH} \
--layers ${LAYERS} \
--patient ${PATIENT} \
--PTM_learning_rate ${LR} \
--max_len ${SEQ_LENS} \
--option ${OPTION} \
--seed ${SEED} \
--main_cuda ${CUDA} \
--batch_size ${BSZ} \
--attribution ${ATTR} \
--p_rate 0.1 \
--q_rate 0.2 \
--mask_layers ${MASK_LAYERS}