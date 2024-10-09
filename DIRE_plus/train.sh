#!/bin/bash

### make sure that you have modified the EXP_NAME, DATASETS, DATASETS_TEST
#eval "$(conda shell.bash hook)"

EXP_NAME="GI_model290000"
DATASETS="GenImage"
DATASETS_TEST="GenImage"
GENERATOR="sd1-5"
python train.py --gpus 0 --exp_name $EXP_NAME datasets $DATASETS datasets_test $DATASETS_TEST generator $GENERATOR
