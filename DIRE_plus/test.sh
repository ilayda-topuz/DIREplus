#!/bin/bash

### make sure that you have modified the EXP_NAME, CKPT, DATASETS_TEST
#eval "$(conda shell.bash hook)"

export CUDA_VISIBLE_DEVICES=1,2,3,4
EXP_NAME="AE_model290000"
CKPT="model_epoch_best.pth" # classifier
DATASETS_TEST="ImageNet"
GENERATOR="adm"  # generator model to test
python test.py --gpus 4 --ckpt $CKPT --exp_name $EXP_NAME datasets_test $DATASETS_TEST generator $GENERATOR