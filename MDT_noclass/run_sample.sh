#!/bin/bash


pip install -e .

MODEL_PATH=output_mdt_xl2/model010000.pt
export OPENAI_LOGDIR=output_mdt_xl2_eval
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=4,5,6,7

#echo 'CFG Class-conditional sampling:'
#MODEL_FLAGS="--image_size 256 --model MDT_XL_2 --decode_layer 2"
#DIFFUSION_FLAGS="--num_sampling_steps 250 --num_samples 50000  --cfg_cond True"
#echo $MODEL_FLAGS
#echo $DIFFUSION_FLAGS
#echo $MODEL_PATH
#python -m torch.distributed.launch --nproc_per_node=$NUM_GPUS scripts/image_sample.py --model_path $MODEL_PATH $MODEL_FLAGS $DIFFUSION_FLAGS
#echo $MODEL_FLAGS
#echo $DIFFUSION_FLAGS
#echo $MODEL_PATH
#python evaluations/evaluator.py ../dataeval/VIRTUAL_imagenet256_labeled.npz $OPENAI_LOGDIR/samples_50000x256x256x3.npz

echo 'Class-conditional sampling:'
MODEL_FLAGS="--image_size 256 --model MDT_XL_2 --decode_layer 2"
DIFFUSION_FLAGS="--num_sampling_steps 250 --num_samples 100"
echo $MODEL_FLAGS
echo $DIFFUSION_FLAGS
echo $MODEL_PATH
torchrun --nproc_per_node=$NUM_GPUS scripts/image_sample.py --model_path $MODEL_PATH $MODEL_FLAGS $DIFFUSION_FLAGS
echo $MODEL_FLAGS
echo $DIFFUSION_FLAGS
echo $MODEL_PATH
#python evaluations/evaluator.py ../dataeval/VIRTUAL_imagenet256_labeled.npz $OPENAI_LOGDIR/samples_50000x256x256x3.npz

