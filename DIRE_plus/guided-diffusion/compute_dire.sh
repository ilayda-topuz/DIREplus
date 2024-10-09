#!/bin/bash

#pip install -r requirements.txt

export CUDA_VISIBLE_DEVICES=1,2,3,4
export NCCL_P2P_DISABLE=1
MODEL_PATH="model290000.pt" # path to pre-trained MDT_noclass

SAMPLE_FLAGS="--batch_size 4 --num_samples 3000  --timestep_respacing ddim20 --use_ddim True"
SAVE_FLAGS="--images_dir images/val/GenImage/real_images/sd1-5 --recons_dir images/val/GenImage/recons_images/sd1-5 --dire_dir data/val/GenImage/1_fake_sd1-5"
MODEL_FLAGS="--mask_ratio 0.30 --decode_layer 2 --attention_resolutions 32,16,8 --class_cond False --diffusion_steps 1000 --dropout 0.1 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_fp16 False --use_scale_shift_norm True"
mpiexec -n 4 python guided-diffusion/compute_dire.py --model_path $MODEL_PATH $MODEL_FLAGS  $SAVE_FLAGS $SAMPLE_FLAGS --has_subfolder False