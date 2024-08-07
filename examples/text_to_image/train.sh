#!/usr/bin/env bash
export CHECKPOINT_DIR="/data/john-challenge"
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="lambdalabs/naruto-blip-captions"
export TRAIN_DATA_DIR="/data/challenge_data/"
# export TRAIN_DATA_DIR="/blobdata/challenge_data/train"
# export TRAIN_TEXT_TO_IMAGE_DIR="/home/ubuntu/john-ai-challenge/diffusers-krea-distillation-challenge/examples/text_to_image"
timestamp=$(date "+%Y.%m.%d-%H.%M.%S")

accelerate launch train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --train_data_dir=$TRAIN_DATA_DIR \
  --cache_dir="/data/challenge_data/cache" \
  --max_train_samples=6400 \
  --resolution=1024 \
  --center_crop \
  --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size=4 \
  --max_train_steps=2001 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --validation_prompt "a man with dark hair and brown eyes" "a man in a hoodie with a fire in the background" "a man with a red hair and a black shirt" "a man in a blue shirt and headband" "street photography capturing a moment of a man interacting with a vintage car in a tropical setting." "This image depicts a musical performance, specifically a choir concert, where individuals dressed in robes are singing, accompanied by musicians playing instruments." \
  --validation_epochs 1 \
  --checkpointing_steps=1000 \
  --max_train_steps=5000 \
  --output_dir="$CHECKPOINT_DIR/sdxl-pexels-pruned-student-6400-samples-bsz-4-lr-2e-05-res-1024-lambda_out-5e-02-lambda_feat_kd-5e-02-$timestamp" \
  --report_to="wandb" \
  --lambda_out=5e-02 \
  --lambda_feat_kd=5e-02 \
  #--checkpointing_steps=5000 \
  #--validation_epochs 5 \
  #--train_batch_size=1 \
  #--dataset_name=$DATASET_NAME \