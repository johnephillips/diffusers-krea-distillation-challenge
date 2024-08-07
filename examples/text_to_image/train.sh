#!/usr/bin/env bash
export CHECKPOINT_DIR="/data/john-challenge"
export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE_NAME="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="lambdalabs/naruto-blip-captions"
# export TRAIN_DATA_DIR="/blobdata/challenge_data/train"
# export TRAIN_TEXT_TO_IMAGE_DIR="/home/ubuntu/john-ai-challenge/diffusers-krea-distillation-challenge/examples/text_to_image"
timestamp=$(date "+%Y.%m.%d-%H.%M.%S")

accelerate launch train_text_to_image_sdxl.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE_NAME \
  --dataset_name=$DATASET_NAME \
  --max_train_samples=32 \
  --resolution=1024 \
  --center_crop \
  --random_flip \
  --proportion_empty_prompts=0.2 \
  --train_batch_size=2 \
  --max_train_steps=100 \
  --learning_rate=2e-05 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --mixed_precision="fp16" \
  --validation_prompt "a man with dark hair and brown eyes" "a man in a hoodie with a fire in the background" "a man with a red hair and a black shirt" "a man in a blue shirt and headband" \
  --validation_epochs 5 \
  --checkpointing_steps=5000 \
  --output_dir="$CHECKPOINT_DIR/sdxl-naruto-model-pruned-student-all-losses-896-samples-$timestamp" \
  --report_to="wandb" \
  --lambda_out=1e-10 \
  --lambda_feat_kd=1e-10 \
  #--max_train_steps=10000 \
  #--checkpointing_steps=5000 \
  #--validation_epochs 5 \
  #--train_batch_size=1 \