#!/bin/bash

# ======================= TRAINING SCRIPT =======================
accelerate launch \
--config_file accelerate_configs/single_node.yaml \
--main_process_port 12345 \
train.py \
--batch-size 2 \
--gradient-accumulate-every 4 \
--output-dir ./output/infini-attn \
--wandb Infini-Attn \
--seed 2024 \
--max-train-steps 100 \
--learning-rate 2e-5 \
--dataset HuggingFaceDataset/slimpajama_optimized \
--model output/optimized-danube2-2.0b-base \
--seq-length 16000 \
--rope-theta 30000 \
--parallel_mode data_parallel
