#!/bin/bash

# --- 1. 核心修复：强制 WandB 离线 & 本地网络直连 ---
export WANDB_MODE=offline
export no_proxy="localhost,127.0.0.1,0.0.0.0"

# --- 2. 环境变量配置 ---
export QWENVL_BBOX_FORMAT='new'
export WANDB_API_KEY=295fcdeab8faf6de12c27e72872c9a9eee668585
export WANDB_PROJECT=homework
export WANDB_EXP_NAME=mvtec_sft_grpo2

# NCCL / CUDA 设置
export NCCL_SOCKET_IFNAME=ens15f0
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1
# 根据你之前的设置，这里使用 4,5,6,7 卡
export CUDA_VISIBLE_DEVICES=4,5,6,7
export NPROC_PER_NODE=4

# --- 3. 启动训练 (参数已调优) ---/data0/limh/models/Qwen3-VL-2B-Thinking
swift rlhf \
    --rlhf_type grpo \
    --run_name mvtec_sft_grpo2 \
    --model /data0/jycheng/homework/MLLM_qwen3vl/sft/finetune/v11-20251228-003741/checkpoint-20 \
    --external_plugins /data0/jycheng/homework/MLLM_qwen3vl/mvtec_reward.py \
    --reward_funcs mvtec_format_reward mvtec_accuracy_reward mvtec_iou_reward  \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8001 \
    --train_type full \
    --torch_dtype bfloat16 \
    --dataset /data0/jycheng/homework/MLLM_qwen3vl/dataset/大模型作业数据集/train_grpo.jsonl\
    --load_from_cache_file true \
    --max_completion_length 4048 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-6 \
    --gradient_accumulation_steps 4 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 100 \
    --metric_for_best_model 'reward' \
    --greater_is_better true \
    --save_steps 100 \
    --save_total_limit 3 \
    --truncation_strategy 'delete' \
    --logging_steps 1 \
    --output_dir /data0/jycheng/homework/MLLM_qwen3vl/output/qwen3vl_2b \
    --resume_from_checkpoint /data0/jycheng/homework/MLLM_qwen3vl/output/qwen3vl_2b/v8-20251229-004418/checkpoint-100 \
    --importance_sampling_level token \
    --warmup_ratio 0.01 \
    --dataloader_num_workers 8 \
    --num_generations 4 \
    --temperature 1.0 \
    --deepspeed zero3 \
    --log_completions true \
    --report_to wandb \
    --num_iterations 1 \
    --async_generate false \
    --beta 0.001