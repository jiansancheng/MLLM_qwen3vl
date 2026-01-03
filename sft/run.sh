# 格式转换
# python convert_to_qwen_format.py \
#   --input output_dataset.jsonl \
#   --dataset_root /opt/data/private/gaoj/GaoJing/curriculum/Fundamentals_and_Applications_of_Large_Models/Model_Finetune/dataset \
#   --output_train train.jsonl

# 微调
# IMAGE_MAX_TOKEN_NUM=1024 \
# CUDA_VISIBLE_DEVICES=4 \
# swift sft \
#   --model ./cache/Qwen/Qwen3-VL-2B-Instruct \
#   --dataset ./train.jsonl \
#   --train_type lora \
#   --torch_dtype bfloat16 \
#   --per_device_train_batch_size 1 \
#   --learning_rate 1e-4 \
#   --lora_rank 8 \
#   --lora_alpha 32 \
#   --target_modules all-linear \
#   --freeze_vit true \
#   --freeze_aligner true \
#   --gradient_checkpointing true \
#   --max_length 4096 \
#   --output_dir ./output



# --- 1. 核心修复：强制 WandB 离线 & 本地网络直连 ---
set -euo pipefail

export WANDB_MODE=offline
export NO_PROXY="localhost,127.0.0.1,0.0.0.0"
export no_proxy="$NO_PROXY"   # 兼容

# --- 2. 环境变量配置 ---
export QWENVL_BBOX_FORMAT='new'
# export WANDB_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  # 离线模式无需硬编码API Key
export WANDB_PROJECT=homework
export WANDB_EXP_NAME=mvtec_sft2
export WANDB_DIR=/data0/jycheng/homework/MLLM_qwen3vl/sft/wandb
# 训练命令
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 \
MAX_PIXELS=1605632 swift sft \
--model /data0/limh/models/Qwen3-VL-2B-Thinking \
--dataset /data0/jycheng/homework/MLLM_qwen3vl/dataset/大模型作业数据集/train.jsonl \
--train_type full \
--freeze_vit false \
--freeze_aligner false \
--freeze_llm false \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--split_dataset_ratio 0.1 \
--output_dir /data0/jycheng/homework/MLLM_qwen3vl/sft/finetune \
--num_train_epochs 6 \
--save_steps 20 \
--eval_steps 20 \
--save_total_limit 2 \
--logging_steps 10 \
--seed 42 \
--learning_rate 2e-5 \
--weight_decay 0.01 \
--adam_beta1 0.9 \
--adam_beta2 0.95 \
--adam_epsilon 1e-08 \
--gradient_accumulation_steps 64 \
--max_grad_norm 1 \
--lr_scheduler_type cosine \
--warmup_ratio 0.05 \
--gradient_checkpointing True \
--bf16 true \
--run_name "${WANDB_EXP_NAME}-$(date +%Y%m%d-%H%M%S)" \
--report_to wandb
# 如果完全不需要 W&B，可改为： --report_to none

# swift export \
#   --model ./cache/Qwen/Qwen3-VL-2B-Instruct \
#   --adapters ./output/v5-20251218-223510/checkpoint-402 \
#   --merge-lora true \
#   --output-dir ./final_qwen3vl_2b_merged
