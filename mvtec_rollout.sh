#!/bin/bash
#以下是启动swift强化学习的推理脚本
# --- 1. 使用环境变量限制 vLLM 并发 (代替报错的命令行参数) ---
export VLLM_MAX_NUM_SEQS=16
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# --- 2. 基础环境设置 ---
export QWENVL_BBOX_FORMAT='new'
export CUDA_VISIBLE_DEVICES=2,3
export NCCL_SOCKET_IFNAME=ens15f0
export NCCL_SOCKET_FAMILY=AF_INET
export NCCL_SHM_DISABLE=1
export NCCL_COLLNET_ENABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=WARN
export PYTHONWARNINGS=ignore
export TOKENIZERS_PARALLELISM=false
export TRITON_CACHE_DIR=/tmp/triton_cache

swift rollout \
    --model /data0/jycheng/homework/MLLM_qwen3vl/sft/finetune/v11-20251228-003741/checkpoint-20 \
    --vllm_data_parallel_size 1 \
    --vllm_tensor_parallel_size 2 \
    --truncation_strategy 'delete' \
    --port 8001 \
    --host 0.0.0.0

# # 下面的代码是用来启动 vLLM API 服务器的脚本。不是swift rollout命令。
# python -m vllm.entrypoints.openai.api_server \
#     --model /data0/jycheng/homework/MLLM_qwen3vl/output/qwen3vl_2b/v7-20251225-223310/checkpoint-100 \
#     --port 8001 \
#     --trust-remote-code \
#     --gpu-memory-utilization 0.95 \
#     --tensor-parallel-size 2
