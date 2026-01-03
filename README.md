# MLLM_qwen_simplify

精简版多模态工业质检项目，包含数据处理、训练与推理脚本，便于快速复现和评估。

## 环境准备
- Python 3.10+，CUDA 按需安装
- 推荐先创建虚拟环境：
  ```bash
  conda create -n mllm_qwen python=3.10 -y
  conda activate mllm_qwen
  ```
- 安装依赖：
  ```bash
  pip install -r requirements.txt
  ```

## 数据
- 数据默认放在 `dataset/` 下，包含 MVTEC-AD 及处理后的标注swift_format_dataset.jsonl和swift_format_dataset_grpo.jsonl分别是sft和grpo阶段所用到的数据集
- 可使用 `dataset/大模型作业数据集/visualize_dataset.html` 查看样例与结果。

## 训练与推理
-
- 训练/微调与强化（按需）：
  ```bash
  bash mvtec_process.sh   # 预处理数据，分割训练集dataset/大模型作业数据集/train.jsonl
  bash mvtec_rollout.sh   # 采样/推理
  bash mvtec_grpo.sh      # GRPO 训练
  ```
- 测试集推理：
  ```bash
  python test_vllm_save.py # vllm并行推理，得到测试指标与结果json
  ```

## 结果
- 关键指标与对比结果位于 `test_results_*.json` 与 `visualize_test_*.html`。


