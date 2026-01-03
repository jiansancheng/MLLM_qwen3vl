"""
文件名: eval_vllm_save.py
功能: 使用 Swift VllmEngine 进行并行推理，计算详细指标，并保存带 Summary 的标准 JSON 文件。
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,6,7"
import json
import numpy as np
import re
from typing import Dict, List, Tuple, Any
import warnings
warnings.filterwarnings('ignore')
from swift.llm import VllmEngine, InferRequest, RequestConfig

class SwiftVLLMEvaluator:
    def __init__(self, model_path: str, tensor_parallel_size: int = 1, gpu_memory_utilization: float = 0.9):
        print(f"🚀 初始化 vLLM 引擎: {model_path}")
        self.engine = VllmEngine(
            model_id_or_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization
        )

    def load_test_data(self, test_file: str) -> List[Dict]:
        print(f"📂 加载测试数据: {test_file}")
        data = []
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try: data.append(json.loads(line))
                    except: pass
        return data

    def prepare_infer_requests(self, data: List[Dict]) -> Tuple[List[InferRequest], List[int]]:
        system_prompt = "你是专业的工业视觉异常检测助手,需精准分析图像变化。"
        user_prompt_template = """你是工业质检人员, 请对比下面两张同一位置不同时间的产品图像,请分析后一张图片相对于前一张图片发生的变化,将产品的异常变化结果以包含"status"和"changes"键的JSON格式输出,"changes"中每个变化包含"bbox"和"description"键。
如果产品没有发生异常变化,则直接输出:
{
"status": "无异常",
"changes": []
}
如果产品发生异常变化，用bbox框标注出每个变化的区域,并分别用一句话描述该区域的异常变化,例如:
{
"status": "异常",
"changes": [
    {
    "bbox": [x1, y1, x2, y2],
    "description": "fold on the leather"
    },
    {
    "bbox": [x1, y1, x2, y2],
    "description": "rough on the tile"
    }
]
}"""
        requests = []
        indices = []
        for idx, sample in enumerate(data):
            images = sample.get('images', [])
            if not images and 'image_a_path' in sample:
                images = [sample['image_a_path'], sample['image_b_path']]
            if len(images) < 2: continue

            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt_template},
                        {"type": "image", "image": images[0]},
                        {"type": "image", "image": images[1]}
                    ]
                }
            ]
            requests.append(InferRequest(messages=messages))
            indices.append(idx)
        return requests, indices

    def extract_thinking_and_response(self, output_text: str) -> Tuple[str, str]:
        think_start = output_text.find('<think>')
        think_end = output_text.find('</think>')
        thinking = ""
        response = output_text
        if think_start != -1 and think_end != -1:
            thinking = output_text[think_start + 7:think_end].strip()
            response = output_text[think_end + 8:].strip()
        return thinking, response

    def parse_json_response(self, response_text: str) -> Dict:
        clean_text = re.sub(r'^```json\s*', '', response_text.strip(), flags=re.MULTILINE)
        clean_text = re.sub(r'^```\s*', '', clean_text, flags=re.MULTILINE)
        clean_text = clean_text.strip('`').strip()
        try: return json.loads(clean_text)
        except:
            start = clean_text.find('{')
            end = clean_text.rfind('}') + 1
            if start != -1 and end > start:
                try: return json.loads(clean_text[start:end])
                except: pass
        return {"status": "解析失败", "changes": []}

    def calculate_iou(self, box1, box2):
        try:
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2
            inter_xmin = max(x1_min, x2_min)
            inter_ymin = max(y1_min, y2_min)
            inter_xmax = min(x1_max, x2_max)
            inter_ymax = min(y1_max, y2_max)
            if inter_xmax < inter_xmin or inter_ymax < inter_ymin: return 0.0
            inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
            box1_area = (x1_max - x1_min) * (y1_max - y1_min)
            box2_area = (x2_max - x2_min) * (y2_max - y2_min)
            union_area = box1_area + box2_area - inter_area
            return inter_area / union_area if union_area > 0 else 0.0
        except: return 0.0

    def process_and_save_results(self, data: List[Dict], results: List[Any], indices: List[int], output_file: str):
        print("\n📊 正在计算详细统计指标...")
        
        # --- 统计计数器 ---
        stats = {
            'total': 0,
            'normal_gt_count': 0,      # 真值是正常的数量
            'anomaly_gt_count': 0,     # 真值是异常的数量
            'normal_correct': 0,       # 正常样本判断正确的数量
            'anomaly_correct': 0,      # 异常样本判断正确的数量 (状态正确)
            'bbox_correct': 0,         # IoU > 0.5 的数量
            'total_iou': 0.0           # 用于计算 mIoU
        }

        processed_samples = []

        for i, resp in enumerate(results):
            idx = indices[i]
            sample = data[idx]
            
            # 1. 获取输入
            images = sample.get('images', [])
            if not images and 'image_a_path' in sample:
                images = [sample['image_a_path'], sample['image_b_path']]
            
            # 2. 解析输出
            full_output = resp.choices[0].message.content
            thinking, json_text = self.extract_thinking_and_response(full_output)
            pred_json = self.parse_json_response(json_text)
            
            # 3. 获取真值
            gt_data = sample.get('solution') or sample.get('label') or {}
            
            # --- 4. 核心指标计算 ---
            gt_status = gt_data.get('status', '无异常')
            pred_status = pred_json.get('status', '未知')
            
            is_gt_anomaly = (gt_status == '异常')
            is_correct_status = (gt_status == pred_status)
            
            stats['total'] += 1
            if is_gt_anomaly:
                stats['anomaly_gt_count'] += 1
                if is_correct_status: stats['anomaly_correct'] += 1
            else:
                stats['normal_gt_count'] += 1
                if is_correct_status: stats['normal_correct'] += 1
            
            # 计算 IoU (仅对 GT=异常 且 Pred=异常 的情况计算，其他情况 IoU=0)
            max_iou = 0.0
            if is_gt_anomaly and pred_status == '异常':
                gt_changes = gt_data.get('changes', [])
                pred_changes = pred_json.get('changes', [])
                
                if gt_changes and pred_changes:
                    # 简化逻辑：对每个 pred 找最大的 GT 匹配
                    # 注意：严格评测可能需要匈牙利匹配，这里做简单可视化评估即可
                    current_ious = []
                    for p_box in pred_changes:
                        best_box_iou = 0.0
                        for g_box in gt_changes:
                            if 'bbox' in p_box and 'bbox' in g_box:
                                iou = self.calculate_iou(p_box['bbox'], g_box['bbox'])
                                best_box_iou = max(best_box_iou, iou)
                        current_ious.append(best_box_iou)
                    
                    if current_ious:
                        max_iou = max(current_ious) # 取预测框中最好的一个展示
                        stats['total_iou'] += np.mean(current_ious) # 平均 IoU 累加
                    
                    if max_iou >= 0.5:
                        stats['bbox_correct'] += 1

            # 构造样本数据
            result_item = {
                "id": idx,
                "image_a": images[0],
                "image_b": images[1],
                "gt": gt_data,
                "pred": pred_json,
                "thinking": thinking,
                "metrics": {
                    "status_correct": is_correct_status,
                    "max_iou": max_iou
                }
            }
            processed_samples.append(result_item)

        # --- 5. 汇总 Summary ---
        summary_metrics = {
            "total_samples": stats['total'],
            "accuracy_all": round((stats['normal_correct'] + stats['anomaly_correct']) / stats['total'], 4) if stats['total'] > 0 else 0,
            
            # 正常样本统计
            "normal_count": stats['normal_gt_count'],
            "normal_acc": round(stats['normal_correct'] / stats['normal_gt_count'], 4) if stats['normal_gt_count'] > 0 else 0,
            
            # 异常样本统计
            "anomaly_count": stats['anomaly_gt_count'],
            "anomaly_acc": round(stats['anomaly_correct'] / stats['anomaly_gt_count'], 4) if stats['anomaly_gt_count'] > 0 else 0,
            
            # 定位统计 (分母为 GT 是异常的数量，还是检测出异常的数量，这里通常用 GT 异常数量作为召回参考，或用检测出的作为准确参考)
            # 这里计算：在所有 GT 是异常的样本中，成功定位 (IoU>0.5) 的比例
            "bbox_recall_iou05": round(stats['bbox_correct'] / stats['anomaly_gt_count'], 4) if stats['anomaly_gt_count'] > 0 else 0,
            "detected_anomalies": stats['anomaly_correct'] # 正确检测出是异常的个数
        }

        final_output = {
            "summary": summary_metrics,
            "data": processed_samples
        }

        print(f"\n📈 评估完成:")
        print(f"   Total: {stats['total']}")
        print(f"   Acc: {summary_metrics['accuracy_all']:.2%}")
        print(f"   Normal Acc: {summary_metrics['normal_acc']:.2%} ({stats['normal_correct']}/{stats['normal_gt_count']})")
        print(f"   Anomaly Acc: {summary_metrics['anomaly_acc']:.2%} ({stats['anomaly_correct']}/{stats['anomaly_gt_count']})")

        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(final_output, f_out, ensure_ascii=False, indent=2)

        print(f"✅ 结果已保存至: {output_file}")

    def run(self, test_file: str, output_file: str):
        data = self.load_test_data(test_file)
        infer_requests, indices = self.prepare_infer_requests(data)
        request_config = RequestConfig(max_tokens=2048, temperature=0.01, top_p=0.9) # 温度调低，保证稳定性
        
        print(f"🚀 开始推理 ({len(infer_requests)} 样本)...")
        results = self.engine.infer(infer_requests, request_config=request_config, use_tqdm=True)
        self.process_and_save_results(data, results, indices, output_file)

if __name__ == "__main__":
   # MODEL_PATH = "/data0/jycheng/homework/MLLM_qwen3vl/sft/finetune/v11-20251228-003741/checkpoint-20"
    # MODEL_PATH = "/data0/limh/models/Qwen3-VL-2B-Thinking"
    MODEL_PATH = "/data0/jycheng/homework/MLLM_qwen3vl/output/qwen3vl_2b/v9-20251229-025757/checkpoint-138"#GRPO强化学习后的模型
    TEST_FILE = "/data0/jycheng/homework/MLLM_qwen3vl/dataset/大模型作业数据集/test.jsonl"#测试集
    OUTPUT_FILE = "/data0/jycheng/homework/MLLM_qwen3vl/test_results_grpo.json" # 确保是 .json
    
    evaluator = SwiftVLLMEvaluator(MODEL_PATH, tensor_parallel_size=2)
    evaluator.run(TEST_FILE, OUTPUT_FILE)