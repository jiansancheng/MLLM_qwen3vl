"""
将MVTec数据集转换为Qwen-VL SFT训练格式
"""
import json
import os
from pathlib import Path

INPUT_FILE = "output_dataset.jsonl"
OUTPUT_FILE = "qwen_sft_dataset.json"
DATASET_ROOT = r"e:\C盘文件迁移\桌面\大模型基础与应用"

def convert_to_qwen_format():
    """转换为Qwen-VL的SFT格式"""
    qwen_data = []
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            
            # 构造绝对路径
            image_a_abs = os.path.join(DATASET_ROOT, sample['image_a_path'])
            image_b_abs = os.path.join(DATASET_ROOT, sample['image_b_path'])
            
            # 构造问题prompt
            question = f"请对比这两张图像，第一张是参考图像（正常状态），第二张是待检测图像。请分析待检测图像是否存在异常，如果存在异常请标注出缺陷的位置和类型。"
            
            # 构造答案
            status = sample['label']['status']
            changes = sample['label']['changes']
            
            if status == "无异常":
                answer = "检测结果：无异常\n待检测图像与参考图像相比，未发现任何缺陷或异常情况。"
            else:
                answer = f"检测结果：异常\n"
                for i, change in enumerate(changes, 1):
                    bbox = change['bbox']
                    desc = change['description']
                    answer += f"\n异常{i}:\n"
                    answer += f"- 位置: {bbox}\n"
                    answer += f"- 描述: {desc}\n"
            
            # Qwen-VL的多图输入格式
            qwen_sample = {
                "id": f"mvtec_{len(qwen_data)}",
                "conversations": [
                    {
                        "from": "user",
                        "value": f"Picture 1: <img>{image_a_abs}</img>\nPicture 2: <img>{image_b_abs}</img>\n{question}"
                    },
                    {
                        "from": "assistant", 
                        "value": answer
                    }
                ]
            }
            
            qwen_data.append(qwen_sample)
    
    # 保存
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(qwen_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 转换完成！")
    print(f"总样本数: {len(qwen_data)}")
    print(f"输出文件: {OUTPUT_FILE}")
    
    # 显示示例
    print("\n样本示例：")
    print(json.dumps(qwen_data[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    convert_to_qwen_format()
