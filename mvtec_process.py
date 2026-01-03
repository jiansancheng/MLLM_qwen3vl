import json
import os
import random
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def _extract_object_name_from_path(image_path: str) -> str:
    """
    从图片路径中解析物体类别名称(MVTEC-AD下一级目录),例如:
    /.../MVTEC-AD/bottle/bottle/test/... -> bottle
    /.../MVTEC-AD/cable/cable/test/...   -> cable
    """
    try:
        parts = Path(image_path).parts
        if "MVTEC-AD" in parts:
            idx = parts.index("MVTEC-AD")
            if idx + 1 < len(parts):
                return parts[idx + 1]
    except Exception:
        pass
    # 兜底: 取父目录名
    return Path(image_path).parent.name


def _ensure_object_in_description(description: str, object_name: str) -> str:
    """
    规范化异常描述,确保包含 'on the <object_name>'
    - 若描述为空 -> 'anomaly on the <object_name>'
    - 若描述已包含 'on the' 且结尾就是 'on the' -> 补齐 object_name
    - 若不包含 'on the' -> 追加 ' on the <object_name>'
    """
    d = (description or "").strip()
    if not d:
        return f"anomaly on the {object_name}"
    low = d.lower()
    if "on the" in low:
        # 结尾是 'on the' 或 'on the ' -> 补齐物体名称
        if low.rstrip().endswith("on the"):
            return d.rstrip() + f" {object_name}"
        return d
    return f"{d} on the {object_name}"


def convert_to_swift_format(input_jsonl, output_jsonl, dataset_base_path):
    """
    将异常检测数据集转换为Swift框架所需的message格式
    
    Args:
        input_jsonl: 输入的jsonl文件路径
        output_jsonl: 输出的jsonl文件路径
        dataset_base_path: 数据集基础路径,用于构建完整的图片路径
    """
    
    # Swift框架的提示词
    system_prompt = "你是专业的工业视觉异常检测助手,需精准分析图像变化。"
    user_prompt = (
        """你是工业质检人员,请对比下面两张同一位置不同时间的产品图像,请分析后一张图片相对于前一张图片发生的变化,将产品的异常变化结果以包含"status"和"changes"键的JSON格式输出,"changes"中每个变化包含"bbox"和"description"键。
如果产品没有发生异常变化,则直接输出:
{
"status": "无异常",
"changes": []
}
如果产品发生异常变化,用bbox框标注出每个变化的区域,并分别用一句话描述该区域的异常变化,格式为"异常类别" on the "物体种类名称",例如:
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
}
   """
    )
    
    converted_samples = []
    skipped_count = 0
    
    print(f"开始处理: {input_jsonl}")
    
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                data = json.loads(line.strip())
                
                # 构建完整的图片路径 - 将 dataset/ 替换为 MVTEC-AD/
                image_a_path = data['image_a_path'].replace('dataset/', '', 1)
                image_b_path = data['image_b_path'].replace('dataset/', '', 1)
                
                image_a = os.path.join(dataset_base_path, "MVTEC-AD", image_a_path)
                image_b = os.path.join(dataset_base_path, "MVTEC-AD", image_b_path)
                
                # 验证图片是否存在
                if not os.path.exists(image_a):
                    print(f"警告: 图片A不存在,跳过样本 (行 {line_num}): {image_a}")
                    skipped_count += 1
                    continue
                    
                if not os.path.exists(image_b):
                    print(f"警告: 图片B不存在,跳过样本 (行 {line_num}): {image_b}")
                    skipped_count += 1
                    continue
                
                # 获取图片B的尺寸(用于归一化bbox)
                try:
                    img_b = Image.open(image_b)
                    img_width, img_height = img_b.size
                    img_b.close()
                except Exception as e:
                    print(f"警告: 无法读取图片B尺寸,跳过样本 (行 {line_num}): {e}")
                    skipped_count += 1
                    continue
                
                label = data['label']
                status = label['status']
                changes = label['changes']

                # 从图片B路径解析物体名称
                object_name = _extract_object_name_from_path(image_b)

                normalized_bboxes = []
                descriptions = []

                for change in changes:
                    bbox = change['bbox']
                    raw_desc = change.get('description', '')
                    
                    # 验证bbox有效性
                    try:
                        x_min, y_min, x_max, y_max = bbox
                        if x_max <= x_min or y_max <= y_min:
                            print(f"警告: 无效bbox {bbox},跳过该异常")
                            continue
                        
                        # 归一化到0-1000范围
                        x1_norm = round((x_min / img_width) * 1000)
                        y1_norm = round((y_min / img_height) * 1000)
                        x2_norm = round((x_max / img_width) * 1000)
                        y2_norm = round((y_max / img_height) * 1000)
                        
                        normalized_bboxes.append([x1_norm, y1_norm, x2_norm, y2_norm])
                        # 统一补齐 'on the <object>'
                        descriptions.append(_ensure_object_in_description(raw_desc, object_name))
                    except (KeyError, ValueError, TypeError) as e:
                        print(f"警告: 跳过无效bbox: {e}")
                        continue
                
                # 构建答案(区分异常/无异常)
                if status == "无异常" or len(normalized_bboxes) == 0:
                    answer = {
                        "status": "无异常",
                        "changes": []
                    }
                else:
                    changes_list = []
                    for i, bbox in enumerate(normalized_bboxes):
                        changes_list.append({
                            "bbox": bbox,
                            "description": descriptions[i]
                        })
                    answer = {
                        "status": "异常",
                        "changes": changes_list
                    }
                
                # 构造message格式 - 完全匹配机房数据集格式
                message_item = {
                    "messages": [
                        {
                            "role": "system",
                            "content": system_prompt
                        },
                        {
                            "role": "user",
                            "content": user_prompt
                        },
                        {
                            "role": "assistant",
                            "content": json.dumps(answer, ensure_ascii=False)
                        }
                    ],
                    "images": [image_a, image_b],
                    "solution": answer
                }
                
                converted_samples.append(message_item)
                
            except json.JSONDecodeError as e:
                print(f"错误: JSON解析失败 (行 {line_num}): {e}")
                skipped_count += 1
                continue
            except Exception as e:
                print(f"错误: 处理样本失败 (行 {line_num}): {e}")
                skipped_count += 1
                continue
    
    # 保存转换后的数据
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    with open(output_jsonl, 'w', encoding='utf-8') as f:
        for item in tqdm(converted_samples, desc="保存数据"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"\n转换完成!")
    print(f"  - 成功转换: {len(converted_samples)} 个样本")
    print(f"  - 跳过样本: {skipped_count} 个")
    print(f"  - 输出文件: {output_jsonl}")
    
    return converted_samples


def split_dataset(input_jsonl, train_output, test_output, train_ratio=0.8, random_seed=42):
    """
    将数据集划分为训练集和测试集
    
    Args:
        input_jsonl: 输入的jsonl文件路径
        train_output: 训练集输出路径
        test_output: 测试集输出路径
        train_ratio: 训练集比例(默认0.8)
        random_seed: 随机种子,确保可重复性
    """
    
    print(f"\n开始划分数据集...")
    print(f"训练集比例: {train_ratio*100:.1f}%")
    print(f"测试集比例: {(1-train_ratio)*100:.1f}%")
    
    # 设置随机种子
    random.seed(random_seed)
    
    # 读取所有数据
    all_samples = []
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                all_samples.append(json.loads(line.strip()))
    
    print(f"总样本数: {len(all_samples)}")
    
    # 随机打乱数据
    random.shuffle(all_samples)
    
    # 按比例划分
    split_point = int(len(all_samples) * train_ratio)
    train_samples = all_samples[:split_point]
    test_samples = all_samples[split_point:]
    
    print(f"训练集: {len(train_samples)} 个样本")
    print(f"测试集: {len(test_samples)} 个样本")
    
    # 保存训练集
    os.makedirs(os.path.dirname(train_output), exist_ok=True)
    with open(train_output, 'w', encoding='utf-8') as f:
        for item in tqdm(train_samples, desc="保存训练集"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"训练集已保存至: {train_output}")
    
    # 保存测试集
    os.makedirs(os.path.dirname(test_output), exist_ok=True)
    with open(test_output, 'w', encoding='utf-8') as f:
        for item in tqdm(test_samples, desc="保存测试集"):
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"测试集已保存至: {test_output}")


if __name__ == "__main__":
    print("="*60)
    print("Swift框架-工业异常检测数据集转换工具")
    print("="*60)
    
    # 配置路径
    input_jsonl = "/data0/jycheng/homework/MLLM_qwen3vl/dataset/大模型作业数据集/output_dataset_grpo_filtered.jsonl"
    output_jsonl = "/data0/jycheng/homework/MLLM_qwen3vl/dataset/大模型作业数据集/swift_format_dataset_grpo.jsonl"
    train_jsonl = "/data0/jycheng/homework/MLLM_qwen3vl/dataset/大模型作业数据集/train_grpo.jsonl"
    test_jsonl = "/data0/jycheng/homework/MLLM_qwen3vl/dataset/大模型作业数据集/test_grpo.jsonl"
    dataset_base_path = "/data0/jycheng/homework/MLLM_qwen3vl/dataset/大模型作业数据集"
    
    print(f"输入文件: {input_jsonl}")
    print(f"转换输出: {output_jsonl}")
    print(f"训练集: {train_jsonl}")
    print(f"测试集: {test_jsonl}")
    print("="*60)
    
    # 执行转换
    try:
        # 第一步: 转换格式
        print("\n【第一步】转换数据格式...")
        convert_to_swift_format(input_jsonl, output_jsonl, dataset_base_path)
        
        # 第二步: 划分数据集
        print("\n【第二步】划分训练测试集...")
        split_dataset(output_jsonl, train_jsonl, test_jsonl, train_ratio=0.8)
        
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
        import traceback
        traceback.print_exc()
    else:
        print(f"\n✅ 所有操作完成!")
        print(f"   - 转换数据: {output_jsonl}")
        print(f"   - 训练集: {train_jsonl}")
        print(f"   - 测试集: {test_jsonl}")
