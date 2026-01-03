from swift.plugin import ORM, orms
from typing import List, Dict, Any, Union
import json
import re


def parse_json_response(response: str) -> Union[Dict, None]:
    """解析模型响应中的JSON格式"""
    # 移除可能的思考标签
    cleaned_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL | re.IGNORECASE)
    
    # 尝试多种JSON匹配模式
    json_patterns = [
        r'\{[^{}]*"status"[^{}]*"changes"[^{}]*\}',
        r'\{.*?"status".*?"changes".*?\}'
    ]
    
    for pattern in json_patterns:
        for match in re.finditer(pattern, cleaned_response, re.DOTALL):
            try:
                json_str = match.group(0)
                # 尝试修复常见的JSON格式问题
                json_str = re.sub(r'([{,]\s*)(\w+):', r'\1"\2":', json_str)
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    
    # 最后尝试直接解析整个响应
    try:
        return json.loads(cleaned_response.strip())
    except json.JSONDecodeError:
        return None


def calculate_bbox_iou(bbox1: List[float], bbox2: List[float]) -> float:
    """计算两个bbox的IoU"""
    if len(bbox1) < 4 or len(bbox2) < 4:
        return 0.0
    
    x1_1, y1_1, x2_1, y2_1 = bbox1[:4]
    x1_2, y1_2, x2_2, y2_2 = bbox2[:4]
    
    # 计算交集
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = area1 + area2 - intersection_area
    
    if union_area <= 0:
        return 0.0
    
    return intersection_area / union_area


def find_best_bbox_matches(pred_bboxes: List[List[float]], gt_bboxes: List[List[float]]) -> float:
    """找到预测bbox与真实bbox的最佳匹配并计算平均IoU"""
    if not pred_bboxes or not gt_bboxes:
        return 0.0
    
    # 构建IoU矩阵
    iou_matrix = []
    for pred_bbox in pred_bboxes:
        iou_row = []
        for gt_bbox in gt_bboxes:
            iou = calculate_bbox_iou(pred_bbox, gt_bbox)
            iou_row.append(iou)
        iou_matrix.append(iou_row)
    
    # 贪心匹配:为每个预测框找到最佳的真实框
    used_gt_indices = set()
    matched_ious = []
    
    for i, pred_bbox in enumerate(pred_bboxes):
        best_iou = 0.0
        best_gt_idx = -1
        for j, gt_bbox in enumerate(gt_bboxes):
            if j not in used_gt_indices and iou_matrix[i][j] > best_iou:
                best_iou = iou_matrix[i][j]
                best_gt_idx = j
        
        if best_gt_idx != -1:
            used_gt_indices.add(best_gt_idx)
            matched_ious.append(best_iou)
        else:
            matched_ious.append(0.0)
    
    avg_iou = sum(matched_ious) / len(matched_ious) if matched_ious else 0.0
    return avg_iou


def accuracy_reward_func(pred_status: str, gt_status: str) -> float:
    """计算状态准确率奖励"""
    anomaly_keywords = ['异常']
    normal_keywords = ['无异常']
    
    def classify_status(status_str):
        s = status_str.strip()
        for kw in anomaly_keywords:
            if kw in s:
                return 'anomaly'
        for kw in normal_keywords:
            if kw in s:
                return 'normal'
        return 'unknown'
    
    pred_cls = classify_status(pred_status)
    gt_cls = classify_status(gt_status)
    
    return 1.0 if (pred_cls == gt_cls and pred_cls != 'unknown') else 0.0


def iou_reward_func(pred_changes: List[Dict], gt_changes: List[Dict], count_weight: float = 0.3) -> float:
    """计算IoU奖励(包含数量和位置准确性)"""
    # 提取预测的bbox
    pred_bboxes = []
    for change in pred_changes:
        if isinstance(change, dict) and 'bbox' in change:
            bbox = change['bbox']
            if isinstance(bbox, list) and len(bbox) >= 4:
                pred_bboxes.append(bbox)
    
    # 提取真实的bbox
    gt_bboxes = []
    for change in gt_changes:
        if isinstance(change, dict) and 'bbox' in change:
            bbox = change['bbox']
            if isinstance(bbox, list) and len(bbox) >= 4:
                gt_bboxes.append(bbox)
    
    # 如果都没有异常,返回满分
    if len(pred_bboxes) == 0 and len(gt_bboxes) == 0:
        return 1.0
    # 如果一方有一方没有,返回0分
    if len(pred_bboxes) == 0 or len(gt_bboxes) == 0:
        return 0.0
    
    # 数量准确性奖励
    count_reward = 1.0 if len(pred_bboxes) == len(gt_bboxes) else 0.0
    
    # 位置准确性奖励(平均IoU)
    avg_iou = find_best_bbox_matches(pred_bboxes, gt_bboxes)
    
    # 综合奖励
    return count_weight * count_reward + (1 - count_weight) * avg_iou


def format_reward_func(response: str) -> float:
    """计算格式正确性奖励"""
    format_score = 0.0
    
    # 检查是否包含JSON结构
    json_result = parse_json_response(response)
    if json_result is not None:
        format_score += 0.3  # 基础JSON格式正确
        
        # 检查必需字段
        if 'status' in json_result and 'changes' in json_result:
            format_score += 0.3  # 包含必需字段
            
            # 检查changes格式
            changes = json_result['changes']
            if isinstance(changes, list):
                format_score += 0.2  # changes是列表
                
                # 检查每个change的格式
                if changes:
                    valid = all(
                        isinstance(c, dict) and 'bbox' in c and 'description' in c
                        for c in changes
                    )
                    if valid:
                        format_score += 0.2  # 所有change格式正确
                else:
                    # 空列表也是有效的(表示无异常)
                    format_score += 0.2
    
    return min(1.0, format_score)


# =============== ORM Classes for Swift Framework ===============

class MVTECFormatReward(ORM):
    """MVTEC数据集格式奖励"""
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        return [format_reward_func(resp) for resp in completions]


class MVTECAccuracyReward(ORM):
    """MVTEC数据集准确率奖励"""
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        solution = kwargs.get('solution', [])
        rewards = []
        
        for i, resp in enumerate(completions):
            pred = parse_json_response(resp)
            if pred is None:
                rewards.append(0.0)
                continue
            
            pred_status = pred.get("status", "")
            gt = solution[i] if i < len(solution) else {"status": "无异常"}
            gt_status = gt.get("status", "")
            
            rewards.append(accuracy_reward_func(pred_status, gt_status))
        
        return rewards


class MVTECIoUReward(ORM):
    """MVTEC数据集IoU奖励"""
    def __call__(self, completions: List[str], **kwargs) -> List[float]:
        solution = kwargs.get('solution', [])
        rewards = []
        
        for i, resp in enumerate(completions):
            pred = parse_json_response(resp)
            if pred is None:
                rewards.append(0.0)
                continue
            
            gt = solution[i] if i < len(solution) else {"status": "无异常", "changes": []}
            
            pred_status = pred.get("status", "")
            gt_status = gt.get("status", "")
            
            # 如果status不一致,直接返回0
            if ("异常" in gt_status) != ("异常" in pred_status):
                rewards.append(0.0)
                continue
            
            pred_changes = pred.get("changes", [])
            gt_changes = gt.get("changes", [])
            
            rewards.append(iou_reward_func(pred_changes, gt_changes, count_weight=0.3))
        
        return rewards


# =============== Register ORM Functions ===============
orms['mvtec_format_reward'] = MVTECFormatReward
orms['mvtec_accuracy_reward'] = MVTECAccuracyReward
orms['mvtec_iou_reward'] = MVTECIoUReward
