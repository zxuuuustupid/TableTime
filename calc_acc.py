import json
import numpy as np
import re
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import pandas as pd
import os

def parse_llm_output(answer_string):
    """
    从 LLM 的原始输出中解析出预测标签。
    兼容 'G3,[G1,G2...]' 或 'G3' 或 'G3, some text' 等格式。
    """

    if not isinstance(answer_string, str):
        return None
        
    match = re.match(r"^(G\d+)", answer_string.strip())
    if match:
        return match.group(1)
    return None

def load_true_labels_from_json(true_labels_path):
    """
    从 JSON 文件加载真实标签，并转换成 {index: label} 的字典。
    """
    try:
        with open(true_labels_path, 'r', encoding='utf-8') as f:
            true_labels_list = json.load(f)
        # 假设 JSON 格式是 [{"index": 0, "label": "G0"}, ...]
        return {item['index']: item['label'] for item in true_labels_list}
    except FileNotFoundError:
        print(f"[ERROR] True labels file not found: {true_labels_path}")
        return None
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[ERROR] Invalid format in true labels file {true_labels_path}. Expected list of dicts with 'index' and 'label'. Error: {e}")
        return None

def analyze_json_results(result_file_path, true_labels_path, llm_name=""):
    """
    分析多分类结果，与从 JSON 文件加载的真实标签进行比较。

    Args:
        result_file_path (str): 预测结果 JSON 文件的路径.
        true_labels_path (str): 真实标签 JSON 文件的路径.
        llm_name (str): 模型名称，用于打印报告.
    """
    # 1. 加载真实标签
    true_labels_map = load_true_labels_from_json(true_labels_path)
    if true_labels_map is None:
        return # 如果加载失败，直接退出

    # 2. 加载预测结果
    try:
        with open(result_file_path, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] Prediction result file not found: {result_file_path}")
        return
    except json.JSONDecodeError:
        print(f"[ERROR] Invalid JSON format in prediction file: {result_file_path}")
        return

    y_pred = []
    y_true = []
    
    format_invalid_count = 0

    # 3. 匹配预测与真实标签
    for item in pred_data:
        if not isinstance(item, dict): continue
        
        idx = item.get("test_index")
        ans = item.get("answer", "")
        
        # 确保该 index 在真实标签中存在
        if idx is not None and idx in true_labels_map:
            predicted_label = parse_llm_output(ans)
            
            if predicted_label:
                y_pred.append(predicted_label)
                y_true.append(true_labels_map[idx])
            else:
                format_invalid_count += 1
                print(f"[WARNING] Format invalid for test_index {idx}. Answer: '{ans[:50]}...'")
        else:
             print(f"[WARNING] test_index {idx} from result file not found in true labels file. Skipping.")


    # --- 4. 计算并打印评估报告 ---
    print("\n" + "="*60)
    print(f"[EVALUATION REPORT] for {os.path.basename(result_file_path)}")
    if llm_name:
        print(f"   Model: {llm_name}")
    print("="*60)

    total_samples = len(true_labels_map)
    evaluated_samples = len(y_true)
    
    print(f"Total Samples in Ground Truth: {total_samples}")
    print(f"Evaluated Samples (Format-Valid & Matched): {evaluated_samples}")
    print(f"Format-Invalid Predictions: {format_invalid_count}")
    
    if not y_true:
        print("\n[ERROR] No valid predictions to evaluate. Cannot calculate metrics.")
        return

    labels = sorted(list(set(y_true) | set(y_pred)))

    # --- 核心指标 ---
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro', labels=labels, zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', labels=labels, zero_division=0)

    print("\n--- Overall Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f} (Treats each class equally)")
    print(f"Weighted F1-Score: {weighted_f1:.4f} (Accounts for class imbalance)")

    # --- 分类报告 ---
    print("\n--- Classification Report ---")
    report = classification_report(y_true, y_pred, labels=labels, target_names=labels, zero_division=0)
    print(report)

    # --- 混淆矩阵 ---
    print("\n--- Confusion Matrix ---")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print("Rows: True Labels, Columns: Predicted Labels")
    print(cm_df)
    print("="*60 + "\n")


# --- 使用示例 ---
if __name__ == '__main__':
    # 1. 指定你的预测结果文件路径
    result_json_path = r'path/to/your/result.json'
    
    # 2. 指定你的真实标签文件路径
    true_labels_json_path = r'path/to/your/true_labels.json'
    
    model_name = 'deepseek-v3.2'
    
    if os.path.exists(result_json_path) and os.path.exists(true_labels_json_path):
         analyze_json_results(result_json_path, true_labels_json_path, model_name)
    else:
        print(f"Path incorrect:\n  - Result: {result_json_path}\n  - True Labels: {true_labels_json_path}")