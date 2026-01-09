import json

def analyze_json_results(file_path,llm_name):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {file_path}")
        return

    total_items = len(data)
    format_valid_count = 0  # 符号规范的条数
    correct_count = 0       # 预测正确的条数

    for item in data:
        idx = item.get("test_index")
        ans = item.get("answer", "")

        # --- 1. 检查规范性 ---
        # 规范要求：以 health 或 fault 开头，且紧跟一个逗号
        is_format_ok = False
        predicted_label = ""

        if ans.startswith("health,"):
            is_format_ok = True
            predicted_label = "health"
        elif ans.startswith("fault,"):
            is_format_ok = True
            predicted_label = "fault"
        
        if is_format_ok:
            format_valid_count += 1

            # --- 2. 检查正确性 ---
            # 标准：0-49 为 health，50-99 为 fault
            standard_label = "health" if 0 <= idx <= 49 else "fault"
            
            if predicted_label == standard_label:
                correct_count += 1

    # 输出结果

    print(f"[INFO] Evaluate json file: {file_path} Use Model: {llm_name}")
    print(f"[EVAL] Total Samples: {total_items}")
    print(f"[EVAL] Format Valid Samples: {format_valid_count}")
    print(f"[EVAL] Correctly Classified in Valid Samples: {correct_count}")
    
    if format_valid_count > 0:
        accuracy = (correct_count / format_valid_count) * 100
        print(f"[EVAL] Accuracy Based on Valid Samples: {accuracy:.2f}%")
    else:
        print("[WARNING] No valid formatted samples to calculate accuracy.")

# 使用方法：将 'result.json' 替换为你的文件名
print("Deepseek-v3.2")
analyze_json_results(r'F:\Project\TableGPT\TableTime\result\BJTU-gearbox\DFLoader\DTW_dist\FM_5_DFLoader_DTW_1_deepseek-v3.2_20260109_182550.json')