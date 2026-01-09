import json

def analyze_json_results(file_path):
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

    print(f"总样本数: {total_items}")
    print(f"符合格式规范数 (以 label+逗号开头): {format_valid_count}")
    print(f"规范样本中分类正确数: {correct_count}")
    
    if format_valid_count > 0:
        accuracy = (correct_count / format_valid_count) * 100
        print(f"基于规范样本的准确率: {accuracy:.2f}%")
    else:
        print("准确率: N/A (没有符合规范的样本)")

# 使用方法：将 'result.json' 替换为你的文件名
print("Deepseek-v3.2")
analyze_json_results('result\\BJTU-gearbox\\DFLoader\\DTW_dist\\FM_5_DFLoader_DTW_1_deepseek-v3.2.json')
print("GLM-4.5-Flash")
analyze_json_results('result\\BJTU-gearbox\\DFLoader\\DTW_dist\\FM_5_DFLoader_DTW_1_glm-4.5-flash.json')
print("MIMO-V2-Flash")
analyze_json_results('result\\BJTU-gearbox\\DFLoader\\DTW_dist\\FM_5_DFLoader_DTW_1_mimo-v2-flash.json')