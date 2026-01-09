# src/code_executor.py
import re
import sys
import io
import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.signal as signal
import json  # <--- [必须加] 否则 LLM 写 json.load 会报错

def extract_code(text):
    """提取 Markdown 代码块"""
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None

def execute_generated_code(code_str, test_path, train_data_path, nei_map_path, result_save_path):
    """
    执行代码，注入数据文件路径和邻居索引文件路径
    """
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output

    local_scope = {
        "pd": pd, "np": np, "stats": stats, "signal": signal, "json": json,
        # 注入变量名，让 LLM 知道文件在哪
        "TEST_DATA_PATH": test_path,       # 待测样本 .npy
        "TRAIN_DATA_PATH": train_data_path,    # 训练数据 .npy
        "NEI_MAP_PATH": nei_map_path,  # <--- [关键] 邻居索引的 json 文件路径
        "RESULT_SAVE_PATH": result_save_path,  # <--- [关键] 结果保存路径
    }

    try:
        exec(code_str, local_scope)
        success = True
        error_msg = None
    except Exception as e:
        success = False
        error_msg = str(e)
    finally:
        sys.stdout = old_stdout

    output = redirected_output.getvalue()
    if not success: return f"[ERROR] Execution failed: {error_msg}"
    return output if output.strip() else "[WARNING] Code ran but printed nothing."