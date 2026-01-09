# src/code_executor.py
import re
import sys
import io
import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.signal as signal

def extract_code(text):
    """提取 Markdown 代码块"""
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None

def execute_generated_code(code_str, file_path):
    """执行代码并捕获 print 输出"""
    # 重定向标准输出，为了抓取 AI 打印的统计结果
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output

    # 给 AI 的代码注入必要的库和变量
    local_scope = {
        "pd": pd, "np": np, "stats": stats, "signal": signal,
        "DATA_PATH": file_path  # 关键：告诉 AI 数据文件在哪
    }

    try:
        exec(code_str, local_scope)
        success = True
        error_msg = None
    except Exception as e:
        success = False
        error_msg = str(e)
    finally:
        sys.stdout = old_stdout # 恢复打印功能

    output = redirected_output.getvalue()
    if not success: return f"Error: {error_msg}"
    return output if output.strip() else "Warning: Code ran but printed nothing." 