# src/code_executor.py
import re, sys, io, os, json, traceback, ast
import pandas as pd
import numpy as np
import scipy.stats as stats
import scipy.signal as signal

def extract_code(text):
    pattern = r"```python\s*(.*?)\s*```"
    match = re.search(pattern, text, re.DOTALL)
    return match.group(1) if match else None

import ast

def get_main_function_name(code_str):
    """
    Safely parses a Python code string to find the name of the function
    called within the `if __name__ == '__main__':` block.

    Args:
        code_str: The string containing the Python code.

    Returns:
        The name of the main function as a string, or None if not found or on error.
    """
    try:
        # 1. Parse the code string into an Abstract Syntax Tree (AST)
        tree = ast.parse(code_str)

        # 2. Walk through all nodes in the tree
        for node in ast.walk(tree):
            # 3. Find the 'if' statement that matches the __main__ check
            if (isinstance(node, ast.If) and
                isinstance(node.test, ast.Compare) and
                isinstance(node.test.left, ast.Name) and
                node.test.left.id == '__name__' and
                isinstance(node.test.ops[0], ast.Eq) and
                isinstance(node.test.comparators[0], ast.Constant) and
                node.test.comparators[0].value == '__main__'):
                
                # 4. Once the block is found, search for a function call inside it
                for sub_node in node.body:
                    # Check if the node is an expression containing a call
                    if (isinstance(sub_node, ast.Expr) and
                        isinstance(sub_node.value, ast.Call)):
                        
                        # Get the name of the function being called
                        if isinstance(sub_node.value.func, ast.Name):
                            # 5. Return the function name and exit
                            return sub_node.value.func.id
    except (SyntaxError, IndexError):
        # If the code is not valid Python, parsing will fail
        return None
        
    # If no __main__ block or no function call inside it is found
    return None

def execute_generated_code(code_str, test_path, train_data_path, nei_map_path, result_save_path):
    old_stdout = sys.stdout
    redirected_output = io.StringIO()
    sys.stdout = redirected_output

    local_scope = {
        "pd": pd, "np": np, "stats": stats, "signal": signal, "json": json,
        "TEST_DATA_PATH": test_path,
        "TRAIN_DATA_PATH": train_data_path,
        "NEI_MAP_PATH": nei_map_path,
        "RESULT_SAVE_PATH": result_save_path
    }

    try:
        # 编译并加载所有函数定义
        compiled_code = compile(code_str, '<string>', 'exec')
        exec(compiled_code, local_scope)
        
        # [关键的“智能点火”逻辑]
        main_function_name = get_main_function_name(code_str)
        if main_function_name and main_function_name in local_scope:
            print(f"DEBUG: Manually invoking main function found in __main__: {main_function_name}()")
            local_scope[main_function_name]()
        else:
            # 后备方案，如果没找到 __main__
            fallback_names = ["main", "analyze_test_samples", "run_analysis"]
            called = False
            for name in fallback_names:
                if name in local_scope and callable(local_scope[name]):
                    print(f"DEBUG: Found and calling fallback function: {name}()")
                    local_scope[name]()
                    called = True
                    break
            if not called:
                 raise RuntimeError("No `if __name__ == '__main__'` block or standard main function found.")
        
        # [终极保险丝] 如果文件还没创建，但 results 变量已生成，我们帮它保存
        if not os.path.exists(result_save_path) and 'results' in local_scope:
            results_data = local_scope['results']
            if isinstance(results_data, list):
                print("[DEBUG] Force-saving 'results' variable to the target file...")
                with open(result_save_path, 'w', encoding='utf-8') as f:
                    json.dump(results_data, f, indent=2)

        success = True
    except Exception:
        success = False
        error_details = traceback.format_exc()
    finally:
        sys.stdout = old_stdout

    output_from_prints = redirected_output.getvalue()
    
    # 最终返回逻辑
    if not success:
        return f"[FATAL_EXECUTION_ERROR]\n{error_details}"
    
    if os.path.exists(result_save_path):
        return f"[SUCCESS] Result file was created.\n" + output_from_prints
    else:
        return f"[ERROR] Code ran without crashing, but the output file was NOT created.\nLog:\n{output_from_prints}"