import os
from openai import OpenAI
import math
from dotenv import load_dotenv

# 初始化客户端


load_dotenv()
client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.zhizengzeng.com/v1"
        )

def get_diagnostic_probability(symptom, root_cause):
    prompt = f"""你是一位机械故障诊断专家。
针对以下现象："{symptom}"
评估根因为 "{root_cause}" 的可能性。

# 请仅输出一个词: "High" 或 "Low"。不要输出任何其他解释。
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",  # 或者 gpt-4o-mini
        messages=[{"role": "user", "content": prompt}],
        # --- 核心参数开始 ---
        logprobs=True,       # 开启概率返回
        top_logprobs=5,      # 返回得分最高的5个候选词
        max_tokens=1,        # 强制模型只吐一个词
        # --- 核心参数结束 ---
        temperature=0        # 设为0保证确定性
    )

    # 提取生成的 Token 和概率信息
    token_info = response.choices[0].logprobs.content[0]
    generated_token = token_info.token
    
    print(f"模型最终选择: {generated_token}")
    print("-" * 30)
    print("底层候选 Token 及其概率分配:")

    probs = {}
    for top_lp in token_info.top_logprobs:
        # logprob 是对数概率，需要用 e 的指数还原为 0-1 之间的概率
        prob = math.exp(top_lp.logprob)
        probs[top_lp.token.strip().lower()] = prob
        print(f"Token: '{top_lp.token}', 概率: {prob:.4f} (Logprob: {top_lp.logprob:.4f})")

    # 专门提取 High 和 Low 的对比
    high_p = probs.get("high", 0.0)
    low_p = probs.get("low", 0.0)
    
    # 归一化处理（论文中常用的方法，只看这两个分支的相对占比）
    if (high_p + low_p) > 0:
        norm_high = high_p / (high_p + low_p)
        norm_low = low_p / (high_p + low_p)
        print("-" * 30)
        print(f"归一化判定概率 -> High: {norm_high:.2%}, Low: {norm_low:.2%}")

# --- 测试运行 ---
symptom_text = "主引擎飞轮不转，起动空气压力正常，啮合开关正常。"
fault_candidate = "起动空气先导电磁阀故障"

get_diagnostic_probability(symptom_text, fault_candidate)