from openai import OpenAI
import torch.nn as nn
import time
from dotenv import load_dotenv
import os
from zai import ZhipuAiClient
from openai.types.chat import ChatCompletionUserMessageParam

class api_output(nn.Module):
    def __init__(self, model, temperature, top_p, max_tokens):
        super(api_output, self).__init__()
        load_dotenv()
        api_key = os.getenv("ZHIPU_API_KEY")
        self.client = ZhipuAiClient(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens

    def forward(self, content):
        import time
        import json

        while True:
            try:
                # print(f"[DEBUG] Sending request with content: {repr(content)}")
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                    stream=False,
                )

                # print(f"[DEBUG] Raw response: {json.dumps(response.to_dict(), indent=2, ensure_ascii=False)}")
                message = response.choices[0].message

                # 兼容所有情况
                message_dict = (
                    message.to_dict() if hasattr(message, "to_dict") else
                    getattr(message, "__dict__", {})
                )

                result = (
                        message_dict.get("content", "") or
                        message_dict.get("reasoning_content", "")
                ).strip()

                # print(f"[DEBUG] Extracted result length: {len(result)}")

                if result:
                    # print("[DEBUG] Received non-empty result.")
                    return result
                else:
                    # print("⚠️ API returned an empty response. Retrying...")
                    time.sleep(1)

            except Exception as e:
                print(f"❌ Exception occurred: {e}")
                time.sleep(2)

class api_output_openai(nn.Module):
    def __init__(self, model, temperature, top_p, max_tokens):
        super().__init__()
        load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url="https://api.zhizengzeng.com/v1"
        )
        self.model = model
        self.params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False
        }

    # def forward(self, content):
    #     while True:
    #         try:
    #             response = self.client.chat.completions.create(
    #                 model=self.model,
    #                 messages=[{"role": "user", "content": content}],
    #                 **self.params
    #             )
                
    #             # 直接获取回答内容
    #             result = response.choices[0].message.content
                
    #             if result and result.strip():
    #                 return result.strip()
                
    #             print("⚠️ Empty response, retrying...")
    #             time.sleep(1)

    #         except Exception as e:
    #             print(f"❌ API Error: {e}")
    #             time.sleep(2)
                
    def forward(self, content):
        # 1. 先打印一下输入长度，作为证据之一
        print(f"[Debug] Sending request with content length: {len(content)} characters...")
        
        retries = 0
        max_retries = 3 # 不要无限重试，防止封号或刷爆余额

        while retries < max_retries:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    **self.params
                )
                
                # 2. 检查 response 是否完整
                # 如果因为Token爆炸导致服务商返回了空数据，这里会捕获到
                if not hasattr(response, 'choices') or response.choices is None:
                    print(f"[Fatal Error] API returned invalid response (Likely Context Limit Exceeded). Raw: {response}")
                    return "ERROR_CONTEXT_LIMIT"

                # 直接获取回答内容
                result = response.choices[0].message.content
                
                if result and result.strip():
                    return result.strip()
                
                print("⚠️ Empty response, retrying...")
                retries += 1
                time.sleep(1)

            except Exception as e:
                error_msg = str(e)
                print(f"❌ API Error: {error_msg}")
                
                # 3. 关键逻辑：如果是以下错误，直接停止，不要重试！
                # 这些关键词意味着再试多少次都没用，必须报错给导师看
                fatal_keywords = [
                    "context_length_exceeded",  # 官方报错
                    "maximum context length",   # 常见报错
                    "400",                      # Bad Request
                    "NoneType",                 # 你的当前报错（数据结构崩了）
                    "rate limit"                # 此时也别硬试了
                ]
                
                if any(k in error_msg for k in fatal_keywords):
                    print("\n[CRITICAL FAILURE] STOPPING EXECUTION.")
                    print("原因: Prompt 太长，超过模型上下文窗口限制。")
                    print("建议: 请截图此报错给导师，证明 Batch 100 样本方案不可行。")
                    return "ERROR_TOKEN_OVERFLOW" # 返回错误标记，结束循环
                
                retries += 1
                time.sleep(2)
        
        return "ERROR_MAX_RETRIES"
                
                
class api_output_openai_xiaomi(nn.Module):
    def __init__(self, model, temperature, top_p, max_tokens):
        super().__init__()
        load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv("XIAOMI_API_KEY"),
            base_url="https://api.xiaomimimo.com/v1"
        )
        self.model = model
        self.params = {
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "stream": False
        }

    def forward(self, content):
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": content}],
                    **self.params
                )
                
                # 直接获取回答内容
                result = response.choices[0].message.content
                
                if result and result.strip():
                    return result.strip()
                
                print("⚠️ Empty response, retrying...")
                time.sleep(1)

            except Exception as e:
                print(f"❌ API Error: {e}")
                time.sleep(2)
                