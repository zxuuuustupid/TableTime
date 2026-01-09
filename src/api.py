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
                    # print(" API returned an empty response. Retrying...")
                    time.sleep(1)

            except Exception as e:
                print(f"Exception occurred: {e}")
                time.sleep(2)


# import os
# import time
# import json
# import traceback
# from dotenv import load_dotenv
# from openai import OpenAI
# import torch.nn as nn

# class api_output_openai(nn.Module):
#     def __init__(self, model, temperature, top_p, max_tokens):
#         super().__init__()
#         load_dotenv()
#         self.client = OpenAI(
#             api_key=os.getenv("OPENAI_API_KEY"),
#             base_url="https://api.zhizengzeng.com/v1"
#         )
#         self.model = model
#         self.params = {
#             "temperature": temperature,
#             "top_p": top_p,
#             "max_tokens": max_tokens,
#             "stream": False
#         }

#     def forward(self, content):
#         max_retries = 3
#         retries = 0

#         while retries < max_retries:
#             try:
#                 print(f"[DEBUG] Attempting OpenAI API call (try {retries + 1}/{max_retries})...")
#                 response = self.client.chat.completions.create(
#                     model=self.model,
#                     messages=[{"role": "user", "content": content}],
#                     **self.params
#                 )

#                 # [关键修改 1] 打印完整的服务器原始响应，用于调试
#                 # 将 response 对象转为字典再用 json 打印，格式清晰
#                 raw_response_dict = response.model_dump()
#                 print("--- [DEBUG] Full Server Response ---")
#                 print(json.dumps(raw_response_dict, indent=2, ensure_ascii=False))
#                 print("------------------------------------")
                
#                 # 健壮性检查
#                 if not response.choices:
#                     print(f"[WARNING] 'choices' field is empty. Retrying... (try {retries + 1}/{max_retries})")
#                     retries += 1
#                     time.sleep(2)
#                     continue
                
#                 # 进一步检查 message content
#                 message = response.choices[0].message
#                 result = message.content.strip() if message and message.content else ""
                
#                 if result:
#                     return result
                
#                 # [关键修改 2] 详细说明为什么是 Empty
#                 finish_reason = response.choices[0].finish_reason
#                 print(f"[WARNING] Received empty 'content'. Finish Reason: '{finish_reason}'. Retrying... (try {retries + 1}/{max_retries})")
#                 retries += 1
#                 time.sleep(1)

#             except Exception as e:
#                 # [关键修改 3] 提供最详尽的异常信息
#                 print("="*60)
#                 print(f"[FATAL] An Exception Occurred During OpenAI API Call (try {retries + 1}/{max_retries})")
#                 print(f"   - Error Type: {type(e).__name__}")
#                 print(f"   - Error Message: {e}")
                
#                 # OpenAI 的 SDK 错误通常会把详细信息放在 e.response.text 或 e.body
#                 if hasattr(e, 'response') and hasattr(e.response, 'text'):
#                     print(f"   - Server Response Body: {e.response.text}")
#                 elif hasattr(e, 'body'):
#                     print(f"   - Error Body: {e.body}")

#                 print("   - Full Traceback:")
#                 traceback.print_exc()
#                 print("="*60)
                
#                 retries += 1
#                 time.sleep(2)
        
#         print("[ERROR] API call failed after multiple retries.")
#         return "[API_CALL_FAILED]"

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
                
                print("[WARNING] Empty response, retrying...")
                time.sleep(1)

            except Exception as e:
                print(f"[ERROR] API Error: {e}")
                time.sleep(2)
                
                
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
                
                print("[WARNING] Empty response, retrying...")
                time.sleep(1)

            except Exception as e:
                print(f"[ERROR] API Error: {e}")
                time.sleep(2)
                