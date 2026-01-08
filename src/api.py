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
