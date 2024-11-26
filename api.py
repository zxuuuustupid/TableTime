from openai import OpenAI
import torch.nn as nn
import time

class api_output(nn.Module):
    def __init__(self,temperature,top_p,max_tokens):
        super(api_output, self).__init__()
        self.client = OpenAI(api_key='',base_url='',)
        self.temperature=temperature
        self.top_p=top_p
        self.max_tokens=max_tokens

    def forward(self, content):
        while True:
            response = self.client.chat.completions.create(
                model='',
                messages=[{"role": "user", "content":content}],
                temperature=self.temperature,
                top_p=self.top_p,
                max_tokens=self.max_tokens,
                stream=False,
                frequency_penalty=0,
                presence_penalty=0)
            try:
                result = response.choices[0].message.content
                return result
            except:
                print("llm request error!")
                time.sleep(5)
