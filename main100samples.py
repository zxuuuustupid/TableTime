from datetime import datetime
import os
import numpy as np
from src.ts_encoding import ts2DFLoader, ts2html, ts2markdown, ts2json
import json
from src.api import api_output, api_output_openai, api_output_openai_xiaomi

import torch.nn as nn

ts_encoding_dict = {'DFLoader': ts2DFLoader, 'html': ts2html, 'markdown': ts2markdown, 'json': ts2json}
dist_name = {'DTW': 'Dynamic Time Warping (DTW)', 'ED': 'euclidean', 'SED': 'standard euclidean',
             'MAN': 'Manhattan distance'}
data_dict = {'DFLoader': 'DFLoader', 'html': 'HTML', 'markdown': 'MarkDown', 'json': 'JSON'}
number_dict={1:'closest',2:'second',3:'third',4:'fourth',5:'fifth',6:'sixth',7:'seventh',8:'eighth',9:'ninth',10:'tenth'}

class FM_PD(nn.Module):
    def __init__(self, dataset, dist, nei_number, encoding_style, channel_list, itr,llm_name,temperature,top_p,max_tokens,n_sample,frequency,time_use):
        super(FM_PD, self).__init__()
        self.x_train = np.load(f'data/{dataset}/X_train.npy', mmap_mode='c')
        self.y_train = np.load(f'data/{dataset}/y_train.npy', mmap_mode='c')
        self.x_test = np.load(f'data/{dataset}/X_valid.npy', mmap_mode='c')
        self.y_test = np.load(f'data/{dataset}/y_valid.npy', mmap_mode='c')
        with open(f'./data_index/{dataset}/{dist}_dist/nearest_{nei_number}_neighbors.json',
                  'r') as f:
            self.data_index = json.load(f)
        self.ts_encoding = ts_encoding_dict[encoding_style](channel_list,n_sample,frequency,time_use)
        self.nei_number = nei_number
        self.dist = dist
        # self.llm = api_output(api=api, llm_name=llm_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        # self.llm = api_output(model=llm_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        self.llm = api_output_openai(model=llm_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        # self.llm = api_output_openai_xiaomi(model=llm_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        
        self.dataset = dataset
        self.encoding_style = encoding_style
        self.itr = itr
        self.doc = data_dict[encoding_style]  
        # self.llm_name = llm_name.replace('/', '_')
        self.llm_name = llm_name
        self.channel_list = channel_list
        self.base_path = f'result/{self.dataset}/{self.doc}/{self.dist}_dist/100samples'
        self.log_dir = os.path.join(self.base_path, 'txt')
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_prefix = f'FM_{self.nei_number}_{self.encoding_style}_{self.dist}_{self.itr}_{self.llm_name}_{self.timestamp}'

    def forward(self):
        answer = []
        for j in range(1): 

            # print(test)
            # print("\033[34m" + str(test) + "\033[0m")
            prompt = (
                '**Role:** You are an expert in high-speed train drivetrain system operation and maintenance, fault diagnosis, and signal processing.\n'
                f'**Goal:** Based on the provided {len(self.channel_list)}-channel time-series data of a high-speed train drivetrain system, perform fault diagnosis and classification on test samples using methods such as time-frequency domain analysis.\n\n'
                
                '### 1. Detailed Dataset Description\n'
                'This dataset is provided by the National Key Laboratory of Advanced Rail Autonomous Operation at Beijing Jiaotong University, derived from fault simulation experiments of a subway train bogie drivetrain system.\n'
                '*   **System Composition:** The power drivetrain chain includes a motor, a reduction gearbox, and an axle box.\n'
                '*   **Drive Source:** Three-phase asynchronous AC motor (speed controlled by an inverter, loaded by a hydraulic device).\n'
                '*   **Motor Bearings:** SKF 6205-2RSH.\n'
                '*   **Gearbox:** Helical gears; driving gear has 16 teeth, driven gear has 107 teeth.\n'
                '*   **Driving Gear Support Bearings:** HRB32305.\n'
                '*   **Axle Box Bearings:** HRB352213.\n'
                '*   **State Definitions:**\n'
                '    *   `0`: health (healthy state without faults)\n'
                '    *   `1`: fault (state with faults)\n'
                '*   **Sampling Parameters:** 24 channels covering vibration, current, speed, and sound. Sampling frequency: **64kHz**.\n\n'

                '### 2. Sampling Channel Locations and Physical Significance\n'
                '1. **Traction Motor:** CH1-CH3 (DE Vibration, g), CH4-CH6 (NDE Vibration, g), CH7-CH9 (Three-phase Current, A), CH10 (Rotational Speed, V).\n'
                '2. **Gearbox:** CH11-CH13 (Input Shaft Vibration, g), CH14-CH16 (Output Shaft Vibration, g).\n'
                '3. **Left Axle Box:** CH17-CH19 (Vibration, g), CH20 (Sound, Pa).\n'
                '4. **Right Axle Box:** CH21-CH23 (Vibration, g), CH24 (Sound, Pa).\n\n'
                f'**Current Data Channels:** The current data includes the following channels: {", ".join(self.channel_list)}\n'
                

                '### 3. Diagnosis Strategy: Similarity Analysis and Clustering Optimization\n'
                f'*   **{dist_name[self.dist]}:** For each test sample, we use {dist_name[self.dist]}to select the most similar samples from the training set. the most similar neighboring samples have been selected from the training set.\n'
                '*   **Clustering Logic:** Treat these similar samples as a cluster. Analyze signal feature consistency and label distribution to assist decision-making.\n\n'

                '### 4. Task Requirements\n'
                '1. **Analysis:** Extract features from the test sample and analyze whether fault signatures exist.\n'
                '2. **Classification:** Determine if the test sample is `health` or `fault`.\n'
                '3. **Clustering Optimization:** Utilize provided similarity label patterns to optimize your result.\n\n'

                '### 5. Constraints (Strictly Enforced)\n'
                '*   **Output Format:** You MUST output a SINGLE JSON List containing results for ALL samples provided below.\n'
                '*   **JSON Structure:** `[{"id": 0, "result": "health"}, {"id": 1, "result": "fault"}, ...]`\n'
                '*   **Content:** The "result" must be `health` or `fault`. Do NOT output any analysis text.\n\n'

                '### 6. Data to Process\n')
                
            for i in range(self.x_test.shape[0]):
                x_use = self.x_test[i]
                nei_index=[]
                nei_value=[]
                nei_label=[]
                nei_enc=[]
                for j in range(self.nei_number):
                    # nei_index.append(self.data_index[i]['nearest_neighbors'][j])
                    nei_index.append(self.data_index[i]['neighbors'][j])
                    nei_value.append(self.x_train[nei_index[j]])
                    nei_label.append(self.y_train[nei_index[j]])
                    nei_enc.append(self.ts_encoding(nei_value[j]))
                test = self.ts_encoding(x_use)  # 测试集编码
                prompt += f'\n\n============= BATCH SAMPLE ID: {i} =============\n'
                prompt+=('**[1. Similar Samples (Training Set)]**\n')
                for k in range(self.nei_number):
                    prompt+= (
                        f'-------Neighbor{ k+1 }-------\n'
                        f'{k+1}**Sample (the {number_dict[k+1]} training sample to the test sample:**- Data (several channels, 100 time steps per channel):{nei_enc[k]}\n ' 
                        f'- Label:{nei_label[k]}\n')
                    
                prompt+= (
                    '**[2. Test Sample to Predict]**\n'
                    f'**Data:** {test}\n\n'
                    ''
                )
                
            prompt += (
    '\n\n##################################################\n'
                '**END OF DATA BATCH.**\n'
                f'I have provided {self.x_test.shape[0]} samples (ID 0 to {self.x_test.shape[0]-1}).\n'
                'Please immediately generate the JSON List containing predictions for ALL samples.\n'
                'Remember: NO analysis text, ONLY JSON.\n'
                'JSON Output:'
            )
            
            import tiktoken
            def print_token_report(text_content, model_limit=128000):
                """
                计算 Token 并打印详细的对比报告
                model_limit: 默认为 128k (目前主流长文本模型的标准上限)
                """
                encoding = tiktoken.get_encoding("cl100k_base")
                token_count = len(encoding.encode(text_content))
                char_count = len(text_content)
                ratio = token_count / model_limit

                print(f"Est Tokens:  {token_count:,}", end=' ')
                print(f"Limit Tokens: {model_limit:,}")
            print_token_report(prompt, model_limit=128000)
            # print("\033[34m" + str(prompt) + "\033[0m")
            est_tokens = len(prompt) * 0.8
            # print(f"Prompt Tokens: {est_tokens:.0f} ---")
            output = self.llm(content=prompt)
            
            print(f"Test index {i}:")
            print(output)
            
            # output = self.llama(role='user', content=prompt)
            log_dir = self.log_dir
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with open(
    os.path.join(log_dir, f'FM_log_{self.nei_number}_{self.encoding_style}_{self.dist}_{self.itr}_{self.llm_name}_{self.timestamp}.txt'),
                    'a', encoding='utf-8') as file:
                file.write(f'{i}\n')
                file.write(output)
                file.write('\n')
            answer.append({'test_index': i, 'answer': output})
        
        json_file_path = os.path.join(self.base_path, f'{self.file_prefix}.json')
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(answer, f, ensure_ascii=False, indent=4)
        print(f"Final results saved to: {json_file_path}")
        return answer

if __name__ == "__main__":
    # Parameters
    dataset = 'BJTU-gearbox'
    dist = 'DTW'
    nei_number = 5
    encoding_style = 'json'
    # channel_list = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'O1', 'O2']
    channel_list = ['CH11','CH12','CH13','CH14','CH15','CH16']
 
    itr = 1
    
    # llm_name = 'mimo-v2-flash'
    # llm_name = 'glm-4.5-flash'
    llm_name = 'deepseek-v3.2'
    temperature = 0.7
    top_p = 1.0
    max_tokens = 4096

    # Instantiate and run the model
    model = FM_PD(
        dataset=dataset,
        dist=dist,
        nei_number=nei_number,
        encoding_style=encoding_style,
        channel_list=channel_list,
        # api=api,
        itr=itr,
        llm_name=llm_name,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        n_sample=100,
        frequency=64000,
        time_use=True
    )
    results = model.forward()
    # with open('result.json', 'w', encoding='utf-8') as f:
    #     json.dump(results, f, ensure_ascii=False, indent=4)
    # print("Results saved to result.json")