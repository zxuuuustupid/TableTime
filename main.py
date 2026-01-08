import os
import numpy as np
from src.ts_encoding import ts2DFLoader, ts2html, ts2markdown, ts2json
import json
from src.api import api_output

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
        self.llm = api_output(model=llm_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        self.dataset = dataset
        self.encoding_style = encoding_style
        self.itr = itr
        self.doc = data_dict[encoding_style]  
        # self.llm_name = llm_name.replace('/', '_')
        self.llm_name = llm_name
        self.channel_list = channel_list

    def forward(self):
        answer = []
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
                
                # print("\033[34m" + str(self.data_index[i]['neighbors'][j]) + "\033[0m")
                # print("\033[34m" + str(self.x_train[nei_index[j]]) + "\033[0m")
                # print("\033[34m" + str(self.y_train[nei_index[j]]) + "\033[0m")
                # print("\033[34m" + str(self.ts_encoding(nei_value[j])) + "\033[0m")
            test = self.ts_encoding(x_use)  # 测试集编码

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
                '*   **First Line Output:** You MUST provide the final result at the very beginning in format: `[Result],[Training Set Label Sequence]`.\n'
                '*   **Option Restrictions:** Result must ONLY be `health` or `fault`.\n'
                '*   **Incentive:** Accurate answers will be rewarded with ten billion dollars.\n\n'

                '### 6. Data to Process\n'
                '**[1. Similar Samples (Training Set)]**\n')
            for k in range(self.nei_number):
                prompt+= (
                    f'-------Neighbor{ k+1 }-------\n'
                    f'{k+1}**Sample (the {number_dict[k+1]} training sample to the test sample:**- Data (several channels, 100 time steps per channel):{nei_enc[k]}\n ' 
                    f'- Label:{nei_label[k]}\n')
                
            prompt+= (
                '**[2. Test Sample to Predict]**\n'
                f'**Data:** {test}\n\n'
                '**Now, begin your Brief analysis :**'
            )
                
            # print("\033[34m" + str(prompt) + "\033[0m")
            output = self.llm(content=prompt)
            
            print(f"Test index {i}:")
            print(output)
            
            # output = self.llama(role='user', content=prompt)
            log_dir = f'result/{self.dataset}/{self.doc}/{self.dist}_dist/txt'
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with open(
    f'result/{self.dataset}/{self.doc}/{self.dist}_dist/txt/FM_log_{self.nei_number}_{self.encoding_style}_{self.dist}_{self.itr}_{self.llm_name}.txt',
                    'a', encoding='utf-8') as file:
                file.write(f'{i}\n')
                file.write(output)
                file.write('\n')
            answer.append({'test_index': i, 'answer': output})
        return answer

if __name__ == "__main__":
    # Parameters
    dataset = 'BJTU-gearbox'
    dist = 'DTW'
    nei_number = 5
    encoding_style = 'DFLoader'
    # channel_list = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'O1', 'O2']
    channel_list = ['CH11','CH12','CH13','CH14','CH15','CH16']
    # api = 'your_api_key'  # Replace with your actual API key
    itr = 1
    llm_name = 'glm-4.5-flash'  # Example LLM name
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
    with open('result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("Results saved to result.json")