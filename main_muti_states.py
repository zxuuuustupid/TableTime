import argparse
from datetime import datetime
import os
import sys
import numpy as np
import yaml
from calc_acc import analyze_json_results
from calc_acc import *
from src.ts_encoding import ts2DFLoader, ts2html, ts2markdown, ts2json
import json
from src.code_executor import extract_code, execute_generated_code
from src.api import api_output, api_output_openai, api_output_openai_xiaomi
import torch.nn as nn
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

    print(f"[INFO] Est Tokens:  {token_count:,}", end=' ')
    print(f"Limit Tokens: {model_limit:,}", end=' ')

ts_encoding_dict = {'DFLoader': ts2DFLoader, 'html': ts2html, 'markdown': ts2markdown, 'json': ts2json}
dist_name = {'DTW': 'Dynamic Time Warping (DTW)', 'ED': 'euclidean', 'SED': 'standard euclidean',
             'MAN': 'Manhattan distance','HDF': 'Hybrid Distance Function (HDF)', 'FIW':'Feature Importance Weighted Distance (FIW)'}
data_dict = {'DFLoader': 'DFLoader', 'html': 'HTML', 'markdown': 'MarkDown', 'json': 'JSON'}
number_dict={1:'closest',2:'second',3:'third',4:'fourth',5:'fifth',6:'sixth',7:'seventh',8:'eighth',9:'ninth',10:'tenth',11:'eleventh',12:'twelfth',13:'thirteenth',14:'fourteenth',15:'fifteenth'}

def load_config(config_path):
    """加载 YAML 配置文件"""
    with open(os.path.join("config", config_path), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

class FM_PD(nn.Module):
    def __init__(self, 
                 dataset, 
                 dist, 
                 nei_number, 
                 encoding_style, 
                 channel_list, 
                 itr,
                 llm_name,
                 temperature,
                 top_p,
                 max_tokens,
                #  train_nums,
                #  test_num,
                #  n_sample,
                #  frequency,
                #  time_use
                 ):
        super(FM_PD, self).__init__()
        # self.x_train = np.load(f'data/{dataset}/X_train.npy', mmap_mode='c')
        # self.y_train = np.load(f'data/{dataset}/y_train.npy', mmap_mode='c')
        # self.x_test = np.load(f'data/{dataset}/X_valid.npy', mmap_mode='c')
        # self.y_test = np.load(f'data/{dataset}/y_valid.npy', mmap_mode='c')
        # with open(f'./data_index/{dataset}/{dist}_dist/nearest_{nei_number}_neighbors.json',
        #           'r') as f:
        #     self.data_index = json.load(f)
        # self.ts_encoding = ts_encoding_dict[encoding_style](channel_list,n_sample,frequency,time_use)
        self.nei_number = nei_number
        self.dist = dist
        if llm_name == 'glm-4.5-flash':
            print("[INFO] Using GLM-4.5-Flash model")
            self.llm = api_output(model=llm_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        elif llm_name == 'mimo-v2-flash':   
            self.llm = api_output_openai_xiaomi(model=llm_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
        else:
            self.llm = api_output_openai(model=llm_name, temperature=temperature, top_p=top_p, max_tokens=max_tokens)
            print("[INFO] Using OpenAI API model")
        self.dataset = dataset
        self.encoding_style = encoding_style
        self.itr = itr
        self.doc = data_dict[encoding_style]  
        self.llm_name = llm_name
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.channel_list = channel_list
        # # self.data_path = f'data/{dataset}/X_valid.npy'
        # self.data_path = os.path.abspath(f'{os.path.dirname(__file__)}/data/{dataset}/X_valid.npy')
        # self.base_path = f'result/{self.dataset}/{self.doc}/{self.dist}_dist'
        # self.log_dir = os.path.join(self.base_path, 'txt')
        # self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # self.file_prefix = f'FM_{self.nei_number}_{self.encoding_style}_{self.dist}_{self.itr}_{self.llm_name}_{self.timestamp}'
        self.base_result_path = f'result/{self.dataset}/{data_dict[encoding_style]}/{self.dist}_dist'
        # self.train_nums = train_nums
        # self.test_num = test_num

    def forward(self,train_nums=None,test_num=None):
        """
        train_nums: list, e.g. [1, 2]
        test_num: int, e.g. 3
        """
         # 1. 构造当前实验的唯一标识符
        train_tag = "_".join(map(str, train_nums))
        exp_id = f"test_WC{test_num}_train_WCs{train_tag}"
        
        print(f"\n[INFO] === Starting Workflow: {exp_id} ===")
        
        # test_data_path = os.path.abspath(f'data/{self.dataset}/X_valid.npy')
        # train_data_path = os.path.abspath(f'data/{self.dataset}/X_train.npy')
        # nei_map_path = os.path.abspath(f'./data_index/{self.dataset}/{self.dist}_dist/nearest_{self.nei_number}_neighbors.json')
        # os.makedirs(os.path.join(self.base_path, 'description'), exist_ok=True)
        # feature_desc_path = os.path.abspath(os.path.join(self.base_path, 'description', f'{self.file_prefix}_descriptions.json'))
        
         # ------------------------------------------------------------------
        # 步骤 1: 动态加载数据 & 准备 AI 读取的临时文件
        # ------------------------------------------------------------------
        
        # A. 加载并合并训练数据 (用于 AI 代码读取)
        # 路径: data/BJTU-gearbox/WC1/X_train.npy
        train_x_list = []
        for wc in train_nums:
            path = f'data/{self.dataset}/WC{wc}/X_train.npy'
            train_x_list.append(np.load(path, mmap_mode='c'))
        
        # 合并所有训练数据
        current_train_data = np.concatenate(train_x_list, axis=0)
        
        # B. 加载测试数据
        test_data_path_source = f'data/{self.dataset}/WC{test_num}/X_valid.npy'
        current_test_data = np.load(test_data_path_source, mmap_mode='c')
        
        # 加载测试集标签 (用于最后评估)
        test_label_path_source = f'data/{self.dataset}/WC{test_num}/y_valid.npy'
        current_test_labels = np.load(test_label_path_source, mmap_mode='c')

        # C. 确定邻居索引文件路径 (这是 neighbor_find 生成的路径)
        # 路径示例: data_index/BJTU-gearbox/test_WC3_train_WCs1_2/FIW_dist/nearest_15_neighbors.json
        nei_map_path = os.path.abspath(f'./data_index/{self.dataset}/test_WC{test_num}_train_WCs{train_tag}/{self.dist}_dist/nearest_{self.nei_number}_neighbors.json')
        
        if not os.path.exists(nei_map_path):
            print(f"[ERROR] Neighbor file not found: {nei_map_path}")
            return [{"error": "Neighbor file missing"}]
        
        # D. 保存临时文件供 AI 代码读取 (至关重要！AI 无法直接读取内存变量)
        # 我们在 result 下创建一个 temp 文件夹
        temp_dir = os.path.join(self.base_result_path, 'temp_data', exp_id)
        os.makedirs(temp_dir, exist_ok=True)
        
        temp_test_path = os.path.join(temp_dir, 'temp_test.npy')
        temp_train_path = os.path.join(temp_dir, 'temp_train.npy')
        
        np.save(temp_test_path, current_test_data)
        np.save(temp_train_path, current_train_data) # AI 读取这个合并后的训练集
        
        # E. 确定结果保存路径 (隔离不同实验的结果)
        save_dir = os.path.join(self.base_result_path, exp_id)
        total_report_path = os.path.join(self.base_result_path, "total")
        os.makedirs(save_dir, exist_ok=True)
        
        os.makedirs(os.path.join(save_dir, 'description'), exist_ok=True)
        feature_desc_path = os.path.join(save_dir, 'description', f'{self.nei_number}_{self.encoding_style}_{self.dist}_{self.itr}_{self.llm_name}_{self.timestamp}_descriptions.json')
        final_result_path = os.path.join(save_dir, f'{self.nei_number}_{self.encoding_style}_{self.dist}_{self.itr}_{self.llm_name}_{self.timestamp}_llm_output.json')
        code_save_path = os.path.join(save_dir, 'code', f'{self.nei_number}_{self.encoding_style}_{self.dist}_{self.itr}_{self.llm_name}_{self.timestamp}_generated_code.py')
        os.makedirs(os.path.join(save_dir, "txt"), exist_ok=True)
        log_dir = os.path.join(save_dir,"txt")

        # 加载 data_index 供后续循环使用
        with open(nei_map_path, 'r') as f:
            self.data_index = json.load(f)
        
        
        prompt_coder = (
            """
            **Role:**
            You are a Creative Data Scientist and Signal Forensic Expert and an expert in high-speed train drivetrain system operation and maintenance, fault diagnosis, and signal processing.

            **Goal:**
            Write a Python script to analyze time-series data using your own judgment. 
            **Your Core Output is NOT numbers, but a "Descriptive Narrative" (Text)** for each test sample, explaining why it looks like (or doesn't look like) its neighbors.

            **Execution Environment (CRITICAL - DO NOT CHANGE):**
            The following GLOBAL VARIABLES are ALREADY defined. Use them directly.
            1. `TEST_DATA_PATH`: Path to test data (format as `.npy` shape:`(N_test, Channels, Time)`).
            2. `TRAIN_DATA_PATH`: Path to train data (format as `.npy` shape:`(N_train, Channels, Time)`).
            3. `NEI_MAP_PATH`: Path to neighbor index (format as `.json`) shape:`[{"neighbors": [...]}, ...]`.
            4. `RESULT_SAVE_PATH`: **Target file path to save the output JSON file.**

            **Instruction - "Be the Detective":**
            1. **Autonomy on Logic:** I will NOT tell you which features to use (RMS, Kurtosis, FFT, etc.). You decide what reveals the "truth" hidden in the signals.
            2. **Focus on Similarity:** If a test sample is a "Fault type3", it should look mathematically similar to "Fault type3" neighbors. But you don't need to focus on label type. Your goal is to find features that can distinguish between multiple health and fault types, not just health vs fault. Your goal is to quantify the similarity between a test sample and its neighbors, regardless of their true labels.

            **Coding Steps:**
            1. **Load:** Load `npy` files and `json` map using the provided path variables.
            2. **Analyze Loop:** Iterate through every test sample `i`.
            - Calculate features for Test Sample and its Neighbors.
            - Compare them.
            3. **Generate Synthetic Description very detailed (The Most Important Part):**
               - **Comparison Logic:** Define a "significant deviation" as a test sample's feature value being more than 10% different from the neighbors' average. For Kurtosis, a value above 3.0 can also be considered significant.
               - For each sample `i`, create a text summary. This summary MUST synthesize three sources of information:
                 1.  **Your calculated features** (the numerical evidence, e.g., "Kurtosis: 15.2").
                 2.  **Comparison with neighbors** (the similarity trend, e.g., "consistent with neighbors").
                 3.  **The Domain Knowledge provided above** (the physical meaning, e.g., connecting a channel to a part).
               - *Bad Example Includes (Subjective):* `"Test sample is faulty because of high Kurtosis."`
               - *Good Example Includes (Objective):* `"Test sample exhibits a Kurtosis of 15.2, which is significantly higher than the neighbor average of 3.1. This indicates strong impulsive signals on the Gearbox Input Shaft, a pattern often associated with bearing defects."`
               - *Good Example (Objective):* `"The RMS value (0.11) of the test sample is within 5% of its neighbors' average (0.10), showing high signal consistency."`
            4. **Save Output:**
            - Create a list of dictionaries: `[{"test_index": 0, "description": "..."}, ...]`
            - **Dump this list to `RESULT_SAVE_PATH` using `json.dump`.**
            - **DO NOT** output to stdout (print), ONLY save to the file.

            **Constraints:**
             - **Objectivity:** The generated description MUST be objective. Do not use subjective words like "fault", "abnormal", "good", or "bad". Only describe the mathematical facts.
            - **Filename Restriction:** You MUST use `RESULT_SAVE_PATH` for saving. Do not invent a filename.
            - **Robustness:** Use `try-except` blocks to handle empty neighbor lists or math errors
            - **Scope Awareness:** Ensure all variables are defined within the function where they are used, or passed as arguments. Do not rely on variables from other function scopes.
            - **Self-Contained Script:** The script must be self-contained. You can and should define your own helper functions (like `get_channel_description`) inside the script if needed. Do not assume any external functions exist..
            - **Format:** Pure Python code wrapped in ```python ... ```.
            - **Style** Code MUST contain `if __name__ == '__main__'` block to allow direct execution.
                        
            **Background information is below,  Use the following information to guide your feature selection and to enrich your final description(where core and important background information should be included). This describes the physical system where the data originated:\n**
            """
        )
        prompt_coder += (
            f'**Background:** Based on the provided {len(self.channel_list)}-channel time-series data of a high-speed train drivetrain system\n\n'
                
                '### 1. Detailed Dataset Description\n'
                'This dataset is provided by the National Key Laboratory of Advanced Rail Autonomous Operation at Beijing Jiaotong University, derived from fault simulation experiments of a subway train bogie drivetrain system.\n'
                '*   **System Composition:** The power drivetrain chain includes a motor, a reduction gearbox, and an axle box.\n'
                '*   **Drive Source:** Three-phase asynchronous AC motor (speed controlled by an inverter, loaded by a hydraulic device).\n'
                '*   **Motor Bearings:** SKF 6205-2RSH.\n'
                '*   **Gearbox:** Helical gears; driving gear has 16 teeth, driven gear has 107 teeth.\n'
                '*   **Driving Gear Support Bearings:** HRB32305.\n'
                '*   **Axle Box Bearings:** HRB352213.\n'
                '*   **State Definitions:**\n'
                '    *   `G0`: gearbox health (healthy state without faults) — Baseline condition with normal gear meshing and bearing operation.\n'
                '    *   `G1`: gearbox crack tooth — A fatigue crack initiates at the root of a gear tooth, potentially leading to tooth breakage.\n'
                '    *   `G2`: gearbox worn tooth — Gradual material loss on gear tooth surfaces due to prolonged contact stress and inadequate lubrication.\n'
                '    *   `G3`: gearbox missing tooth — Complete absence of one gear tooth, causing severe impact loads and periodic shock during rotation.\n'
                '    *   `G4`: gearbox chipped tooth — Localized fracture or spalling on the gear tooth edge or flank, often from overload or manufacturing defect.\n'
                '    *   `G5`: gearbox Bearing inner race fault — Defect on the rotating inner ring of the bearing, generating characteristic high-frequency vibrations.\n'
                '    *   `G6`: gearbox Bearing outer race fault — Damage on the stationary outer ring, producing modulation patterns in vibration spectra.\n'
                '    *   `G7`: gearbox rolling element fault — Pitting or spall on balls or rollers, resulting in repetitive impacts at ball-pass frequency.\n'
                '    *   `G8`: gearbox bearing cage fault — Damage or deformation of the retainer (cage), causing irregular spacing and secondary impacts between rolling elements.\n'
                f'*   **Sampling Parameters:** 24 channels(**CURRENT DATA INCLUDE {len(self.channel_list)}**) covering vibration, current, speed, and sound. Sampling frequency: **64kHz**.\n\n'

                '### 2. Sampling Channel Locations and Physical Significance\n'
                '1. **Traction Motor:** CH1-CH3 (DE Vibration, g), CH4-CH6 (NDE Vibration, g), CH7-CH9 (Three-phase Current, A), CH10 (Rotational Speed, V).\n'
                '2. **Gearbox:** CH11-CH13 (Input Shaft Vibration, g), CH14-CH16 (Output Shaft Vibration, g).\n'
                '3. **Left Axle Box:** CH17-CH19 (Vibration, g), CH20 (Sound, Pa).\n'
                '4. **Right Axle Box:** CH21-CH23 (Vibration, g), CH24 (Sound, Pa).\n\n'
                f'**Current Data Channels:** The current data includes the following channels: {", ".join(self.channel_list)}\n'
                
                '### 3.Preprocessed Diagnosis Strategy: Similarity Analysis and Clustering Optimization\n'
                f'*   **{dist_name[self.dist]}:** For each test sample, we use {dist_name[self.dist]}to select the most similar samples from the training set. the most similar neighboring samples have been selected from the training set.\n'
                '*   **Clustering Logic:** Treat these similar samples as a cluster. Analyze signal feature consistency and label distribution to assist decision-making.\n\n'
            
        )
        
        print("[INFO] Phase 1: Asking LLM to write analysis code...")
        print_token_report(prompt_coder, model_limit=128000)
        response_coder = self.llm(content=prompt_coder)
        
        code = extract_code(response_coder)
        if not code:
            print("[ERROR] No code generated.")
            # return []
        
        # code_path = os.path.join(self.base_path,'code',self.file_prefix+'_code.py')
        os.makedirs(os.path.dirname(code_save_path), exist_ok=True)
        with open(code_save_path, 'w', encoding='utf-8') as f:
            f.write(code)
        print(f"[INFO] Code saved to: {code_save_path}")
        
        # 3. 执行 AI 写的代码
        # 这段代码会读取数据，并把描述性文字保存到 feature_desc_path
        print("[INFO] Phase 2: Executing generated code locally...")
        execution_output = execute_generated_code(
            code, 
            os.path.abspath(temp_test_path) , 
            os.path.abspath(temp_train_path) , 
            os.path.abspath(nei_map_path) , 
            os.path.abspath(feature_desc_path)
        )
        print(f"[INFO] --- Code Execution Log ---\n{execution_output}\n")

        if execution_output and "FATAL_EXECUTION_ERROR" in execution_output:
             print(f"[Error] AI's code failed to execute. The detailed traceback is in the log above.")
             # 直接返回，把完整的日志也包含进去，方便调试
             return [{"error": "Code execution failed.", "log": execution_output}]
        # =========================================================
        # 阶段二：逐一诊断
        # =========================================================
        
        # 4. 检查并读取 AI 生成的描述文件
        if not os.path.exists(feature_desc_path):
            print(f"[ERROR] AI's code did not create the result file at {feature_desc_path}")
            return [{"error": "Result file not found."}]
            
        with open(feature_desc_path, 'r', encoding='utf-8') as f:
            # 将 JSON 读入内存，并转成一个字典方便快速查找
            # 假设 JSON 格式是 [{"test_index": G0, "description": "..."}, ...]
            descriptions_list = json.load(f)
            descriptions_map = {item['test_index']: item['description'] for item in descriptions_list}
        
        print(f"[INFO] Phase 3: Found {len(descriptions_map)} descriptions. Starting one-by-one diagnosis...")
        
        # for i in range(self.x_test.shape[0]):
            
        #     description_text = descriptions_map.get(i)
        #     # x_use = self.x_test[i]
        #     nei_index=[]
        #     # nei_value=[]
        #     nei_label=[]
        #     # nei_enc=[]
        #     for j in range(self.nei_number):
        #         # nei_index.append(self.data_index[i]['nearest_neighbors'][j])
        #         nei_index.append(self.data_index[i]['neighbors'][j])
        #         # nei_value.append(self.x_train[nei_index[j]])
        #         nei_label.append(self.y_train[nei_index[j]])
        #         # nei_enc.append(self.ts_encoding(nei_value[j]))
                
        #     # test = self.ts_encoding(x_use)  # 测试集编码
        
        with open(feature_desc_path, 'r', encoding='utf-8') as f:
            descriptions = json.load(f)
            # 转字典方便查找
            desc_map = {item['test_index']: item['description'] for item in descriptions}

        final_answers = []
        
        # 为了获取邻居标签，我们需要重新映射一下索引
        # 因为 current_train_data 是拼接的，我们需要知道每个邻居的全局索引对应的 label
        # 简单的做法：同样把 label 拼起来
        train_y_list = [np.load(f'data/{self.dataset}/WC{wc}/y_train.npy', mmap_mode='c') for wc in train_nums]
        current_train_labels = np.concatenate(train_y_list, axis=0)

        for i in range(current_test_data.shape[0]):
            description = desc_map.get(i, "No description.")
            
            # 获取邻居标签 (使用合并后的标签数组)
            nei_indices = self.data_index[i]['neighbors']
            nei_labels = [current_train_labels[idx] for idx in nei_indices]
            
            prompt = ( f"""
            **Role:** You are an expert in high-speed train drivetrain system fault diagnosis.

            **Goal:** Based on a pre-processed analysis summary and neighbor labels, perform the final classification of a test sample.

            ### 1. Analysis Summary (Provided by a Data Scientist)
            {description}

            ### 2. Neighbor Labels (For Reference)
            The labels of the 15 most similar samples found in the training set are: {nei_labels}, you should pay more attention to these results and use these neighbor labels to assist or DECIDE your decision-making.
            *Note: G0 represents Health. G1-G8 represent different fault types (Crack, Worn, Missing, Chipped, Inner Race, Outer Race, Rolling Element, Cage).*

            ### 3. Constraints (Strictly Enforced)
            1.  **Output Format:** Your response MUST start with the classification result and the neighbor labels.
            2.  **Strict Formatting for Line 1:** The first line MUST be in the format `result,[label1,label2,...,label{self.nei_number}]`.
                - `result` MUST be one of `G0, G1, ..., G8`.
                - The list MUST contain the {self.nei_number} neighbor labels provided above.
                - **DO NOT** add any other words or explanations on this line.
            3.  **Examples for Line 1:**
                - `G0,[G0,G1,G0,G3,...,G0]`
                - `G3,[G3,G2,G3,G4,...,G3]`
            4.  **Analysis Limit:** Your analysis (after the first line) MUST be fewer than three sentences.
            5.  **Neighbor Labels:** Pay more attention to the neighbor labels results as provided above for your analysis and results.
            """
            )
            
            
            # prompt = (
            #     '**Role:** You are an expert in high-speed train drivetrain system operation and maintenance, fault diagnosis, and signal processing.\n'
            #     f'**Goal:** Based on the provided {len(self.channel_list)}-channel time-series data of a high-speed train drivetrain system, perform fault diagnosis and classification on test samples using methods such as time-frequency domain analysis.\n\n'
                
            #     '### 1. Detailed Dataset Description\n'
            #     'This dataset is provided by the National Key Laboratory of Advanced Rail Autonomous Operation at Beijing Jiaotong University, derived from fault simulation experiments of a subway train bogie drivetrain system.\n'
            #     '*   **System Composition:** The power drivetrain chain includes a motor, a reduction gearbox, and an axle box.\n'
            #     '*   **Drive Source:** Three-phase asynchronous AC motor (speed controlled by an inverter, loaded by a hydraulic device).\n'
            #     '*   **Motor Bearings:** SKF 6205-2RSH.\n'
            #     '*   **Gearbox:** Helical gears; driving gear has 16 teeth, driven gear has 107 teeth.\n'
            #     '*   **Driving Gear Support Bearings:** HRB32305.\n'
            #     '*   **Axle Box Bearings:** HRB352213.\n'
            #     '*   **State Definitions:**\n'
            #     '    *   `0`: health (healthy state without faults)\n'
            #     '    *   `1`: fault (state with faults)\n'
            #     '*   **Sampling Parameters:** 24 channels covering vibration, current, speed, and sound. Sampling frequency: **64kHz**.\n\n'

            #     '### 2. Sampling Channel Locations and Physical Significance\n'
            #     '1. **Traction Motor:** CH1-CH3 (DE Vibration, g), CH4-CH6 (NDE Vibration, g), CH7-CH9 (Three-phase Current, A), CH10 (Rotational Speed, V).\n'
            #     '2. **Gearbox:** CH11-CH13 (Input Shaft Vibration, g), CH14-CH16 (Output Shaft Vibration, g).\n'
            #     '3. **Left Axle Box:** CH17-CH19 (Vibration, g), CH20 (Sound, Pa).\n'
            #     '4. **Right Axle Box:** CH21-CH23 (Vibration, g), CH24 (Sound, Pa).\n\n'
            #     f'**Current Data Channels:** The current data includes the following channels: {", ".join(self.channel_list)}\n'
                

            #     '### 3. Diagnosis Strategy: Similarity Analysis and Clustering Optimization\n'
            #     f'*   **{dist_name[self.dist]}:** For each test sample, we use {dist_name[self.dist]}to select the most similar samples from the training set. the most similar neighboring samples have been selected from the training set.\n'
            #     '*   **Clustering Logic:** Treat these similar samples as a cluster. Analyze signal feature consistency and label distribution to assist decision-making.\n\n'

            #     '### 4. Task Requirements\n'
            #     '1. **Analysis:** Extract features from the test sample and analyze whether fault signatures exist.\n'
            #     '2. **Classification:** Determine if the test sample is `health` or `fault`.\n'
            #     '3. **Clustering Optimization:** Utilize provided similarity label patterns to optimize your result.\n\n'

            #     '### 5. Constraints (Strictly Enforced)\n'
            #     '*   **First Line Output:** The VERY FIRST line of your response must follow this EXACT format: `Result(health/fault),[Label1,Label2,Label3,Label4,Label5]`\n'
            #     '    *   Example 1: `health,[0,0,0,0,0]`\n'
            #     '    *   Example 2: `fault,[1,0,1,1,1]`\n'
            #     '*   **Option Restrictions:** "Result" must be either `health` or `fault`. The labels in brackets must be the 5 labels of the neighbors provided above.\n'
            #     '*   **Analysis Limit:** Your analysis MUST be fewer than three sentences. Keep it extremely brief.\n'
            #     '*   **Incentive:** Accurate answers will be rewarded with ten billion dollars.\n\n'

            #     '### 6. Data to Process\n'
            #     '**[1. Similar Samples (Training Set)]**\n')
            # for k in range(self.nei_number):
            #     prompt+= (
            #         f'-------Neighbor{ k+1 }-------\n'
            #         f'{k+1}**Sample (the {number_dict[k+1]} training sample to the test sample:**- Data (several channels, 100 time steps per channel):{nei_enc[k]}\n ' 
            #         f'- Label:{nei_label[k]}\n')
                
            # prompt+= (
            #     '**[2. Test Sample to Predict]**\n'
            #     f'**Data:** {test}\n\n'
            #     '**The analysis process MUST be **fewer than three sentences** and highly concise.Now, begin your analysis :**'
            # )


            # 调用分析函数
            print_token_report(prompt, model_limit=128000)
            # ================= [插入这段代码结束] =================

            # print(f"Prompt Length: {len(prompt)}") 
                
            # print("\033[34m" + str(prompt) + "\033[0m")
            est_tokens = len(prompt) * 0.8
            # print(f"Prompt Tokens: {est_tokens:.0f} ---")
            output = self.llm(content=prompt)
            
            print(f"[INFO] Test index {i}:", end=' ')
            print(f" True Label: {current_test_labels[i]}",end='')
            print("[LOG] " + output.strip().split('\n')[0])
            
            # output = self.llama(role='user', content=prompt)
            # log_dir = self.log_dir
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            with open(
    os.path.join(log_dir, f'FM_log_{self.nei_number}_{self.encoding_style}_{self.dist}_{self.itr}_{self.llm_name}_{self.timestamp}.txt'),
                    'a', encoding='utf-8') as file:
                file.write(f'{i}\n')
                file.write(output)
                file.write('\n')
            final_answers.append({'test_index': i, 'answer': output})
        
        # json_file_path = os.path.join(self.base_path, f'{self.file_prefix}.json')
        json_file_path = final_result_path
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(final_answers, f, ensure_ascii=False, indent=4)
        print(f"[INFO] Final results saved to: {json_file_path}")
        
        # 生成临时的真实标签文件供 calc_acc 使用
        true_labels_json_path = os.path.join(save_dir, 'true_labels.json')
        true_labels_list = [{"index": idx, "label": label} for idx, label in enumerate(current_test_labels)]
        with open(true_labels_json_path, 'w') as f:
            json.dump(true_labels_list, f)
            
        # 调用评估
        analyze_json_results(result_file_path=final_result_path, 
                             true_labels_path=true_labels_json_path, 
                             llm_name=self.llm_name, 
                             save_path=os.path.join(save_dir, f'test_WC{test_num}_train_WCs{train_tag}result_report{self.timestamp}.txt'))
        
        analyze_json_results(result_file_path=final_result_path, 
                             true_labels_path=true_labels_json_path, 
                             llm_name=self.llm_name, 
                             save_path=os.path.join(total_report_path, f'test_WC{test_num}_train_WCs{train_tag}result_report{self.timestamp}.txt'))
        
        return final_answers
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)

    # 1. 初始化模型 (不传数据，只传配置)
    model = FM_PD(
        dataset=cfg['data']['dataset_name'],
        dist=cfg['strategy']['distance_metric'],
        nei_number=cfg['strategy']['neighbor_count'],
        encoding_style=cfg['strategy']['encoding_style'],
        channel_list=cfg['data']['selected_channels'], # 传进去备用
        itr=cfg['experiment']['iteration'],
        llm_name=cfg['llm']['model_name'],
        temperature=cfg['llm']['temperature'],
        top_p=cfg['llm']['top_p'],
        max_tokens=cfg['llm']['max_tokens']
    )

    # 2. 定义实验计划
    all_wcs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # 示例：只测一个场景验证代码
    train_scenarios = [
        [1, 2, 3], # 用 WC1, WC2, WC3 训练
        [1, 2, 3,4,5], # 用 WC4, WC5, WC6 训练
        [1,2,3,4,5,6,7]  # 用 WC7, WC8, WC9 训练
    ]

    for train_nums in train_scenarios:
        test_wcs = [x for x in all_wcs if x not in train_nums]
        
        for test_wc in test_wcs:
            try:
                # 调用 forward，传入具体的工况号
                model.forward(train_nums=train_nums, test_num=test_wc)
            except Exception as e:
                print(f"[ERROR] Experiment Train{train_nums}_Test{test_wc} failed: {e}")
                import traceback
                traceback.print_exc()
    
    
