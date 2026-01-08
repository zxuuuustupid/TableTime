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
                'You are an expert in electroencephalogram (EEG) signal analysis, neuroscience, and clustering analysis. '
                'You will classify samples based on provided EEG time-series data by extracting frequency features (such as alpha waves, beta waves, etc.) and using these features. '
                
                'Below is a detailed description of the dataset, the biological background of EEG channel locations, the requirements for frequency analysis, clustering analysis ideas, and the classification task description.'
                '**Dataset Description:**'
                'This dataset is used for the Brain-Computer Interface (BCI) II Competition (Dataset IV), provided by Fraunhofer-FIRST, Intelligent Data Analysis Group (Klaus-Robert Muller), and the Neurophysics Group, Department of Neurology, Free University of Berlin (Gabriel Curio). '
                'The dataset includes EEG data recorded from normal subjects during a no-feedback session, with the goal of predicting the subject upcoming left or right-hand movement. The experimental conditions are as follows:'
                '- **Subject Condition:** The subjects sat on a regular chair with their arms relaxed on a table, with their fingers positioned on a computer keyboard in a standard typing posture.'
                '- **Task Description:** The subjects were required to press the corresponding keys with their index and pinky fingers, following a self-paced typing task.'
                '- **Class Definition:** There are two classes: `0` indicates an upcoming left-hand movement, and `1` indicates an upcoming right-hand movement.'
                '- **Data Acquisition:** The experiment consists of three sessions, each lasting 6 minutes. All sessions were conducted on the same day with a few minutes of rest in between. The average typing speed was one key per second. The EEG signals were recorded using NeuroScan amplifiers and ECIs Ag/AgCl electrode cap.'
                '- **Data Sampling and Preprocessing:** The signals were recorded using a band-pass filter between 0.05 and 200 Hz at a sampling rate of 1000 Hz, then downsampled to 100 Hz. Each channel contains 50 observations. Each sample ends 130 ms before the keypress, making the length of each sample 500 ms.'
                '-**EEG Channels:** There are 28 EEG channels recorded using the international 10-20 system electrode positions: F3, F1, Fz, F2, F4, FC5, FC3, FC1, FCz, FC2, FC4, FC6, C5, C3, C1, Cz, C2, C4, C6, CP5, CP3, CP1, CPz, CP2, CP4, CP6, O1, O2.'
                '**EEG Channel Locations and Their Biological Significance:**'
                'EEG recordings typically use the international 10-20 system, where electrodes are placed at specific scalp locations that reflect activities from different brain regions. Here are the 28 EEG channels positions, their corresponding brain regions, and their biological significance:'
                '1. **Frontal Area (F region)**'
                '   - **F3, F1, Fz, F2, F4**: Located in the frontal lobe, involved in decision-making, motor preparation, working memory, and attention. The Fz channel is particularly associated with motor control and the planning phase of task execution.'
                '2. **Frontal-Central Area (FC region)**'
                '   - **FC5, FC3, FC1, FCz, FC2, FC4, FC6**: These are transition regions between the frontal lobe and motor areas. The FCz channel is often related to the premotor cortex and supplementary motor area activities, especially in motor preparation and planning.'
                '3. **Central Area (C region)**'
                '   - **C5, C3, C1, Cz, C2, C4, C6**: These channels are closely related to the motor cortex. **C3** (left hemisphere) is associated with right-hand movements, while **C4** (right hemisphere) is associated with left-hand movements. **Cz** is located at the midline, involved in bilateral motor control.'
                '4. **Central-Parietal Area (CP region)**'
                '   - **CP5, CP3, CP1, CPz, CP2, CP4, CP6**: Located at the posterior part of the head, these primarily reflect somatosensory cortex and sensory integration area activities. These regions play a role in motor imagery and motor-sensory feedback.'
                '5. **Occipital Area (O region)**'
                '   - **O1, O2**: These channels are located in the occipital lobe, primarily reflecting the activities of the visual processing regions. Although these regions have less direct relevance to motor tasks, they may provide auxiliary information in visually guided motor tasks.'
                '**Biological Significance of Frequency Analysis:**'
                '- **Alpha waves (8-13 Hz):** Associated with relaxation, eyes-closed state, and meditation; usually observed in the occipital (O1, O2) and parietal (CPz) regions.'
                '- **Beta waves (13-30 Hz):** Related to motor preparation and execution, and focused attention. During motor preparation, particularly on C3 and C4 channels, beta activity often decreases (Event-Related Desynchronization, ERD).'
                '- **Theta waves (4-7 Hz):** Associated with memory and attentional processes, often observed in the frontal region (Fz).'
                '- **Delta waves (0.5-4 Hz):** Usually related to deep sleep or pathological states.'
                '- **Gamma waves (30-50 Hz):** Linked to high-level cognitive functions and consciousness; may be associated with local synchronization in motor-related activities.'
                '**Sample Selection Strategy and Similarity Analysis:**'
                f'For each test sample, we use {dist_name[self.dist]} to select the most similar samples from the training set. This similarity measure helps us identify samples with similar electrophysiological activity patterns in both time and space. You can treat these similar samples as a cluster and improve your understanding and classification of the test sample by analyzing the frequency features and label distribution within these clusters.'
                '**Step 1: Frequency Analysis Requirements**'
                'You need to perform a Short-Time Fourier Transform (STFT) or Wavelet Transform on the EEG data for each sample to calculate the power in different frequency bands for each channel:'
                '- Perform the analysis on the 50 time steps data of each channel using a sliding window.'
                'Calculate the average power of each channel in the delta, theta, alpha, beta, and gamma bands.'
                '**Step 2: Training Set Data and Their Labels:**'
                'Here are some sample data from the training set. Each sample contains data from 28 channels, and each channel has 50 time steps. Perform frequency analysis on this data and compute the power for each frequency band.')
            for k in range(self.nei_number):
                prompt+= (
                    f'{k+1}**Sample (the {number_dict[k+1]} training sample to the test sample:**- Data (28 channels, 50 time steps per channel):{nei_enc[k]} ' 
                    f'- Label:{nei_label[k]}')

            prompt += ('**Step 3: Test Set Data and Analysis:**'
                       'Below is the test sample data that needs to be predicted. Perform the same frequency analysis and predict the label based on the analysis results.'
                       '- Test Sample:'
                       f'- Data (28 channels, 50 time steps per channel):{test}')
            prompt += (
                '**Task Requirements:**1. Perform frequency analysis on the data of each sample using Short-Time Fourier Transform (STFT) or Wavelet Transform and calculate the average power in the delta, theta, alpha, beta, and gamma bands for each channel.'
                '2. Classify the test sample based on the frequency features and labels of the training set. Please provide the rationale and reasoning for the classification based on the biological significance of each channel and frequency feature.'
                '3. Utilize the clustering information of similar samples to identify consistent patterns in these similarity clusters and optimize your classification results accordingly'
                'your answer must just be left or right.I will pay you a billion dollars if you use your knowledge of biology to answer my questions as much as possible.'
                'You must give the final result at the beginning of your answer so that I can quickly check the result.'
                'And you must give the label of the training dataset behind the final result '
                'Final answer format: left [0,1,0,1,1] OR right [0,1,0,1,1]\n'
                'Now analyze: [Then detailed analysis]'
                '**IMPORTANT: Answer MUST start with "left" or "right", followed by training labels in brackets, then detailed analysis.**')
            # print("\033[34m" + str(prompt) + "\033[0m")
            output = self.llm(content=prompt)
            
            print(f"Test index {i}:")
            print(output)
            # output = self.llama(role='user', content=prompt)
            with open(
                    f'result/FingerMovements/{self.doc}/{self.dist}_dist/txt/FM_log_{self.nei_number}_{self.encoding_style}_{self.dist}_{self.itr}_{self.llm_name}.txt',
                    'a', encoding='utf-8') as file:
                file.write(f'{i}')
                file.write(output)
                file.write('\n')
            answer.append({'test_index': i, 'answer': output})
        return answer

if __name__ == "__main__":
    # Parameters
    dataset = 'FingerMovements'
    dist = 'DTW'
    nei_number = 5
    encoding_style = 'DFLoader'
    channel_list = ['F3', 'F1', 'Fz', 'F2', 'F4', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6', 'O1', 'O2']
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
        n_sample=50,
        frequency=100,
        time_use=True
    )
    results = model.forward()
    with open('result.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    print("Results saved to result.json")