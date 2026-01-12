import json
import sys
import os

# 获取当前脚本所在目录的上一层目录（即项目根目录）
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import os
import numpy as np
import json
from dtaidistance import dtw_ndim
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.stats import skew, kurtosis
from scipy.fft import fft, fftfreq
import numpy as np
import os
from tqdm import tqdm
from scipy.spatial.distance import cdist
from scipy.stats import skew, kurtosis
from scipy.fft import fft
from sklearn.decomposition import PCA

# def standardize(X):
#     means = np.mean(X, axis=1, keepdims=True)
#     stds = np.std(X, axis=1, keepdims=True)
#     Z = (X - means) / stds
#     return Z

# def standard_ED(X,Y):
#     X_standard=standardize(X)
#     Y_standard=standardize(Y)
#     return np.linalg.norm(Y_standard-X_standard)

# # def find_nearest_neighbors_DTW(train_data, test_data, num_neighbors=2):
# #     results = []
# #     for test_index, test_seq in tqdm(enumerate(test_data)):
# #         distances = [dtw_ndim.distance(test_seq, train_seq) for train_seq in train_data]
# #         nearest_indices = np.argsort(distances)[:num_neighbors]
# #         results.append({"test_index": test_index, "neighbors": nearest_indices.tolist()})
# #     return results

# def find_nearest_neighbors_DTW(train_data, test_data, num_neighbors=2):
#     results = []
#     for test_index, test_seq in tqdm(enumerate(test_data), desc="DTW Normalized"):
        
#         # [关键] 先对测试序列进行标准化
#         test_seq_std = standardize(test_seq)
        
#         # 在计算距离时，对每一个训练序列也进行标准化
#         distances = [dtw_ndim.distance(test_seq_std, standardize(train_seq)) for train_seq in train_data]
        
#         nearest_indices = np.argsort(distances)[:num_neighbors]
#         results.append({"test_index": test_index, "neighbors": nearest_indices.tolist()})
#     return results


# def find_nearest_neighbors_ED(train_data,test_data,num_neighbors):
#     results=[]
#     for test_index,test_seq in tqdm(enumerate(test_data)):
#         distances = [np.linalg.norm(test_seq-train_seq) for train_seq in train_data]
#         nearest_indices = np.argsort(distances)[:num_neighbors]
#         results.append({"test_index": test_index, "neighbors": nearest_indices.tolist()})
#     return results

# def find_nearest_neighbors_standard_ED(train_data,test_data,num_neighbors):
#     results=[]
#     for test_index,test_seq in tqdm(enumerate(test_data)):
#         distances = [standard_ED(test_seq,train_seq) for train_seq in train_data]
#         nearest_indices = np.argsort(distances)[:num_neighbors]
#         results.append({"test_index": test_index, "neighbors": nearest_indices.tolist()})
#     return results

# def find_nearest_neighbors_MAN(train_data,test_data,num_neighbors):
#     result=[]
#     for test_index,test_seq in tqdm(enumerate(test_data)):
#         distances = [np.sum(np.abs(test_seq-train_seq)) for train_seq in train_data]
#         nearest_indices = np.argsort(distances)[:num_neighbors]
#         result.append({"test_index": test_index, "neighbors": nearest_indices.tolist()})
#     return result

# def calculate_feature_vector(sample_data, fs=64000):
#     """
#     为单个样本 (Channels x TimePoints) 计算一个扁平化的特征向量。
#     """
#     num_channels, num_points = sample_data.shape
#     all_channel_features = []

#     for i in range(num_channels):
#         signal = sample_data[i]
        
#         # 时域特征
#         rms = np.sqrt(np.mean(signal**2))
#         peak = np.max(np.abs(signal))
#         crest_factor = peak / rms if rms > 0 else 0
#         kur = kurtosis(signal, fisher=False)
#         skw = skew(signal)
        
#         # 频域特征
#         fft_vals = np.abs(fft(signal))[:num_points//2]
#         freqs = fftfreq(num_points, 1/fs)[:num_points//2]
#         dominant_freq = freqs[np.argmax(fft_vals)] if len(fft_vals) > 0 else 0
#         spectral_centroid = np.sum(freqs * fft_vals) / np.sum(fft_vals) if np.sum(fft_vals) > 0 else 0

#         channel_features = [rms, peak, crest_factor, kur, skw, dominant_freq, spectral_centroid]
#         all_channel_features.extend(channel_features)
        
#     return np.array(all_channel_features)

# def calculate_robust_feature_vector(sample_data, fs=64000):
#     """
#     改进版特征提取：专注于无量纲指标，减少受转速影响的能量指标。
#     """
#     num_channels, num_points = sample_data.shape
#     all_channel_features = []

#     for i in range(num_channels):
#         signal = sample_data[i]
        
#         # 1. 基础统计量
#         rms = np.sqrt(np.mean(signal**2)) + 1e-9
#         peak = np.max(np.abs(signal))
#         abs_mean = np.mean(np.abs(signal)) + 1e-9
        
#         # 2. 无量纲指标 (这些指标对转速不敏感，只对信号形状敏感)
#         kur = kurtosis(signal, fisher=False)  # 峭度：反映冲击性
#         skw = skew(signal)                     # 偏度：反映分布对称性
#         crest = peak / rms                     # 峰值因子
#         shape = rms / abs_mean                 # 波形因子
#         impulse = peak / abs_mean              # 脉冲因子
        
#         # 3. 频域归一化特征
#         fft_vals = np.abs(fft(signal))[:num_points//2]
#         # 使用能量归一化频谱，关注频谱形状而非绝对强度
#         norm_fft = fft_vals / (np.sum(fft_vals) + 1e-9)
        
#         # 提取频域前3个主峰的相对能量分布（代替绝对频率位置）
#         top_peaks = np.sort(norm_fft)[-3:]
        
#         channel_features = [kur, skw, crest, shape, impulse] 
#         channel_features.extend(top_peaks.tolist())
#         all_channel_features.extend(channel_features)
        
#     return np.array(all_channel_features)

from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier

# --- 复用之前的辅助函数 ---
# 1. standardize(X)
# 2. calculate_feature_vector(sample_data)

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.fft import fft, fftfreq
import numpy as np
from tqdm import tqdm

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy import signal
from scipy.fft import fft, fftfreq
import numpy as np
from tqdm import tqdm

# --- 1. 保持高级特征提取函数不变 ---
def extract_advanced_features(time_series, fs=1000):
    """
    高级特征提取：多尺度时频特征融合
    """
    # 确保输入是 (Time, Channels) 格式
    if time_series.shape[0] < time_series.shape[1]: 
        time_series = time_series.T
        
    n_channels = time_series.shape[1]
    all_features = []
    
    for ch in range(n_channels):
        signal_data = time_series[:, ch]
        
        # 时域特征
        time_features = [
            np.mean(signal_data),
            np.std(signal_data),
            np.max(np.abs(signal_data)),
            np.max(signal_data) - np.min(signal_data),
            np.sqrt(np.mean(signal_data**2)),
            np.max(np.abs(signal_data)) / (np.sqrt(np.mean(signal_data**2)) + 1e-10),
            np.sum(signal_data**4) / (np.sum(signal_data**2)**2 + 1e-10),
            np.sum((signal_data - np.mean(signal_data))**3) / (len(signal_data) * np.std(signal_data)**3 + 1e-10)
        ]
        
        # 频域特征
        f, Pxx = signal.welch(signal_data, fs=fs, nperseg=min(1024, len(signal_data)), noverlap=512)
        dominant_freq = f[np.argmax(Pxx)]
        spectral_centroid = np.sum(f * Pxx) / (np.sum(Pxx) + 1e-10)
        
        freq_features = [
            dominant_freq,
            spectral_centroid,
            np.max(Pxx),
            np.mean(Pxx),
            np.std(Pxx)
        ]
        
        all_features.extend(time_features + freq_features)
    
    # 通道相关性
    correlation_features = []
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            corr = np.corrcoef(time_series[:, i], time_series[:, j])[0, 1]
            correlation_features.append(corr if not np.isnan(corr) else 0)
            
    return np.concatenate([np.array(all_features), np.array(correlation_features)])

# # --- 2. 修改后的检索函数（增加了 train_labels）---
# def find_nearest_neighbors_weighted_feature(train_data, train_labels, test_data, num_neighbors):
#     """
#     使用 [高级特征] + [有监督自适应加权] 进行近邻搜索。
#     利用标签信息计算类内方差，给稳定的特征更高的权重。
#     """
    
#     # --- 步骤 1: 批量提取特征 ---
#     print("Extracting advanced features...")
#     train_features = np.array([extract_advanced_features(seq) for seq in tqdm(train_data, desc="Train Feat")])
#     test_features = np.array([extract_advanced_features(seq) for seq in tqdm(test_data, desc="Test Feat")])

#     # --- 步骤 2: 特征标准化 ---
#     scaler = StandardScaler()
#     train_features_scaled = scaler.fit_transform(train_features)
#     test_features_scaled = scaler.transform(test_features)
    
#     # --- 步骤 3: 计算特征权重 (这是加了标签后的核心提升) ---
#     print("Calculating supervised feature weights...")
#     unique_classes = np.unique(train_labels)
#     n_features = train_features_scaled.shape[1]
    
#     # 初始化权重累加器
#     feature_weights = np.zeros(n_features)
    
#     # 对每个类别，计算特征的稳定性（方差的倒数）
#     for label in unique_classes:
#         # 找到属于该类的样本
#         class_mask = (train_labels == label)
#         class_data = train_features_scaled[class_mask]
        
#         if len(class_data) > 1:
#             # 计算类内方差
#             class_var = np.var(class_data, axis=0)
#             # 方差越小，特征越重要。加 1e-5 防止除以0
#             weight = 1.0 / (class_var + 1e-5)
#             feature_weights += weight
    
#     # 取平均并归一化权重到 [0, 1]
#     feature_weights = feature_weights / len(unique_classes)
#     feature_weights = feature_weights / (np.max(feature_weights) + 1e-10)
    
#     # 应用权重：重要的特征被放大，噪声特征被缩小
#     print("Applying feature weights...")
#     train_weighted = train_features_scaled * feature_weights
#     test_weighted = test_features_scaled * feature_weights
    
#     # --- 步骤 4: 最近邻搜索 ---
#     print(f"Searching for {num_neighbors} nearest neighbors in weighted space...")
#     nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='auto', metric='euclidean', n_jobs=-1)
#     nbrs.fit(train_weighted)
    
#     distances, indices = nbrs.kneighbors(test_weighted)

#     # --- 步骤 5: 格式化输出 ---
#     results = []
#     for test_index in range(len(test_data)):
#         results.append({
#             "test_index": test_index, 
#             "neighbors": indices[test_index].tolist()
#         })

#     return results

def find_nearest_neighbors_weighted_feature(train_data, train_labels, test_data, num_neighbors):
    # --- 步骤 1 & 2: 提取特征并标准化 (保持不变) ---
    print("Extracting advanced features...")
    train_features = np.array([extract_advanced_features(seq) for seq in tqdm(train_data, desc="Train Feat")])
    test_features = np.array([extract_advanced_features(seq) for seq in tqdm(test_data, desc="Test Feat")])
    
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    # --- 步骤 3: [修改这里] 删掉所有权重计算，直接搜索 ---
    print(f"Searching for {num_neighbors} nearest neighbors using Euclidean distance...")
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='auto', metric='euclidean', n_jobs=-1)
    # 直接用标准化后的特征 fit，不再乘以权重
    nbrs.fit(train_features_scaled)
    distances, indices = nbrs.kneighbors(test_features_scaled)

    # --- 步骤 5: 格式化输出 (保持不变) ---
    results = [{"test_index": i, "neighbors": indices[i].tolist()} for i in range(len(test_data))]
    return results


def neighbor_find(dataset, 
                  train_work_condition_nums,
                  test_work_condition_num,
                  neighbor_num,
                  dist_map={'FIW': find_nearest_neighbors_weighted_feature}
): 
    """
    查找最近邻，并支持跳过一个或多个特定标签的数据。
    
    Args:
        skip_labels (list, optional): 要跳过的标签列表, e.g., ['G3', 'G5']. 默认为 None.
    """
    
    # --- 加载所有数据，包括标签 ---
    print(f"Loading data for dataset: {dataset}")
    
    # 核心改动：循环读取多个工况并合并
    train_x_list = [np.load(f'few_shot_test/data/{dataset}/WC{wc}/X_train.npy', mmap_mode='c') for wc in train_work_condition_nums]
    train_y_list = [np.load(f'few_shot_test/data/{dataset}/WC{wc}/y_train.npy', mmap_mode='c') for wc in train_work_condition_nums]
    
    full_train_data = np.concatenate(train_x_list, axis=0)
    full_train_labels = np.concatenate(train_y_list, axis=0)
    
    # 测试集加载保持不变（假设测试集依然是单个工况）
    
    full_test_data = np.load(f'few_shot_test/data/{dataset}/WC{test_work_condition_num}/X_valid.npy', mmap_mode='c')
    full_test_labels = np.load(f'few_shot_test/data/{dataset}/WC{test_work_condition_num}/y_valid.npy', mmap_mode='c')
    
    print(f"Original train size: {len(full_train_data)}")
    print(f"Original test size: {len(full_test_data)}")
    
    train_data = full_train_data
    test_data = full_test_data

    train_tag = "_".join(map(str, train_work_condition_nums))

    # --- 后续逻辑不变 ---
    for name, func in dist_map.items():
        output_dir = f'few_shot_test/data_index/{dataset}/test_WC{test_work_condition_num}_train_WCs{train_tag}/{name}_dist'
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nCalculating neighbors using {name}...")
        
        for j in range(neighbor_num, neighbor_num + 1):
            print(f"  - Finding {j} nearest neighbors...")
            # 只有当 func 是这个带权重的函数时，才传 label，或者统一都传
            # 这里假设你的 dist_map 里只有这一个函数，或者其他函数也适配了参数
            result = func(train_data, full_train_labels, test_data, num_neighbors=j)
            
            output_path = f'{output_dir}/nearest_{j}_neighbors.json'
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=4)
            print(f"    -> Saved results to {output_path}")

def generate_json(dataset):
    # dataset='FingerMovements'
    for work_condition in range(1,10):
        dataset_work_condition=f'{dataset}_WC{work_condition}'
        x_train=np.load(f'few_shot_test/data/{dataset}/WC{work_condition}/X_train.npy',mmap_mode='c')
        x_valid=np.load(f'few_shot_test/data/{dataset}/WC{work_condition}/X_valid.npy',mmap_mode='c')
        y_train=np.load(f'few_shot_test/data/{dataset}/WC{work_condition}/y_train.npy',mmap_mode='c')
        y_valid=np.load(f'few_shot_test/data/{dataset}/WC{work_condition}/y_valid.npy',mmap_mode='c')
        train_index=[]
        test_index=[]

        for i in range(x_train.shape[0]):train_index.append({'index':i,'label':y_train[i]})
        for i in range(x_valid.shape[0]):test_index.append({'index':i,'label':y_valid[i]})

        os.makedirs(f'few_shot_test/data/index/{dataset}/WC{work_condition}', exist_ok=True)
        with open(f'few_shot_test/data/index/{dataset}/WC{work_condition}/train_index.json','w') as f:
            json.dump(train_index,f)
        with open(f'few_shot_test/data/index/{dataset}/WC{work_condition}/test_index.json','w') as f:
            json.dump(test_index,f)


def load_labels_as_map(label_file_path):
    """
    加载标签 JSON 文件，并将其转换为一个易于查找的字典（映射）。
    
    Args:
        label_file_path (str): 标签文件的路径。

    Returns:
        dict: 一个从 index (int) 到 label (str) 的映射, e.g., {0: 'G0', 1: 'G0', ...}
              如果文件不存在或格式错误，返回 None。
    """
    try:
        with open(label_file_path, 'r', encoding='utf-8') as f:
            label_list = json.load(f)
        
        # 将 [{"index": 0, "label": "G0"}, ...] 转换为 {0: "G0", ...}
        label_map = {item['index']: item['label'] for item in label_list}
        return label_map
    except FileNotFoundError:
        print(f"[ERROR] 标签文件未找到: {label_file_path}")
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[ERROR] 标签文件格式错误. 需要 'index' 和 'label' 键. 错误: {e}")
    return None


def calculate_retrieval_accuracy(retrieval_results_path, test_labels_path, train_labels_path):
    """
    计算最近邻检索的准确度（纯度）。

    Args:
        retrieval_results_path (str): 检索结果的 JSON 文件路径。
        test_labels_path (str): 测试集真实标签的 JSON 文件路径。
        train_labels_path (str): 训练集真实标签的 JSON 文件路径 (用于查找邻居的标签)。
    """
    
    # 1. 加载所有必要的标签数据
    test_label_map = load_labels_as_map(test_labels_path)
    train_label_map = load_labels_as_map(train_labels_path)
    
    try:
        with open(retrieval_results_path, 'r', encoding='utf-8') as f:
            retrieval_data = json.load(f)
    except FileNotFoundError:
        print(f"[ERROR] 检索结果文件未找到: {retrieval_results_path}")
        return
    except json.JSONDecodeError as e:
        print(f"[ERROR] 检索结果文件格式错误. 错误: {e}")
        return

    if not test_label_map or not train_label_map:
        print("无法继续评估，因为标签文件加载失败。")
        return

    all_purities = []
    
    # 2. 遍历每一个测试样本的检索结果
    for result_item in retrieval_data:
        test_index = result_item.get('test_index')
        neighbor_indices = result_item.get('neighbors', [])
        
        if test_index is None or not neighbor_indices:
            print(f"[WARNING] 跳过 test_index {test_index}，因为数据不完整。")
            continue
            
        # 获取当前测试样本的真实标签
        true_test_label = test_label_map.get(test_index)
        if true_test_label is None:
            print(f"[WARNING] 在标签文件中找不到 test_index {test_index} 的真实标签。")
            continue
            
        # 3. 计算邻居纯度
        correct_neighbors = 0
        for neighbor_idx in neighbor_indices:
            # 从训练集标签映射中查找邻居的标签
            neighbor_label = train_label_map.get(neighbor_idx)
            
            if neighbor_label is not None and neighbor_label == true_test_label:
                correct_neighbors += 1
        
        # 纯度 = (与测试样本同类的邻居数) / (总邻居数)
        purity = correct_neighbors / len(neighbor_indices)
        all_purities.append(purity)
        
    # 4. 计算并打印总体结果
    if not all_purities:
        print("[ERROR] 没有可供评估的有效检索结果。")
        return 0.0
    
    mean_accuracy = np.mean(all_purities) * 100
    
    print("\n" + "="*50)
    print(f"[INFO] 最近邻检索精度评估报告")
    print(f"[INFO] - 检索文件: {os.path.basename(retrieval_results_path)}")
    print("="*50)
    print(f"[INFO] 总计评估的测试样本数: {len(all_purities)}")
    print(f"[INFO] 平均检索精度 (Mean Purity @ k): {mean_accuracy:.2f}%")
    print("="*50)
    print(f"[INFO] (该指标衡量的是：对于一个测试样本，其找到的邻居有多大概率与它自己是同一类别)")
    
    return mean_accuracy


# weight_DTW=0.1
# weight_feature=0.9
# def pipeline():
#     generate_json(dataset=dataset)
#     neighbor_find(dataset=dataset,
#                     train_work_condition_num=train_work_condition_num,
#                     test_work_condition_num=test_work_condition_num,
#                     dist_map = dist_map,
#                     neighbor_num = neighbor_num,
#                     skip_labels = None,)
#     calculate_retrieval_accuracy(retrieval_results_path=os.path.join("data_index", dataset, f"test_WC{test_work_condition_num}_train_WC{train_work_condition_num}",f"{list(dist_map.keys())[0]}_dist", f'nearest_{neighbor_num}_neighbors.json'),test_labels_path=os.path.join("data", "index",dataset,f"WC{test_work_condition_num}","test_index.json"), train_labels_path=os.path.join("data", "index", dataset,f"WC{train_work_condition_num}", "train_index.json"))

def pipeline(dataset, train_nums, test_num, dist_map, neighbor_num):
    # 1. 生成基础索引 (如果需要)
    generate_json(dataset=dataset)
    
    # 2. 寻找近邻 (注意这里传的是列表 train_nums)
    neighbor_find(dataset=dataset,
                  train_work_condition_nums=train_nums,
                  test_work_condition_num=test_num,
                  dist_map=dist_map,
                  neighbor_num=neighbor_num)
    
    # 3. 构造路径标识 (例如 [1,2,3] -> "1_2_3")
    train_tag = "_".join(map(str, train_nums))
    results_path = os.path.join("few_shot_test/data_index", dataset, f"test_WC{test_num}_train_WCs{train_tag}", 
                                f"{list(dist_map.keys())[0]}_dist", f'nearest_{neighbor_num}_neighbors.json')
    
    # 4. 合并训练集标签 (核心改动：因为合并后的训练集索引是连续的，需要手动合并字典)
    merged_train_labels = {}
    current_offset = 0
    for wc in train_nums:
        path = os.path.join("few_shot_test/data", "index", dataset, f"WC{wc}", "train_index.json")
        with open(path, 'r') as f:
            labels = json.load(f)
            for item in labels:
                # 将该工况的标签存入合并字典，键为全局偏移后的索引
                merged_train_labels[current_offset] = item['label']
                current_offset += 1
    
    # 5. 加载测试集标签
    test_labels_path = os.path.join("few_shot_test/data", "index", dataset, f"WC{test_num}", "test_index.json")
    
    # 6. 计算准确率 (这里需要稍微修改 calculate_retrieval_accuracy 使其支持直接传字典，或者如下快捷处理)
    # 为了最小化改动，我们临时写一个合并后的json
    temp_train_labels_path = "temp_merged_train_labels.json"
    with open(temp_train_labels_path, 'w') as f:
        json.dump([{"index": k, "label": v} for k, v in merged_train_labels.items()], f)
        
    acc=calculate_retrieval_accuracy(retrieval_results_path=results_path,
                                 test_labels_path=test_labels_path,
                                 train_labels_path=temp_train_labels_path)

    return acc
    
# if __name__ == "__main__":
        
#     dataset='BJTU-gearbox'
#     dist_map = {'FIW': find_nearest_neighbors_weighted_feature}
#     neighbor_num = 15
#     train_work_condition_num=1
#     test_work_condition_num=2
#     pipeline()

if __name__ == "__main__":
    import datetime # 确保导入 datetime
    
    dataset = 'BJTU-gearbox'
    # dataset = 'BJTU-motor'
    # dataset = 'BJTU-leftaxlebox'
    dist_map_name = 'FIW'
    dist_map = {dist_map_name: find_nearest_neighbors_weighted_feature}
    neighbor_num = 3
    all_wcs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    # 定义训练场景
    # all_train_scenarios = [
    #     [1, 4],
    #     [1, 4, 7],
    #     [1, 2, 3, 4, 6],
    #     [1, 2, 3, 4, 5, 6, 7]
    # ]
    
    # all_train_scenarios = [
    #     [1, 2],
    #     [1, 2, 3],
    #     [1, 2, 3, 4, 5],
    #     [1, 2, 3, 4, 5, 6, 7]
    # ]
    
    all_train_scenarios = [
        [1, 2, 3,]
    ]
    
    # 用于收集所有实验结果的列表
    experiment_logs = []
    
    # --- 开始大循环 ---
    for train_nums in all_train_scenarios:
        test_wcs = [wc for wc in all_wcs if wc not in train_nums]
        
        
        print(f"\n{'='*60}")
        print(f"🚀 大实验启动：训练集组合 = {train_nums}")
        print(f"{'='*60}")
        
        scenario_accuracies = []
        
        for test_wc in test_wcs:
            print(f"\n>>> [当前配置] 训练: {train_nums} | 测试: WC{test_wc}")
            # 获取准确率
            acc = pipeline(dataset, train_nums, test_wc, dist_map, neighbor_num)
            
            # 记录单次结果
            log_str = f"Train: {train_nums} | Test: WC{test_wc} | Accuracy: {acc:.2f}%"
            experiment_logs.append(log_str)
            scenario_accuracies.append(acc)
        
        # 记录该场景的平均准确率
        avg_acc = np.mean(scenario_accuracies) if scenario_accuracies else 0
        experiment_logs.append(f"--- Scenario Average (Train {train_nums}): {avg_acc:.2f}% ---\n")

    # --- 实验结束，保存汇总结果 ---
    
    # 1. 生成文件名
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "few_shot_test/result/log"
    os.makedirs(log_dir, exist_ok=True)
    filename = f"{dataset}_{dist_map_name}_{timestamp}.txt"
    filepath = os.path.join(log_dir, filename)
    
    # 2. 构建完整报告内容
    final_report = []
    final_report.append("="*60)
    final_report.append(f"实验汇总报告")
    final_report.append(f"时间: {timestamp}")
    final_report.append(f"数据集: {dataset}")
    final_report.append(f"距离度量: {dist_map_name}")
    final_report.append(f"邻居数: {neighbor_num}")
    final_report.append("="*60 + "\n")
    final_report.extend(experiment_logs)
    
    final_report_str = "\n".join(final_report)
    
    # 3. 打印并保存
    print("\n" + "#"*60)
    print("实验全部完成！汇总结果如下：")
    print("#"*60)
    print(final_report_str)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(final_report_str)
        
    print(f"\n[INFO] 汇总日志已保存至: {filepath}")