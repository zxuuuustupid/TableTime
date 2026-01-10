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

# --- 2. 修改后的检索函数（增加了 train_labels）---
def find_nearest_neighbors_weighted_feature(train_data, train_labels, test_data, num_neighbors):
    """
    使用 [高级特征] + [有监督自适应加权] 进行近邻搜索。
    利用标签信息计算类内方差，给稳定的特征更高的权重。
    """
    
    # --- 步骤 1: 批量提取特征 ---
    print("Extracting advanced features...")
    train_features = np.array([extract_advanced_features(seq) for seq in tqdm(train_data, desc="Train Feat")])
    test_features = np.array([extract_advanced_features(seq) for seq in tqdm(test_data, desc="Test Feat")])

    # --- 步骤 2: 特征标准化 ---
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)
    test_features_scaled = scaler.transform(test_features)
    
    # --- 步骤 3: 计算特征权重 (这是加了标签后的核心提升) ---
    print("Calculating supervised feature weights...")
    unique_classes = np.unique(train_labels)
    n_features = train_features_scaled.shape[1]
    
    # 初始化权重累加器
    feature_weights = np.zeros(n_features)
    
    # 对每个类别，计算特征的稳定性（方差的倒数）
    for label in unique_classes:
        # 找到属于该类的样本
        class_mask = (train_labels == label)
        class_data = train_features_scaled[class_mask]
        
        if len(class_data) > 1:
            # 计算类内方差
            class_var = np.var(class_data, axis=0)
            # 方差越小，特征越重要。加 1e-5 防止除以0
            weight = 1.0 / (class_var + 1e-5)
            feature_weights += weight
    
    # 取平均并归一化权重到 [0, 1]
    feature_weights = feature_weights / len(unique_classes)
    feature_weights = feature_weights / (np.max(feature_weights) + 1e-10)
    
    # 应用权重：重要的特征被放大，噪声特征被缩小
    print("Applying feature weights...")
    train_weighted = train_features_scaled * feature_weights
    test_weighted = test_features_scaled * feature_weights
    
    # --- 步骤 4: 最近邻搜索 ---
    print(f"Searching for {num_neighbors} nearest neighbors in weighted space...")
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='auto', metric='euclidean', n_jobs=-1)
    nbrs.fit(train_weighted)
    
    distances, indices = nbrs.kneighbors(test_weighted)

    # --- 步骤 5: 格式化输出 ---
    results = []
    for test_index in range(len(test_data)):
        results.append({
            "test_index": test_index, 
            "neighbors": indices[test_index].tolist()
        })

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
    train_x_list = [np.load(f'data/{dataset}/WC{wc}/X_train.npy', mmap_mode='c') for wc in train_work_condition_nums]
    train_y_list = [np.load(f'data/{dataset}/WC{wc}/y_train.npy', mmap_mode='c') for wc in train_work_condition_nums]
    
    full_train_data = np.concatenate(train_x_list, axis=0)
    full_train_labels = np.concatenate(train_y_list, axis=0)
    
    # 测试集加载保持不变（假设测试集依然是单个工况）
    
    full_test_data = np.load(f'data/{dataset}/WC{test_work_condition_num}/X_valid.npy', mmap_mode='c')
    full_test_labels = np.load(f'data/{dataset}/WC{test_work_condition_num}/y_valid.npy', mmap_mode='c')
    
    print(f"Original train size: {len(full_train_data)}")
    print(f"Original test size: {len(full_test_data)}")
    
    train_data = full_train_data
    test_data = full_test_data

    train_tag = "_".join(map(str, train_work_condition_nums))

    # --- 后续逻辑不变 ---
    for name, func in dist_map.items():
        output_dir = f'data_index/{dataset}/test_WC{test_work_condition_num}_train_WCs{train_tag}/{name}_dist'
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