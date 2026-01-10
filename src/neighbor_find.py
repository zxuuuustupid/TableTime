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

def standardize(X):
    means = np.mean(X, axis=1, keepdims=True)
    stds = np.std(X, axis=1, keepdims=True)
    Z = (X - means) / stds
    return Z

def standard_ED(X,Y):
    X_standard=standardize(X)
    Y_standard=standardize(Y)
    return np.linalg.norm(Y_standard-X_standard)

# def find_nearest_neighbors_DTW(train_data, test_data, num_neighbors=2):
#     results = []
#     for test_index, test_seq in tqdm(enumerate(test_data)):
#         distances = [dtw_ndim.distance(test_seq, train_seq) for train_seq in train_data]
#         nearest_indices = np.argsort(distances)[:num_neighbors]
#         results.append({"test_index": test_index, "neighbors": nearest_indices.tolist()})
#     return results

def find_nearest_neighbors_DTW(train_data, test_data, num_neighbors=2):
    results = []
    for test_index, test_seq in tqdm(enumerate(test_data), desc="DTW Normalized"):
        
        # [关键] 先对测试序列进行标准化
        test_seq_std = standardize(test_seq)
        
        # 在计算距离时，对每一个训练序列也进行标准化
        distances = [dtw_ndim.distance(test_seq_std, standardize(train_seq)) for train_seq in train_data]
        
        nearest_indices = np.argsort(distances)[:num_neighbors]
        results.append({"test_index": test_index, "neighbors": nearest_indices.tolist()})
    return results


def find_nearest_neighbors_ED(train_data,test_data,num_neighbors):
    results=[]
    for test_index,test_seq in tqdm(enumerate(test_data)):
        distances = [np.linalg.norm(test_seq-train_seq) for train_seq in train_data]
        nearest_indices = np.argsort(distances)[:num_neighbors]
        results.append({"test_index": test_index, "neighbors": nearest_indices.tolist()})
    return results

def find_nearest_neighbors_standard_ED(train_data,test_data,num_neighbors):
    results=[]
    for test_index,test_seq in tqdm(enumerate(test_data)):
        distances = [standard_ED(test_seq,train_seq) for train_seq in train_data]
        nearest_indices = np.argsort(distances)[:num_neighbors]
        results.append({"test_index": test_index, "neighbors": nearest_indices.tolist()})
    return results

def find_nearest_neighbors_MAN(train_data,test_data,num_neighbors):
    result=[]
    for test_index,test_seq in tqdm(enumerate(test_data)):
        distances = [np.sum(np.abs(test_seq-train_seq)) for train_seq in train_data]
        nearest_indices = np.argsort(distances)[:num_neighbors]
        result.append({"test_index": test_index, "neighbors": nearest_indices.tolist()})
    return result

def calculate_feature_vector(sample_data, fs=64000):
    """
    为单个样本 (Channels x TimePoints) 计算一个扁平化的特征向量。
    """
    num_channels, num_points = sample_data.shape
    all_channel_features = []

    for i in range(num_channels):
        signal = sample_data[i]
        
        # 时域特征
        rms = np.sqrt(np.mean(signal**2))
        peak = np.max(np.abs(signal))
        crest_factor = peak / rms if rms > 0 else 0
        kur = kurtosis(signal, fisher=False)
        skw = skew(signal)
        
        # 频域特征
        fft_vals = np.abs(fft(signal))[:num_points//2]
        freqs = fftfreq(num_points, 1/fs)[:num_points//2]
        dominant_freq = freqs[np.argmax(fft_vals)] if len(fft_vals) > 0 else 0
        spectral_centroid = np.sum(freqs * fft_vals) / np.sum(fft_vals) if np.sum(fft_vals) > 0 else 0

        channel_features = [rms, peak, crest_factor, kur, skw, dominant_freq, spectral_centroid]
        all_channel_features.extend(channel_features)
        
    return np.array(all_channel_features)

def calculate_robust_feature_vector(sample_data, fs=64000):
    """
    改进版特征提取：专注于无量纲指标，减少受转速影响的能量指标。
    """
    num_channels, num_points = sample_data.shape
    all_channel_features = []

    for i in range(num_channels):
        signal = sample_data[i]
        
        # 1. 基础统计量
        rms = np.sqrt(np.mean(signal**2)) + 1e-9
        peak = np.max(np.abs(signal))
        abs_mean = np.mean(np.abs(signal)) + 1e-9
        
        # 2. 无量纲指标 (这些指标对转速不敏感，只对信号形状敏感)
        kur = kurtosis(signal, fisher=False)  # 峭度：反映冲击性
        skw = skew(signal)                     # 偏度：反映分布对称性
        crest = peak / rms                     # 峰值因子
        shape = rms / abs_mean                 # 波形因子
        impulse = peak / abs_mean              # 脉冲因子
        
        # 3. 频域归一化特征
        fft_vals = np.abs(fft(signal))[:num_points//2]
        # 使用能量归一化频谱，关注频谱形状而非绝对强度
        norm_fft = fft_vals / (np.sum(fft_vals) + 1e-9)
        
        # 提取频域前3个主峰的相对能量分布（代替绝对频率位置）
        top_peaks = np.sort(norm_fft)[-3:]
        
        channel_features = [kur, skw, crest, shape, impulse] 
        channel_features.extend(top_peaks.tolist())
        all_channel_features.extend(channel_features)
        
    return np.array(all_channel_features)

from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestClassifier

# --- 复用之前的辅助函数 ---
# 1. standardize(X)
# 2. calculate_feature_vector(sample_data)

def find_nearest_neighbors_weighted_feature(train_data, test_data, num_neighbors):
    """
    集成子空间对齐（Subspace Alignment）的改进检索算法。
    专为跨工况（不同转速/负载）设计。
    """
    
    # --- 步骤 1: 批量特征提取 ---
    print("Extracting robust features...")
    train_features = np.array([calculate_robust_feature_vector(s) for s in tqdm(train_data, desc="Train Feat")])
    test_features = np.array([calculate_robust_feature_vector(s) for s in tqdm(test_data, desc="Test Feat")])

    # --- 步骤 2: 子空间对齐 (Subspace Alignment) ---
    # 核心逻辑：将测试集的特征空间对齐到训练集，消除工况差异
    print("Performing Subspace Alignment to align working conditions...")
    
    # 自动选择保留的主成分数量（取特征维度的一半或固定值）
    n_components = min(train_features.shape[1] // 2, 20)
    
    pca_train = PCA(n_components=n_components).fit(train_features)
    pca_test = PCA(n_components=n_components).fit(test_features)
    
    # 获取特征空间基向量
    L_train = pca_train.components_.T
    L_test = pca_test.components_.T
    
    # 对齐矩阵 M = L_test' * L_train
    # 此矩阵可以将测试集特征转换到训练集的分布空间
    M = np.dot(L_test.T, L_train)
    
    # 转换后的特征
    # 我们将两个集合都映射到训练集的子空间中进行比较
    train_projected = np.dot(train_features, L_train)
    test_aligned = np.dot(np.dot(test_features, L_test), M)

    # --- 步骤 3: 距离计算 ---
    # 在对齐后的空间使用标准化欧氏距离
    print("Calculating distances in aligned subspace...")
    # 注意：cdist 的 'seuclidean' 需要在投影空间计算
    distances_matrix = cdist(test_aligned, train_projected, metric='seuclidean')

    results = []
    for test_index in range(len(test_data)):
        distances = distances_matrix[test_index]
        nearest_indices = np.argsort(distances)[:num_neighbors]
        results.append({
            "test_index": test_index, 
            "neighbors": nearest_indices.tolist()
        })

    return results

# def find_nearest_neighbors_hybrid(train_data, test_data, num_neighbors, dtw_weight=0.2, feature_weight=0.8):
#     """
#     使用 DTW 距离和特征向量距离的加权和来寻找最近邻。
#     """
    
#     # --- 1. 离线特征提取与标准化 ---
#     print("Pre-calculating feature vectors for all training data...")
#     train_features = np.array([calculate_feature_vector(train_seq) for train_seq in tqdm(train_data, desc="Featuring Train")])
#     train_mean = np.mean(train_features, axis=0)
#     train_std = np.std(train_features, axis=0)
#     train_std[train_std == 0] = 1
#     train_features_std = (train_features - train_mean) / train_std

#     results = []
#     print("\nFinding nearest neighbors using Hybrid distance...")
#     for test_index, test_seq in tqdm(enumerate(test_data), desc="Hybrid Search"):
        
#         # --- 2. 计算两种距离 ---
#         # a. 标准化的 DTW 距离
#         test_seq_std = standardize(test_seq)
#         dtw_distances = np.array([dtw_ndim.distance(test_seq_std, standardize(train_seq)) for train_seq in train_data])

#         # b. 标准化的特征向量距离
#         test_feature = calculate_feature_vector(test_seq)
#         test_feature_std = (test_feature - train_mean) / train_std
#         feature_distances = cdist(test_feature_std.reshape(1, -1), train_features_std, metric='euclidean')[0]
        
#         # --- 3. 归一化与加权 ---
#         # Min-Max Normalization to [0, 1]
#         dtw_norm = (dtw_distances - np.min(dtw_distances)) / (np.max(dtw_distances) - np.min(dtw_distances)) if np.ptp(dtw_distances) > 0 else np.zeros_like(dtw_distances)
#         feat_norm = (feature_distances - np.min(feature_distances)) / (np.max(feature_distances) - np.min(feature_distances)) if np.ptp(feature_distances) > 0 else np.zeros_like(feature_distances)

#         # c. 计算总距离
#         total_distances = (dtw_weight * dtw_norm) + (feature_weight * feat_norm)
        
#         # --- 4. 排序 ---
#         nearest_indices = np.argsort(total_distances)[:num_neighbors]
        
#         results.append({"test_index": test_index, "neighbors": nearest_indices.tolist()})

#     return results


# dataset='RacketSports'
# dataset = 'FingerMovements'
# train_data=np.load(f'data/{dataset}/X_train.npy', mmap_mode='c')
# test_data=np.load(f'data/{dataset}/X_valid.npy', mmap_mode='c')

# dist={'DTW':find_nearest_neighbors_DTW,'ED':find_nearest_neighbors_ED,
#       'SED':find_nearest_neighbors_standard_ED,'MAN':find_nearest_neighbors_MAN}

# for i in ['DTW','ED','SED','MAN']:
#     for j in [1,2,3,4,5,6,7,8,9,10]:
#         result=dist[i](train_data,test_data,num_neighbors=j)
#         with open(f'data_index/{dataset}/{i}_dist/nearest_{j}_neighbors.json', 'w') as f:
#             json.dump(result,f,indent=4)

def neighbor_find(dataset, 
                  train_work_condition_num,
                  test_work_condition_num,
                  neighbor_num,
                  dist_map={'DTW': find_nearest_neighbors_DTW, 
                            'FIW': find_nearest_neighbors_weighted_feature},
                  skip_labels=None,
): 
    """
    查找最近邻，并支持跳过一个或多个特定标签的数据。
    
    Args:
        skip_labels (list, optional): 要跳过的标签列表, e.g., ['G3', 'G5']. 默认为 None.
    """
    
    # --- 加载所有数据，包括标签 ---
    print(f"Loading data for dataset: {dataset}")
    full_train_data = np.load(f'data/{dataset}/WC{train_work_condition_num}/X_train.npy', mmap_mode='c')
    full_train_labels = np.load(f'data/{dataset}/WC{train_work_condition_num}/y_train.npy', mmap_mode='c')
    
    full_test_data = np.load(f'data/{dataset}/WC{test_work_condition_num}/X_valid.npy', mmap_mode='c')
    full_test_labels = np.load(f'data/{dataset}/WC{test_work_condition_num}/y_valid.npy', mmap_mode='c')
    
    print(f"Original train size: {len(full_train_data)}")
    print(f"Original test size: {len(full_test_data)}")
    
    # --- [修改2] 过滤逻辑 ---
    # 检查 skip_labels 是否是一个非空列表
    if skip_labels and isinstance(skip_labels, list):
        print(f"Filtering out labels: {skip_labels}")
        
        # 使用 np.isin() 来高效地创建掩码
        # np.isin(A, B) 会返回一个布尔数组，表示 A 中的元素是否在 B 中
        # 我们用 ~ (取反) 来选择那些 *不在* skip_labels 列表中的元素
        train_mask = ~np.isin(full_train_labels, skip_labels)
        test_mask = ~np.isin(full_test_labels, skip_labels)
        
        # 应用掩码
        train_data = full_train_data[train_mask]
        test_data = full_test_data[test_mask]
        
        print(f"Filtered train size: {len(train_data)}")
        print(f"Filtered test size: {len(test_data)}")
    else:
        # 如果不跳过任何标签，则使用全部数据
        train_data = full_train_data
        test_data = full_test_data

    # --- 后续逻辑不变 ---
    for name, func in dist_map.items():
        output_dir = f'data_index/{dataset}/test_WC{test_work_condition_num}_train_WC{train_work_condition_num}/{name}_dist'
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nCalculating neighbors using {name}...")
        
        for j in range(neighbor_num, neighbor_num + 1):
            print(f"  - Finding {j} nearest neighbors...")
            result = func(train_data, test_data, num_neighbors=j)
            
            output_path = f'{output_dir}/nearest_{j}_neighbors.json'
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=4)
            print(f"    -> Saved results to {output_path}")