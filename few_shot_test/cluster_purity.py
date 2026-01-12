# import numpy as np
# import os
# from tqdm import tqdm
# from sklearn.preprocessing import StandardScaler
# from sklearn.neighbors import NearestNeighbors
# from scipy import signal

# # --- 1. 配置路径与参数 ---
# DATASET = "BJTU-gearbox"
# BASE_PATH = f"few_shot_test/data/{DATASET}"
# NUM_NEIGHBORS = 1

# # --- 2. 核心功能函数 (完全保留你的数学逻辑) ---

# def load_wc_data(wc_id):
#     path = os.path.join(BASE_PATH, f"WC{wc_id}")
#     # 强制转为 float32 防止精度引起的微小差异
#     x_train = np.load(os.path.join(path, 'X_train.npy'), mmap_mode='c').astype(np.float32)
#     y_train = np.load(os.path.join(path, 'y_train.npy'), mmap_mode='c').astype(str)
#     x_valid = np.load(os.path.join(path, 'X_valid.npy'), mmap_mode='c').astype(np.float32)
#     y_valid = np.load(os.path.join(path, 'y_valid.npy'), mmap_mode='c').astype(str)
#     return x_train, y_train, x_valid, y_valid

# def extract_advanced_features(time_series, fs=1000):
#     if time_series.shape[0] < time_series.shape[1]: 
#         time_series = time_series.T
#     n_channels = time_series.shape[1]
#     all_features = []
#     for ch in range(n_channels):
#         signal_data = time_series[:, ch]
#         # 时域特征 8个
#         time_features = [
#             np.mean(signal_data), np.std(signal_data), np.max(np.abs(signal_data)),
#             np.max(signal_data) - np.min(signal_data), np.sqrt(np.mean(signal_data**2)),
#             np.max(np.abs(signal_data)) / (np.sqrt(np.mean(signal_data**2)) + 1e-10),
#             np.sum(signal_data**4) / (np.sum(signal_data**2)**2 + 1e-10),
#             np.sum((signal_data - np.mean(signal_data))**3) / (len(signal_data) * np.std(signal_data)**3 + 1e-10)
#         ]
#         # 频域特征 5个
#         f, Pxx = signal.welch(signal_data, fs=fs, nperseg=min(1024, len(signal_data)), noverlap=512)
#         freq_features = [f[np.argmax(Pxx)], np.sum(f * Pxx) / (np.sum(Pxx) + 1e-10), np.max(Pxx), np.mean(Pxx), np.std(Pxx)]
#         all_features.extend(time_features + freq_features)
#     # 通道相关性
#     correlation_features = [np.corrcoef(time_series[:, i], time_series[:, j])[0, 1] 
#                             for i in range(n_channels) for j in range(i+1, n_channels)]
#     return np.concatenate([np.array(all_features), np.nan_to_num(np.array(correlation_features))])

# def fiw_core_logic(search_feat, search_label, test_feat, test_label, num_neighbors):
#     """
#     这是你的 FIW 核心算法逻辑实现
#     """
#     # 1. 这里的标准化基准非常重要，必须确保 search_feat 包含了所有参考信息
#     scaler = StandardScaler()
#     search_scaled = scaler.fit_transform(search_feat)
#     test_scaled = scaler.transform(test_feat)

#     # 2. 有监督特征权重计算
#     unique_classes = np.unique(search_label)
#     n_features = search_scaled.shape[1]
#     feature_weights = np.zeros(n_features)
    
#     for label in unique_classes:
#         class_data = search_scaled[search_label == label]
#         if len(class_data) > 1:
#             class_var = np.var(class_data, axis=0)
#             weight = 1.0 / (class_var + 1e-5)
#             feature_weights += weight
    
#     feature_weights = feature_weights / len(unique_classes)
#     feature_weights = feature_weights / (np.max(feature_weights) + 1e-10)

#     # 3. 应用权重
#     search_weighted = search_scaled * feature_weights
#     test_weighted = test_scaled * feature_weights

#     # 4. 检索
#     nbrs = NearestNeighbors(n_neighbors=num_neighbors, metric='euclidean', n_jobs=-1)
#     nbrs.fit(search_weighted)
#     _, indices = nbrs.kneighbors(test_weighted)

#     # 5. 统计细节
#     all_labels = np.unique(test_label)
#     label_purities = {}
#     total_correct = 0
    
#     for lbl in all_labels:
#         idx_in_test = np.where(test_label == lbl)[0]
#         lbl_correct = 0
#         for i in idx_in_test:
#             hits = np.sum(search_label[indices[i]] == lbl)
#             lbl_correct += hits
#             total_correct += hits
#         label_purities[lbl] = (lbl_correct / (len(idx_in_test) * num_neighbors)) * 100

#     overall_purity = (total_correct / (len(test_label) * num_neighbors)) * 100
#     return overall_purity, label_purities

# # --- 3. 实验流程控制 ---

# def run_custom_experiment(target_wc, source_wcs):
#     print(f"\n🚀 启动实验: Target=WC{target_wc} | Sources={source_wcs}")

#     # A. 加载目标工况
#     xt_train, yt_train, xt_valid, yt_valid = load_wc_data(target_wc)
#     feat_t_train = np.array([extract_advanced_features(s) for s in tqdm(xt_train, desc="Target Train")])
#     feat_t_valid = np.array([extract_advanced_features(s) for s in tqdm(xt_valid, desc="Target Valid")])

#     # B. 加载源工况
#     all_s_feats, all_s_labels = [], []
#     for s_id in source_wcs:
#         xs_train, ys_train, _, _ = load_wc_data(s_id)
#         s_feat = np.array([extract_advanced_features(s) for s in tqdm(xs_train, desc=f"WC{s_id} Train")])
#         all_s_feats.append(s_feat)
#         all_s_labels.append(ys_train)

#     scope1_feat = np.concatenate(all_s_feats, axis=0)
#     scope1_label = np.concatenate(all_s_labels, axis=0)

#     # 方案 2 的库
#     scope2_feat = np.concatenate([scope1_feat, feat_t_train], axis=0)
#     scope2_label = np.concatenate([scope1_label, yt_train], axis=0)

#     # C. 计算结果
#     # 注意：这里调用的是同一个核心逻辑函数
#     overall1, detailed1 = fiw_core_logic(scope1_feat, scope1_label, feat_t_valid, yt_valid, NUM_NEIGHBORS)
#     overall2, detailed2 = fiw_core_logic(scope2_feat, scope2_label, feat_t_valid, yt_valid, NUM_NEIGHBORS)

#     # D. 打印报告
#     print(f"\n{'故障标签':<10} | {'方案1(S)':<12} | {'方案2(S+T)':<12} | {'提升'}")
#     print("-" * 50)
#     for lbl in sorted(detailed1.keys()):
#         p1, p2 = detailed1[lbl], detailed2[lbl]
#         print(f"{lbl:<14} | {p1:>11.2f}% | {p2:>11.2f}% | {p2-p1:>+7.2f}%")
#     print("-" * 50)
#     print(f"{'总体平均':<14} | {overall1:>11.2f}% | {overall2:>11.2f}% | {overall2-overall1:>+7.2f}%")

# if __name__ == "__main__":
#     run_custom_experiment(target_wc=8, source_wcs=[1, 2, ])


import numpy as np
import os
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy import signal

# --- 1. 配置路径与参数 ---
DATASET = "BJTU-gearbox"
BASE_PATH = f"few_shot_test/data/{DATASET}"
NUM_NEIGHBORS = 3  # 设定为1，只找最近的一个

# --- 2. 核心功能函数 ---

def load_wc_data(wc_id):
    path = os.path.join(BASE_PATH, f"WC{wc_id}")
    x_train = np.load(os.path.join(path, 'X_train.npy'), mmap_mode='c').astype(np.float32)
    y_train = np.load(os.path.join(path, 'y_train.npy'), mmap_mode='c').astype(str)
    x_valid = np.load(os.path.join(path, 'X_valid.npy'), mmap_mode='c').astype(np.float32)
    y_valid = np.load(os.path.join(path, 'y_valid.npy'), mmap_mode='c').astype(str)
    return x_train, y_train, x_valid, y_valid

def extract_advanced_features(time_series, fs=1000):
    """提取高级特征 (保持不变)"""
    if time_series.shape[0] < time_series.shape[1]: 
        time_series = time_series.T
    n_channels = time_series.shape[1]
    all_features = []
    for ch in range(n_channels):
        signal_data = time_series[:, ch]
        time_features = [
            np.mean(signal_data), np.std(signal_data), np.max(np.abs(signal_data)),
            np.max(signal_data) - np.min(signal_data), np.sqrt(np.mean(signal_data**2)),
            np.max(np.abs(signal_data)) / (np.sqrt(np.mean(signal_data**2)) + 1e-10),
            np.sum(signal_data**4) / (np.sum(signal_data**2)**2 + 1e-10),
            np.sum((signal_data - np.mean(signal_data))**3) / (len(signal_data) * np.std(signal_data)**3 + 1e-10)
        ]
        f, Pxx = signal.welch(signal_data, fs=fs, nperseg=min(1024, len(signal_data)), noverlap=512)
        freq_features = [f[np.argmax(Pxx)], np.sum(f * Pxx) / (np.sum(Pxx) + 1e-10), np.max(Pxx), np.mean(Pxx), np.std(Pxx)]
        all_features.extend(time_features + freq_features)
    correlation_features = [np.corrcoef(time_series[:, i], time_series[:, j])[0, 1] 
                            for i in range(n_channels) for j in range(i+1, n_channels)]
    return np.concatenate([np.array(all_features), np.nan_to_num(np.array(correlation_features))])

def euclidean_core_logic(search_feat, search_label, test_feat, test_label, num_neighbors):
    """
    [核心改动]：无监督高级欧氏距离检索逻辑
    """
    # 1. 整体标准化 (高级欧氏距离的基础)
    scaler = StandardScaler()
    search_scaled = scaler.fit_transform(search_feat)
    test_scaled = scaler.transform(test_feat)

    # 2. 直接进行最近邻搜索 (不再计算 feature_weights)
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, metric='euclidean', n_jobs=-1)
    nbrs.fit(search_scaled)
    _, indices = nbrs.kneighbors(test_scaled)

    # 3. 统计结果 (仅用于评估，检索过程不使用标签)
    all_labels = np.unique(test_label)
    label_purities = {}
    total_correct = 0
    
    for lbl in all_labels:
        idx_in_test = np.where(test_label == lbl)[0]
        lbl_correct = 0
        for i in idx_in_test:
            # 检查最近邻的标签是否与真值一致
            hits = np.sum(search_label[indices[i]] == lbl)
            lbl_correct += hits
            total_correct += hits
        label_purities[lbl] = (lbl_correct / (len(idx_in_test) * num_neighbors)) * 100

    overall_purity = (total_correct / (len(test_label) * num_neighbors)) * 100
    return overall_purity, label_purities

# --- 3. 实验流程控制 ---

def run_custom_experiment(target_wc, source_wcs):
    print(f"\n🚀 启动[无监督欧氏距离]实验: Target=WC{target_wc} | Sources={source_wcs}")

    xt_train, yt_train, xt_valid, yt_valid = load_wc_data(target_wc)
    feat_t_train = np.array([extract_advanced_features(s) for s in tqdm(xt_train, desc="Target Train")])
    feat_t_valid = np.array([extract_advanced_features(s) for s in tqdm(xt_valid, desc="Target Valid")])

    all_s_feats, all_s_labels = [], []
    for s_id in source_wcs:
        xs_train, ys_train, _, _ = load_wc_data(s_id)
        s_feat = np.array([extract_advanced_features(s) for s in tqdm(xs_train, desc=f"WC{s_id} Train")])
        all_s_feats.append(s_feat)
        all_s_labels.append(ys_train)

    scope1_feat = np.concatenate(all_s_feats, axis=0)
    scope1_label = np.concatenate(all_s_labels, axis=0)

    scope2_feat = np.concatenate([scope1_feat, feat_t_train], axis=0)
    scope2_label = np.concatenate([scope1_label, yt_train], axis=0)

    # 执行检索
    overall1, detailed1 = euclidean_core_logic(scope1_feat, scope1_label, feat_t_valid, yt_valid, NUM_NEIGHBORS)
    overall2, detailed2 = euclidean_core_logic(scope2_feat, scope2_label, feat_t_valid, yt_valid, NUM_NEIGHBORS)

    print(f"\n{'故障标签':<10} | {'方案1(S)':<12} | {'方案2(S+T)':<12} | {'提升'}")
    print("-" * 50)
    for lbl in sorted(detailed1.keys()):
        p1, p2 = detailed1[lbl], detailed2[lbl]
        print(f"{lbl:<14} | {p1:>11.2f}% | {p2:>11.2f}% | {p2-p1:>+7.2f}%")
    print("-" * 50)
    print(f"{'总体平均':<14} | {overall1:>11.2f}% | {overall2:>11.2f}% | {overall2-overall1:>+7.2f}%")

if __name__ == "__main__":
    run_custom_experiment(target_wc=6, source_wcs=[1, 2])