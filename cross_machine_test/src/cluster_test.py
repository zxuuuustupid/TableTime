import numpy as np
import os
import json
from tqdm import tqdm
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy import stats  # 增加这一行

# =========================================================
# 1. 算法部分 (完全复用你提供的代码)
# =========================================================

# def extract_advanced_features(time_series, fs=1000):
#     if time_series.shape[0] < time_series.shape[1]: 
#         time_series = time_series.T
#     n_channels = time_series.shape[1]
#     all_features = []
    
#     for ch in range(n_channels):
#         sig = time_series[:, ch]
#         # 0. 基础去均值
#         sig = sig - np.mean(sig)
#         rms = np.sqrt(np.mean(sig**2)) + 1e-10
        
#         # 1. 【核心：无量纲时域特征】—— 这些指标跨机器非常鲁棒
#         # 它们衡量的是“有多像故障”，而不是“振动有多大”
#         # kur = signal.kurtosis(sig)           # 峭度（反映冲击）
#         # skw = signal.skew(sig)
#         # # 偏度（反映不对称）
#         kur = stats.kurtosis(sig)           # 峭度（反映冲击）
#         skw = stats.skew(sig)               # 偏度（反映不对
#         crest = np.max(np.abs(sig)) / rms    # 峰值因子
#         shape = rms / (np.mean(np.abs(sig)) + 1e-10) # 波形因子
#         impulse = np.max(np.abs(sig)) / (np.mean(np.abs(sig)) + 1e-10) # 脉冲因子
        
#         # 2. 【核心：频谱能量分布】—— 关注能量分布在哪些频段
#         f, Pxx = signal.welch(sig, fs=fs, nperseg=256)
#         # 将频谱平分成 8 个频段，计算每个频段占总能量的比例
#         Pxx_norm = Pxx / (np.sum(Pxx) + 1e-10)
#         bands = np.array_split(Pxx_norm, 8)
#         band_energies = [np.sum(b) for b in bands]
        
#         all_features.extend([kur, skw, crest, shape, impulse] + band_energies)
        
#     return np.array(all_features)


def extract_advanced_features(time_series, fs=1000):
    if time_series.shape[0] < time_series.shape[1]: 
        time_series = time_series.T
        
    n_channels = time_series.shape[1]
    all_features = []
    
    for ch in range(n_channels):
        signal_data = time_series[:, ch]
        
        # === [核心修改：在此处加入信号标准化] ===
        # 这一行能消除不同机器传感器增益、功率导致的幅值巨大差异
        signal_data = (signal_data - np.mean(signal_data)) / (np.std(signal_data) + 1e-10)
        # =====================================

        # 随后的时域特征（均值、标准差、最大值等）将基于标准化后的信号计算
        time_features = [
            np.mean(signal_data), # 标准化后该值趋于0
            np.std(signal_data),  # 标准化后该值趋于1
            np.max(np.abs(signal_data)),
            # ... 其余代码不变
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
    
    correlation_features = []
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            corr = np.corrcoef(time_series[:, i], time_series[:, j])[0, 1]
            correlation_features.append(corr if not np.isnan(corr) else 0)
            
    return np.concatenate([np.array(all_features), np.array(correlation_features)])

def find_nearest_neighbors_weighted_feature(train_data, train_labels, test_data, num_neighbors):
    print("Extracting advanced features...")
    train_features = np.array([extract_advanced_features(seq) for seq in tqdm(train_data, desc="Train Feat")])
    test_features = np.array([extract_advanced_features(seq) for seq in tqdm(test_data, desc="Test Feat")])

    # --- [核心修改 1：独立标准化] ---
    # 训练集用自己的均值方差，测试集也用自己的均值方差
    # 这一步能强行抵消掉不同机器带来的全局特征偏移
    train_features_scaled = StandardScaler().fit_transform(train_features)
    test_features_scaled = StandardScaler().fit_transform(test_features)
    
    print("Calculating supervised feature weights...")
    unique_classes = np.unique(train_labels)
    n_features = train_features_scaled.shape[1]
    feature_weights = np.zeros(n_features)
    
    for label in unique_classes:
        class_mask = (train_labels == label)
        class_data = train_features_scaled[class_mask]
        
        if len(class_data) > 1:
            class_var = np.var(class_data, axis=0)
            # --- [核心修改 2：增大平滑项] ---
            # 跨机器时，权重不能给得太极端，0.1 是平衡鲁棒性的经验值
            weight = 1.0 / (class_var + 0.2) 
            feature_weights += weight
    
    feature_weights = feature_weights / len(unique_classes)
    feature_weights = feature_weights / (np.max(feature_weights) + 1e-10)
    
    print("Applying feature weights...")
    train_weighted = train_features_scaled * feature_weights
    test_weighted = test_features_scaled * feature_weights
    
    print(f"Searching for {num_neighbors} nearest neighbors...")
    # --- [核心修改 3：改用余弦距离] ---
    # metric='cosine' 只看特征向量的方向，不看长度，对跨机器环境极其有效
    # nbrs = NearestNeighbors(n_neighbors=num_neighbors, algorithm='auto', metric='cosine', n_jobs=-1)
    
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, metric='cosine', n_jobs=-1)
    nbrs.fit(train_weighted)
    
    distances, indices = nbrs.kneighbors(test_weighted)

    results = []
    for test_index in range(len(test_data)):
        results.append({
            "test_index": test_index, 
            "neighbors": indices[test_index].tolist()
        })
    return results, train_weighted, test_weighted
    
    
def visualize_results(X_train_raw, X_test_raw, train_feat_w, test_feat_w, y_train, y_test, neighbor_results):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    
    # 1. 空间特征分布图 (t-SNE)
    print("正在生成空间分布图 (t-SNE)...")
    tsne = TSNE(n_components=2, init='pca', random_state=42)
    all_feats = np.vstack([train_feat_w, test_feat_w])
    all_2d = tsne.fit_transform(all_feats)
    
    train_2d = all_2d[:len(train_feat_w)]
    test_2d = all_2d[len(train_feat_w):]

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    # 绘制训练集 (用空心圆表示，颜色区分故障)
    for lbl in np.unique(y_train):
        idx = np.where(y_train == lbl)
        plt.scatter(train_2d[idx, 0], train_2d[idx, 1], label=f'Train-{lbl}', alpha=0.3, marker='o')
    # 绘制测试集 (用星号表示，颜色区分故障)
    for lbl in np.unique(y_test):
        idx = np.where(y_test == lbl)
        plt.scatter(test_2d[idx, 0], test_2d[idx, 1], label=f'Test-{lbl}', marker='x', edgecolors='black')
    plt.title("Spatial Feature Distribution (Weighted)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # 2. 折线图对比 (取测试集第1个样本及其最近邻)
    plt.subplot(1, 2, 2)
    test_idx = 0 
    nei_idx = neighbor_results[test_idx]['neighbors'][0] # 最近的一个
    
    # 这里的维度是 (1, 2048), 取 [0] 变成一维
    plt.plot(X_test_raw[test_idx][0], label=f'Test Sample (Class: {y_test[test_idx]})', alpha=0.8)
    plt.plot(X_train_raw[nei_idx][0], label=f'Nearest Neighbor (Class: {y_train[nei_idx]})', alpha=0.6, linestyle='--')
    plt.title("Raw Signal Comparison (Test vs Neighbor)")
    plt.legend()
    
    plt.tight_layout()
    plt.show() # 只显示，不保存

# =========================================================
# 2. 检索执行程序
# =========================================================

def run_retrieval():
    ROOT = "cross_machine_test"
    DATA_ROOT = os.path.join(ROOT, "data")
    INDEX_ROOT = os.path.join(ROOT, "data_index")
    NUM_NEIGHBORS = 5
    
    # 1. 定义源（训练集/检索库）
    # 假设使用 BJTU WC1 作为源
    source_name = "BJTU_leftaxlebox/WC1"
    train_x_path = os.path.join(DATA_ROOT, source_name, "X_train.npy")
    train_y_path = os.path.join(DATA_ROOT, source_name, "y_train.npy")
    
    if not os.path.exists(train_x_path):
        print(f"错误: 找不到源数据 {train_x_path}")
        return

    X_train = np.load(train_x_path)
    y_train = np.load(train_y_path)
    
    # 2. 定义所有需要查询的目标（测试集）
    # 包括 BJTU 自己的验证集和 Ottawa 的验证集
    test_tasks = [
        {"name": "BJTU_leftaxlebox_WC1", "path": "BJTU_leftaxlebox/WC1"},
        {"name": "BJTU_gearbox_WC1", "path": "BJTU_gearbox/WC1"},
        # {"name": "Ottawa_A", "path": "Ottawa/A"},
        # {"name": "Ottawa_B", "path": "Ottawa/B"},
        # {"name": "Ottawa_C", "path": "Ottawa/C"},
        # {"name": "Ottawa_D", "path": "Ottawa/D"},
        # {"name": "swjtu",    "path": "swjtu/WC1"}
    ]
    
    for task in test_tasks:
        print(f"\n🚀 开始检索任务: 查询 {task['name']} -> 源 {source_name}")
        
        test_x_path = os.path.join(DATA_ROOT, task['path'], "X_valid.npy")
        if not os.path.exists(test_x_path):
            print(f"跳过: 找不到测试数据 {test_x_path}")
            continue
            
        X_test = np.load(test_x_path)
        
        # 执行检索算法 (取最近的 5 个邻居)
        neighbor_results, train_feat_w, test_feat_w = find_nearest_neighbors_weighted_feature(
            train_data=X_train, train_labels=y_train, test_data=X_test, num_neighbors=NUM_NEIGHBORS
        )
        
        # === [新增：计算聚类纯度] ===
        test_y_path = os.path.join(DATA_ROOT, task['path'], "y_valid.npy")
        y_test = np.load(test_y_path)  # 加载测试集真实标签
        
        total_correct = 0
        num_neighbors = len(neighbor_results[0]['neighbors'])
        
        for i, res in enumerate(neighbor_results):
            true_label = y_test[i]
            # 获取这 k 个邻居在训练集里的标签
            neighbor_labels = y_train[res['neighbors']]
            # 统计命中数
            total_correct += np.sum(neighbor_labels == true_label)
            
        purity = (total_correct / (len(y_test) * num_neighbors)) * 100
        print(f"📊 聚类纯度 (Purity@k={num_neighbors}): {purity:.2f}%")
        # =========================
        
        # 3. [新增] 调用可视化
        visualize_results(X_train, X_test, train_feat_w, test_feat_w, y_train, y_test, neighbor_results)
        
        
        # 3. 保存结果
        output_dir = os.path.join(INDEX_ROOT, task['name'])
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"nearest_{NUM_NEIGHBORS}_neighbors.json")
        
        with open(output_file, 'w') as f:
            json.dump(neighbor_results, f, indent=4)
            
        print(f"✅ 任务完成! 结果已保存至: {output_file}")

if __name__ == "__main__":
    run_retrieval()