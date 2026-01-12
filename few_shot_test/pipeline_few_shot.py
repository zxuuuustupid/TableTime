import json
import sys
import os
import datetime
import numpy as np
from tqdm import tqdm
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# 获取当前脚本所在目录的上一层目录（即项目根目录）
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# --- 1. 配置参数 ---
DATASET = "BJTU-gearbox"
# 确保这里是你存放数据的真实路径
BASE_PATH = f"few_shot_test/data/{DATASET}" 

# 核心设置：每个工况、每个类别放入库中的样本数
# 如果设为 1，就是 1-shot (极度稀疏)；设为 3 就是 3-shot
N_SHOTS = 3  

# 找最近邻的个数 (建议设为 1，因为样本太少了)
NUM_NEIGHBORS = 3 

# --- 2. 核心功能函数 ---

def extract_advanced_features(time_series, fs=1000):
    """
    高级特征提取：时域 + 频域 + 通道相关性
    """
    if time_series.shape[0] < time_series.shape[1]: 
        time_series = time_series.T
    n_channels = time_series.shape[1]
    all_features = []
    
    for ch in range(n_channels):
        signal_data = time_series[:, ch]
        # 时域特征
        time_features = [
            np.mean(signal_data), np.std(signal_data), np.max(np.abs(signal_data)),
            np.max(signal_data) - np.min(signal_data), np.sqrt(np.mean(signal_data**2)),
            np.max(np.abs(signal_data)) / (np.sqrt(np.mean(signal_data**2)) + 1e-10),
            np.sum(signal_data**4) / (np.sum(signal_data**2)**2 + 1e-10),
            np.sum((signal_data - np.mean(signal_data))**3) / (len(signal_data) * np.std(signal_data)**3 + 1e-10)
        ]
        # 频域特征
        f, Pxx = signal.welch(signal_data, fs=fs, nperseg=min(1024, len(signal_data)), noverlap=512)
        freq_features = [f[np.argmax(Pxx)], np.sum(f * Pxx) / (np.sum(Pxx) + 1e-10), np.max(Pxx), np.mean(Pxx), np.std(Pxx)]
        all_features.extend(time_features + freq_features)
    
    # 通道相关性
    correlation_features = [np.corrcoef(time_series[:, i], time_series[:, j])[0, 1] 
                            for i in range(n_channels) for j in range(i+1, n_channels)]
    return np.concatenate([np.array(all_features), np.nan_to_num(np.array(correlation_features))])

def load_few_shot_data(wc_id, n_shots):
    """
    [核心修改] 加载数据，并严格模拟小样本 (Few-Shot)
    从训练集中，每个类别只取前 n_shots 个样本
    """
    path = os.path.join(BASE_PATH, f"WC{wc_id}")
    x_train = np.load(os.path.join(path, 'X_train.npy'), mmap_mode='c').astype(np.float32)
    y_train = np.load(os.path.join(path, 'y_train.npy'), mmap_mode='c').astype(str)
    
    # 验证集作为测试集，全部保留
    x_valid = np.load(os.path.join(path, 'X_valid.npy'), mmap_mode='c').astype(np.float32)
    y_valid = np.load(os.path.join(path, 'y_valid.npy'), mmap_mode='c').astype(str)
    
    # --- 制作小样本训练集 ---
    unique_labels = np.unique(y_train)
    selected_indices = []
    
    for label in unique_labels:
        # 找到该标签的所有索引
        indices = np.where(y_train == label)[0]
        # 只取前 n_shots 个
        selected_indices.extend(indices[:n_shots])
    
    selected_indices = np.array(selected_indices)
    x_few_shot = x_train[selected_indices]
    y_few_shot = y_train[selected_indices]
    
    return x_few_shot, y_few_shot, x_valid, y_valid

def euclidean_retrieval(train_feat, train_labels, test_feat, test_labels, num_neighbors):
    """
    使用标准化欧氏距离进行检索 (无监督，不使用标签权重)
    """
    # 1. 整体标准化 (非常重要：消除特征量级差异)
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_feat)
    test_scaled = scaler.transform(test_feat)
    
    # 2. 最近邻搜索
    nbrs = NearestNeighbors(n_neighbors=num_neighbors, metric='euclidean', n_jobs=-1)
    nbrs.fit(train_scaled)
    _, indices = nbrs.kneighbors(test_scaled)
    
    # 3. 计算准确率
    total_correct = 0
    total_samples = len(test_labels)
    
    for i in range(total_samples):
        # 找到的近邻标签
        neighbor_labels = train_labels[indices[i]]
        # 真实标签
        true_label = test_labels[i]
        
        # 统计命中数
        hits = np.sum(neighbor_labels == true_label)
        total_correct += hits
        
    # 计算纯度/准确率
    accuracy = (total_correct / (total_samples * num_neighbors)) * 100
    return accuracy

# --- 3. 实验主流程 ---

def run_few_shot_experiment(train_wcs, test_wcs, n_shots):
    print(f"\n{'='*60}")
    print(f"🧪 启动小样本实验 (N_SHOTS={n_shots})")
    print(f"📚 知识库包含工况: {train_wcs}")
    print(f"🎯 测试目标工况: {test_wcs}")
    print(f"{'='*60}")

    # --- A. 构建全局小样本知识库 ---
    print("正在构建知识库 (提取特征)...")
    library_feats = []
    library_labels = []
    
    for wc in train_wcs:
        # 加载经过筛选的小样本
        x_s, y_s, _, _ = load_few_shot_data(wc, n_shots)
        
        # 提取特征
        feats = np.array([extract_advanced_features(s) for s in x_s]) # 这里数据量很小，不用tqdm也没事
        
        library_feats.append(feats)
        library_labels.append(y_s)
        
    # 合并成大库
    X_library = np.concatenate(library_feats, axis=0)
    y_library = np.concatenate(library_labels, axis=0)
    
    print(f"✅ 知识库构建完成: 总样本数 {len(X_library)} (来自 {len(train_wcs)} 个工况)")

    # --- B. 逐个工况进行测试 ---
    results = []
    
    for test_wc in test_wcs:
        print(f"\n>>> 正在测试工况 WC{test_wc} ...")
        
        # 加载测试集 (X_valid)
        _, _, x_test, y_test = load_few_shot_data(test_wc, n_shots) # n_shots这里不影响valid
        
        # 提取测试集特征
        print(f"    提取测试集特征 ({len(x_test)} 样本)...")
        X_test_feat = np.array([extract_advanced_features(s) for s in tqdm(x_test, leave=False)])
        
        # 执行检索评估
        acc = euclidean_retrieval(X_library, y_library, X_test_feat, y_test, NUM_NEIGHBORS)
        
        print(f"    🎯 WC{test_wc} 诊断准确率: {acc:.2f}%")
        results.append(acc)

    # --- C. 汇总报告 ---
    avg_acc = np.mean(results)
    print(f"\n{'='*60}")
    print(f"🏆 实验结束 | 平均准确率: {avg_acc:.2f}%")
    print(f"{'='*60}")
    
    return results, avg_acc

if __name__ == "__main__":
    # 配置：所有的工况都参与知识库构建，也参与测试
    # 模拟场景：我们有一个包含所有工况数据的库，但每个工况只有极少的样本
    all_wcs = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    
    # 运行实验
    # 这里的 all_wcs 既是 source 也是 target，因为我们要看 knowledge base 能否覆盖所有情况
    accuracies, avg = run_few_shot_experiment(train_wcs=all_wcs, test_wcs=all_wcs, n_shots=N_SHOTS)
    
    # 保存结果到日志
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "few_shot_test/result/log_fewshot"
    os.makedirs(log_dir, exist_ok=True)
    
    log_content = [
        f"Time: {timestamp}",
        f"Dataset: {DATASET}",
        f"N_Shots: {N_SHOTS}",
        f"Neighbors (k): {NUM_NEIGHBORS}",
        f"Train WCs: {all_wcs}",
        "-" * 30
    ]
    
    for wc, acc in zip(all_wcs, accuracies):
        log_content.append(f"Test WC{wc}: {acc:.2f}%")
    
    log_content.append("-" * 30)
    log_content.append(f"Average Accuracy: {avg:.2f}%")
    
    with open(os.path.join(log_dir, f"fewshot_results_{timestamp}.txt"), 'w') as f:
        f.write("\n".join(log_content))