import json
import sys
import os
import numpy as np
from tqdm import tqdm
from scipy import signal
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# 路径防报错
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

# --- 1. 配置参数 ---
DATASET = "BJTU-gearbox"
BASE_PATH = f"few_shot_test/data/{DATASET}"
NUM_NEIGHBORS = 1  # 找最近的1个

# --- 2. 核心工具函数 ---

def extract_advanced_features(time_series, fs=1000):
    """(保持不变)"""
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

def load_labels_as_map(label_file_path):
    try:
        with open(label_file_path, 'r', encoding='utf-8') as f:
            label_list = json.load(f)
        return {item['index']: item['label'] for item in label_list}
    except Exception:
        return {}

# def calculate_retrieval_accuracy(retrieval_results_path, test_labels_path, train_labels_path):
#     """(保持不变)"""
#     test_label_map = load_labels_as_map(test_labels_path)
#     train_label_map = load_labels_as_map(train_labels_path)
    
#     with open(retrieval_results_path, 'r', encoding='utf-8') as f:
#         retrieval_data = json.load(f)

#     all_purities = []
#     class_stats = {}

#     for result_item in retrieval_data:
#         test_index = result_item.get('test_index')
#         neighbor_indices = result_item.get('neighbors', [])
        
#         true_test_label = test_label_map.get(test_index)
        
#         correct_neighbors = 0
#         for neighbor_idx in neighbor_indices:
#             neighbor_label = train_label_map.get(neighbor_idx)
#             if neighbor_label == true_test_label:
#                 correct_neighbors += 1
        
#         if len(neighbor_indices) > 0:
#             purity = correct_neighbors / len(neighbor_indices)
#             all_purities.append(purity)

#             # [新增] 2. 在循环内累计每个标签的纯度和计数
#             if true_test_label not in class_stats:
#                 class_stats[true_test_label] = [0.0, 0]
            
#             class_stats[true_test_label][0] += purity  # 累加纯度
#             class_stats[true_test_label][1] += 1       # 累加数量
            
#     mean_accuracy = np.mean(all_purities) * 100
    
#     # [新增] 3. 打印详细的分项结果
#     print(f"\n{'='*20} 详细分类结果 {'='*20}")
#     print(f"{'故障类型':<10} | {'准确率'}")
#     print("-" * 35)
    
#     # 按标签字母顺序排序打印
#     for label in sorted(class_stats.keys()):
#         total_purity, count = class_stats[label]
#         acc = (total_purity / count) * 100 if count > 0 else 0
#         print(f"{label:<14} | {acc:.2f}%")
        
#     print("-" * 35)
#     print(f"{'总体平均':<14} | {mean_accuracy:.2f}%")
#     print("=" * 46 + "\n")

#     return mean_accuracy



def calculate_retrieval_accuracy(retrieval_results_path, test_labels_path, train_labels_path):
    """(带调试版)"""
    print(f"\n--- 开始计算准确率 ---")
    
    # 1. 加载标签
    test_label_map = load_labels_as_map(test_labels_path)
    train_label_map = load_labels_as_map(train_labels_path)
    
    # [调试] 打印标签Map的基本信息
    print(f"[DEBUG] 测试集标签数量: {len(test_label_map)}")
    print(f"[DEBUG] 训练集标签数量: {len(train_label_map)}")
    
    # [调试] 打印前5个测试集索引，看看长什么样
    first_5_keys = list(test_label_map.keys())[:5]
    print(f"[DEBUG] 测试集索引示例(前5个): {first_5_keys}")

    with open(retrieval_results_path, 'r', encoding='utf-8') as f:
        retrieval_data = json.load(f)
    
    print(f"[DEBUG] 检索结果条目数: {len(retrieval_data)}")

    all_purities = []
    class_stats = {}

    for i, result_item in enumerate(retrieval_data):
        test_index = result_item.get('test_index')
        neighbor_indices = result_item.get('neighbors', [])
        
        # 获取真实标签
        true_test_label = test_label_map.get(test_index)
        
        # [关键调试] 如果找不到标签，立刻报错并打印详情，而不是跳过
        if true_test_label is None:
            print(f"\n[CRITICAL ERROR] 在第 {i} 条检索结果中发现异常！")
            print(f"  - 检索结果中的 test_index: {test_index} (类型: {type(test_index)})")
            print(f"  - test_label_map 中是否存在该Key? {test_index in test_label_map}")
            # 尝试转换类型再查一次，排除 int/str 不匹配的问题
            print(f"  - 尝试转为 int 查找: {test_label_map.get(int(test_index))}")
            print(f"  - 尝试转为 str 查找: {test_label_map.get(str(test_index))}")
            raise ValueError(f"无法找到 test_index={test_index} 的真实标签！请检查上述调试信息。")

        correct_neighbors = 0
        for neighbor_idx in neighbor_indices:
            neighbor_label = train_label_map.get(neighbor_idx)
            if neighbor_label == true_test_label:
                correct_neighbors += 1
        
        if len(neighbor_indices) > 0:
            purity = correct_neighbors / len(neighbor_indices)
            all_purities.append(purity)

            # 统计分项结果
            if true_test_label not in class_stats:
                class_stats[true_test_label] = [0.0, 0]
            class_stats[true_test_label][0] += purity
            class_stats[true_test_label][1] += 1
            
    mean_accuracy = np.mean(all_purities) * 100
    
    # 打印详细结果
    print(f"\n{'='*20} 详细分类结果 {'='*20}")
    print(f"{'故障类型':<10} | {'准确率'}")
    print("-" * 35)
    
    # 此时 class_stats 里绝对没有 None，可以放心排序
    for label in sorted(class_stats.keys()):
        total_purity, count = class_stats[label]
        acc = (total_purity / count) * 100 if count > 0 else 0
        print(f"{label:<14} | {acc:.2f}%")
        
    print("-" * 35)
    print(f"{'总体平均':<14} | {mean_accuracy:.2f}%")
    print("=" * 46 + "\n")

    return mean_accuracy
# --- 3. 核心检索函数 (改回原名 neighbor_find 并支持混合逻辑) ---

def neighbor_find_mixed(dataset, 
                        source_wcs, 
                        target_wc, 
                        target_n_shots, 
                        neighbor_num, 
                        dist_map):
    """
    专门处理混合工况：Source(全量) + Target(小样本)
    关键：完全保留原来的文件夹命名逻辑
    """
    
    # 构造原来的 train_nums 列表概念，用于命名
    # 注意顺序：先放 Source，最后放 Target
    train_nums = source_wcs + [target_wc]
    train_tag = "_".join(map(str, train_nums))
    
    print(f"正在加载训练数据 (Tag: {train_tag})...")

    # --- A. 动态构建训练集与标签 ---
    # 我们需要在内存里同时构建 features 和 merged_train_labels，确保索引一一对应
    
    train_feats_list = []
    merged_train_labels = {} # {0: 'G0', 1: 'G0', ...}
    global_idx = 0
    
    # 1. 加载 Source WCs (全量)
    for wc in source_wcs:
        path = os.path.join(BASE_PATH, f"WC{wc}")
        x = np.load(os.path.join(path, 'X_train.npy'), mmap_mode='c').astype(np.float32)
        y = np.load(os.path.join(path, 'y_train.npy'), mmap_mode='c').astype(str)
        
        # 提取特征
        print(f"  -> 提取 Source WC{wc} 特征...")
        feats = np.array([extract_advanced_features(s) for s in tqdm(x, leave=False)])
        train_feats_list.append(feats)
        
        # 记录标签
        for label in y:
            merged_train_labels[global_idx] = label
            global_idx += 1
            
    # 2. 加载 Target WC (小样本)
    path = os.path.join(BASE_PATH, f"WC{target_wc}")
    x_t = np.load(os.path.join(path, 'X_train.npy'), mmap_mode='c').astype(np.float32)
    y_t = np.load(os.path.join(path, 'y_train.npy'), mmap_mode='c').astype(str)
    
    # 筛选前 n_shots
    unique_labels = np.unique(y_t)
    sel_indices = []
    for lbl in unique_labels:
        idxs = np.where(y_t == lbl)[0]
        sel_indices.extend(idxs[:target_n_shots])
    
    x_t_few = x_t[sel_indices]
    y_t_few = y_t[sel_indices]
    
    print(f"  -> 提取 Target WC{target_wc} 小样本特征 ({len(x_t_few)}个)...")
    t_feats = np.array([extract_advanced_features(s) for s in x_t_few])
    train_feats_list.append(t_feats)
    
    for label in y_t_few:
        merged_train_labels[global_idx] = label
        global_idx += 1
    
    # 合并训练特征
    X_train_std = np.concatenate(train_feats_list, axis=0)
    
    # --- B. 加载测试集 (Target WC Valid) ---
    path = os.path.join(BASE_PATH, f"WC{target_wc}")
    x_valid = np.load(os.path.join(path, 'X_valid.npy'), mmap_mode='c').astype(np.float32)
    # y_valid 我们后面单独处理生成 test_index.json 用
    
    print(f"  -> 提取测试集 WC{target_wc} 特征...")
    X_test_std = np.array([extract_advanced_features(s) for s in tqdm(x_valid, leave=False)])
    
    # --- C. 标准化与检索 ---
    # 你的算法现在是纯欧氏距离，所以要把特征提取后的 X_train_std 放入 Scaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_std)
    X_test_scaled = scaler.transform(X_test_std)
    
    # 遍历 dist_map (虽然现在逻辑固定了，但为了保持文件夹结构)
    for dist_name, _ in dist_map.items():
        # [关键] 严格保持原本的文件夹命名格式！
        output_dir = os.path.join("few_shot_test", "data_index", dataset, 
                                  f"test_WC{target_wc}_train_WCs{train_tag}", 
                                  f"{dist_name}_dist")
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n计算近邻 ({dist_name})...")
        nbrs = NearestNeighbors(n_neighbors=neighbor_num, metric='euclidean', n_jobs=-1)
        nbrs.fit(X_train_scaled)
        _, indices = nbrs.kneighbors(X_test_scaled)
        
        results = []
        for i in range(len(indices)):
            results.append({
                "test_index": i,
                "neighbors": indices[i].tolist()
            })
            
        output_path = os.path.join(output_dir, f'nearest_{neighbor_num}_neighbors.json')
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"    -> 结果已保存: {output_path}")
        
        # 返回路径供下一步使用
        return output_path, merged_train_labels

# --- 4. 管道函数 (逻辑复原) ---

def pipeline(dataset, source_wcs, target_wc, target_n_shots, dist_map, neighbor_num):
    # 1. 生成原本的 test_index.json (测试集索引)
    # 因为测试集是固定的 WC{target_wc} 的 X_valid，我们可以直接用原来的 generate_json 逻辑
    # 或者为了简单，直接在这里写一个临时的
    path = os.path.join(BASE_PATH, f"WC{target_wc}")
    y_valid = np.load(os.path.join(path, 'y_valid.npy'), mmap_mode='c').astype(str)
    test_index_json = [{"index": i, "label": l} for i, l in enumerate(y_valid)]
    
    temp_test_labels_path = "temp_test_labels.json"
    with open(temp_test_labels_path, 'w') as f:
        json.dump(test_index_json, f)

    # 2. 执行检索 & 获取合并后的训练标签
    # 这里的检索函数会自动按旧格式保存 json 文件
    results_path, merged_train_labels = neighbor_find_mixed(
        dataset, source_wcs, target_wc, target_n_shots, neighbor_num, dist_map
    )
    
    # 3. 保存临时的合并训练标签 (因为是混合数据，不能用磁盘上原来的 json)
    temp_train_labels_path = "temp_merged_train_labels.json"
    train_json_list = [{"index": k, "label": v} for k, v in merged_train_labels.items()]
    with open(temp_train_labels_path, 'w') as f:
        json.dump(train_json_list, f)
        
    # 4. 计算准确率
    acc = calculate_retrieval_accuracy(results_path, temp_test_labels_path, temp_train_labels_path)
    
    # 清理临时文件 (可选)
    # os.remove(temp_test_labels_path)
    # os.remove(temp_train_labels_path)
    
    return acc

if __name__ == "__main__":
    import datetime
    
    dataset = 'BJTU-gearbox'
    # 这里的名字 FIW 会决定文件夹叫 FIW_dist，虽然内部已经是欧氏距离
    dist_map = {'FIW': None} 
    neighbor_num = 3
    
    # 配置实验
    target_wc = 8       # 目标测试工况
    source_wcs = [1, 2, 3, 4, 5, 7, 9]  # 源工况
    target_n_shots = 3
    # 目标工况混入几个样本
    
    print(f"\n{'='*60}")
    print(f"🚀 实验开始: Source={source_wcs} + Target=WC{target_wc}({target_n_shots} shot)")
    print(f"{'='*60}")
    
    acc = pipeline(dataset, source_wcs, target_wc, target_n_shots, dist_map, neighbor_num)
    
    print(f"\n最终准确率: {acc:.2f}%")
    