import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy import signal
from tqdm import tqdm
import time

def extract_advanced_features(time_series, fs=1000):
    """
    高明的特征提取：多尺度时频特征融合
    输入: time_series (5000, 6) - 单个样本的6通道时序数据
    输出: feature_vector - 融合特征向量
    """
    n_channels = time_series.shape[1]
    all_features = []
    
    for ch in range(n_channels):
        signal_data = time_series[:, ch]
        
        # 1. 时域特征 (12个)
        time_features = [
            np.mean(signal_data),
            np.std(signal_data),
            np.max(np.abs(signal_data)),
            np.min(signal_data),
            np.max(signal_data) - np.min(signal_data),  # 峰峰值
            np.sqrt(np.mean(signal_data**2)),  # RMS
            np.max(np.abs(signal_data)) / np.sqrt(np.mean(signal_data**2)),  # 峰值因子
            np.sum(np.abs(np.diff(signal_data))) / (len(signal_data)-1),  # 平均变化率
            np.mean(np.abs(signal_data - np.mean(signal_data))),  # 平均绝对偏差
            np.percentile(signal_data, 75) - np.percentile(signal_data, 25),  # 四分位距
            np.sum(signal_data**4) / (np.sum(signal_data**2)**2 + 1e-10),  # 峭度
            np.sum((signal_data - np.mean(signal_data))**3) / (len(signal_data) * np.std(signal_data)**3 + 1e-10)  # 偏度
        ]
        
        # 2. 频域特征 (10个) - 使用Welch方法
        f, Pxx = signal.welch(signal_data, fs=fs, nperseg=1024, noverlap=512)
        dominant_freq = f[np.argmax(Pxx)]
        spectral_centroid = np.sum(f * Pxx) / np.sum(Pxx)
        spectral_bandwidth = np.sqrt(np.sum(((f - spectral_centroid)**2) * Pxx) / np.sum(Pxx))
        spectral_entropy = -np.sum((Pxx/np.sum(Pxx)) * np.log2(Pxx/np.sum(Pxx) + 1e-10))
        
        freq_features = [
            dominant_freq,
            spectral_centroid,
            spectral_bandwidth,
            spectral_entropy,
            np.max(Pxx),
            np.mean(Pxx),
            np.std(Pxx),
            np.sum(Pxx[:len(f)//4]),  # 低频能量
            np.sum(Pxx[len(f)//4:len(f)//2]),  # 中频能量
            np.sum(Pxx[len(f)//2:])  # 高频能量
        ]
        
        # 3. 时频域特征 (4个) - 频带统计特征
        f, Pxx = signal.welch(signal_data, fs=1000, nperseg=1024)
        total_power = np.sum(Pxx)
        low_freq = np.sum(Pxx[f <= 50]) / total_power if total_power > 0 else 0
        mid_freq = np.sum(Pxx[(f > 50) & (f <= 200)]) / total_power if total_power > 0 else 0
        high_freq = np.sum(Pxx[f > 200]) / total_power if total_power > 0 else 0
        spectral_flatness = np.exp(np.mean(np.log(Pxx + 1e-10))) / np.mean(Pxx + 1e-10)
        wavelet_features = [low_freq, mid_freq, high_freq, spectral_flatness]
        
        # 合并所有特征
        channel_features = np.concatenate([time_features, freq_features, wavelet_features])
        all_features.append(channel_features)
    
    # 4. 通道间相关性特征 (15个)
    correlation_features = []
    for i in range(n_channels):
        for j in range(i+1, n_channels):
            corr = np.corrcoef(time_series[:, i], time_series[:, j])[0, 1]
            correlation_features.append(corr)
    
    # 5. 多尺度统计特征 (6个)
    multi_scale_features = []
    signal_data = time_series[:, 0]  # 使用第一个通道进行多尺度分析
    for scale in [100, 500, 1000, 2500]:
        if len(signal_data) > scale:
            segments = np.array_split(signal_data, len(signal_data)//scale)
            segment_means = [np.mean(seg) for seg in segments]
            segment_stds = [np.std(seg) for seg in segments]
            multi_scale_features.extend([
                np.std(segment_means),
                np.std(segment_stds),
                np.max(segment_means) - np.min(segment_means),
                np.max(segment_stds) - np.min(segment_stds),
                np.mean(np.abs(np.diff(segment_means))),
                np.mean(np.abs(np.diff(segment_stds)))
            ])
    
    # 最终特征向量
    final_features = np.concatenate([
        np.array(all_features).flatten(),
        np.array(correlation_features),
        np.array(multi_scale_features)
    ])
    
    return final_features

def adaptive_distance_metric(X_train, y_train, n_neighbors=10):
    """
    高明的自适应距离度量学习
    通过分析训练数据的分布特性，自适应调整不同特征维度的权重
    """
    from sklearn.preprocessing import MinMaxScaler
    
    # 1. 计算每个样本的局部密度
    nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm='auto').fit(X_train)
    distances, indices = nbrs.kneighbors(X_train)
    local_density = 1.0 / (np.mean(distances[:, 1:], axis=1) + 1e-10)
    
    # 2. 为每个类别计算特征重要性
    unique_classes = np.unique(y_train)
    class_weights = np.zeros((len(unique_classes), X_train.shape[1]))
    
    for i, cls in enumerate(unique_classes):
        class_mask = (y_train == cls)
        class_data = X_train[class_mask]
        
        if len(class_data) > 1:
            # 计算类内方差
            class_var = np.var(class_data, axis=0)
            # 类内方差小的特征更重要（更具有判别性）
            feature_importance = 1.0 / (class_var + np.mean(class_var))
            class_weights[i] = feature_importance / np.sum(feature_importance)
    
    # 3. 为每个样本分配权重（基于局部密度和类别）
    sample_weights = np.zeros((X_train.shape[0], X_train.shape[1]))
    for i in range(X_train.shape[0]):
        cls = y_train[i]
        cls_idx = np.where(unique_classes == cls)[0][0]
        # 结合局部密度和类别权重
        sample_weights[i] = class_weights[cls_idx] * (local_density[i] / np.max(local_density))
    
    # 4. 归一化权重
    scaler = MinMaxScaler()
    sample_weights = scaler.fit_transform(sample_weights)
    
    return sample_weights

def sophisticated_clustering_analysis(target_condition, source_conditions, fault_types):
    """
    高明的聚类分析程序
    参数:
    - target_condition: int, 目标工况（数据A）
    - source_conditions: list of int, 源工况列表（数据B）
    - fault_types: list of int, 要聚类的故障类型列表
    """
    
    # =============== 1. 路径设置 ===============
    current_dir = Path(__file__).parent.absolute() if '__file__' in globals() else Path.cwd()
    output_dir = current_dir / "output"
    result_dir = current_dir / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    
    # =============== 2. 加载数据 ===============
    # 加载目标数据A
    data_A = []
    labels_A = []
    
    for fault_type in fault_types:
        file_path = output_dir / f"G{fault_type}_WC{target_condition}.npy"
        if not file_path.exists():
            continue
        
        samples = np.load(file_path)[:50]
        data_A.append(samples)
        labels_A.extend([fault_type] * len(samples))
    
    if not data_A:
        raise ValueError("没有找到有效的目标数据A")
    
    data_A = np.vstack(data_A)
    labels_A = np.array(labels_A)
    
    # 加载源数据B
    data_B = []
    labels_B = []
    
    for condition in source_conditions:
        if condition == target_condition:  # 确保排除目标工况
            continue
        for fault_type in fault_types:
            file_path = output_dir / f"G{fault_type}_WC{condition}.npy"
            if not file_path.exists():
                continue
            samples = np.load(file_path)[:50]
            data_B.append(samples)
            labels_B.extend([fault_type] * len(samples))
    
    if not data_B:
        raise ValueError("没有找到有效的源数据B")
    
    data_B = np.vstack(data_B)
    labels_B = np.array(labels_B)
    
    # =============== 3. 高级特征提取 ===============
    # 为A提取特征
    features_A = []
    for i in range(len(data_A)):
        features = extract_advanced_features(data_A[i])
        features_A.append(features)
    features_A = np.array(features_A)
    
    # 为B提取特征
    features_B = []
    for i in range(len(data_B)):
        features = extract_advanced_features(data_B[i])
        features_B.append(features)
    features_B = np.array(features_B)
    
    # 特征标准化
    scaler = StandardScaler()
    features_B_scaled = scaler.fit_transform(features_B)
    features_A_scaled = scaler.transform(features_A)
    
    # =============== 4. 自适应距离度量学习 ===============
    adaptive_weights = adaptive_distance_metric(features_B_scaled, labels_B)
    
    # 应用自适应权重
    weighted_features_B = features_B_scaled * adaptive_weights.mean(axis=0)
    weighted_features_A = features_A_scaled * adaptive_weights.mean(axis=0)
    
    # =============== 5. 邻居搜索 ===============
    nbrs = NearestNeighbors(n_neighbors=50, algorithm='auto', metric='euclidean', n_jobs=-1)
    nbrs.fit(weighted_features_B)
    distances, indices = nbrs.kneighbors(weighted_features_A)
    
    # =============== 6. 保存结果 ===============
    result_file = result_dir / f"clustering_result_C{target_condition}_vs_{'_'.join(map(str, source_conditions))}_faults{'_'.join(map(str, fault_types))}.npz"
    np.savez(result_file,
             distances=distances,
             neighbor_indices=indices,
             labels_A=labels_A,
             labels_B=labels_B)
    
    # =============== 7. 聚类准确率计算（修正为邻居纯度）==============
    # 获取每个A样本对应的50个B邻居的真实标签
    neighbor_labels = labels_B[indices]  # shape: (N_A, 50)
    
    # 计算每个A样本的50个邻居中，与自身真实标签一致的比例
    match_ratios = np.mean(neighbor_labels == labels_A[:, None], axis=1)
    
    # 总体纯度：所有A样本的平均一致比例
    total_accuracy = np.mean(match_ratios)
    
    # 按故障类型计算纯度
    class_accuracies = {}
    for fault_type in fault_types:
        mask = (labels_A == fault_type)
        if np.sum(mask) > 0:
            class_acc = np.mean(match_ratios[mask])
            class_accuracies[fault_type] = class_acc
    
    return {
        'total_accuracy': total_accuracy,
        'class_accuracies': class_accuracies,
        'result_file': result_file
    }


if __name__ == "__main__":
    import argparse

    # =============== 输入接口配置 ===============
    parser = argparse.ArgumentParser(description="跨工况机械故障聚类分析工具")
    
    # 接口 1: 选择单一目标工况 (待测数据)
    parser.add_argument('--target', type=int, default=1, 
                        help='指定待测的目标工况编号 (例如: 1)')
    
    # 接口 2: 选择多个源工况 (用于检索的样本池)
    parser.add_argument('--sources', type=int, nargs='+', default=[2, 3, 4], 
                        help='指定包含在检索样本池中的工况编号列表 (例如: 2 3 4)')
    
    # 接口 3: 选择故障类型 (可选)
    parser.add_argument('--faults', type=int, nargs='+', default=list(range(0, 9)), 
                        help='指定参与分析的故障类型 (默认 0-8)')

    args = parser.parse_args()

    # 将输入赋值给变量
    target_wc = args.target
    source_wcs = args.sources
    fault_list = args.faults

    # =============== 执行校验与分析 ===============
    print("="*40)
    print(f"🚀 开始实验分析")
    print(f"📍 待测目标工况 (Target): WC{target_wc}")
    print(f"📚 检索样本来源 (Sources): {[f'WC{c}' for c in source_wcs]}")
    print(f"🛠️ 故障类型范围: G{fault_list[0]} - G{fault_list[-1]}")
    print("="*40)

    # 逻辑检查：防止目标工况出现在源工况中（导致数据泄露）
    if target_wc in source_wcs:
        print(f"⚠️ 警告: 目标工况 {target_wc} 同时也出现在源工קות列表中！")
        print(f"系统将自动从检索池中剔除工况 {target_wc} 以保证实验严谨性。")
        source_wcs = [c for c in source_wcs if c != target_wc]

    try:
        # 执行核心分析逻辑
        result = sophisticated_clustering_analysis(
            target_condition=target_wc,
            source_conditions=source_wcs,
            fault_types=fault_list
        )

        # =============== 输出本次特定实验的结果 ===============
        print("\n" + "✅ 分析完成".center(34, "-"))
        print(f"总体准确率 (Retrieval Purity): {result['total_accuracy']:.4f}")
        print("-" * 40)
        print("各故障类型准确率细节:")
        for f_type, acc in result['class_accuracies'].items():
            print(f"  故障 G{f_type}: {acc:.4f}")
        print("-" * 40)
        print(f"结果文件已保存至: {result['result_file']}")

    except Exception as e:
        print(f"❌ 运行失败: {str(e)}")

# if __name__ == "__main__":
#     # 所有工况列表
#     all_conditions = list(range(1, 10))
#     # 所有故障类型
#     all_fault_types = list(range(0, 9))
    
#     # 存储所有结果
#     all_results = {}
    
#     # 循环处理每种工况作为目标
#     for target_condition in all_conditions:
#         # 源工况 = 所有工况 - 目标工况
#         source_conditions = [c for c in all_conditions if c != target_condition]
        
#         try:
#             # 执行聚类分析
#             result = sophisticated_clustering_analysis(
#                 target_condition=target_condition,
#                 source_conditions=source_conditions,
#                 fault_types=all_fault_types
#             )
            
#             # 保存结果
#             all_results[target_condition] = result
            
#         except Exception as e:
#             print(f"❌ 工况 {target_condition} 处理失败: {str(e)}")
#             continue
    
#     # 输出最终结果
#     print("工况\t总体准确率")
#     print("-" * 20)
#     total_accuracies = []
#     for condition in sorted(all_results.keys()):
#         acc = all_results[condition]['total_accuracy']
#         total_accuracies.append(acc)
#         print(f"{condition}\t{acc:.4f}")
    
#     if total_accuracies:
#         avg_accuracy = np.mean(total_accuracies)
#         max_accuracy = np.max(total_accuracies)
#         min_accuracy = np.min(total_accuracies)
#         max_cond = sorted(all_results.keys())[np.argmax(total_accuracies)]
#         min_cond = sorted(all_results.keys())[np.argmin(total_accuracies)]
        
#         print("-" * 20)
#         print(f"平均准确率: {avg_accuracy:.4f}")
#         print(f"最高准确率: {max_accuracy:.4f} (工况 {max_cond})")
#         print(f"最低准确率: {min_accuracy:.4f} (工况 {min_cond})")