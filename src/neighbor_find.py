import os
import numpy as np
import json
from dtaidistance import dtw_ndim
from tqdm import tqdm

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

def neighbor_find(dataset, neighbor_num,
                  dist_map={'DTW': find_nearest_neighbors_DTW},
                  skip_labels=None): # <--- [修改1] 参数变成 skip_labels (列表)
    """
    查找最近邻，并支持跳过一个或多个特定标签的数据。
    
    Args:
        skip_labels (list, optional): 要跳过的标签列表, e.g., ['G3', 'G5']. 默认为 None.
    """
    
    # --- 加载所有数据，包括标签 ---
    print(f"Loading data for dataset: {dataset}")
    full_train_data = np.load(f'data/{dataset}/X_train.npy', mmap_mode='c')
    full_train_labels = np.load(f'data/{dataset}/y_train.npy', mmap_mode='c')
    
    full_test_data = np.load(f'data/{dataset}/X_valid.npy', mmap_mode='c')
    full_test_labels = np.load(f'data/{dataset}/y_valid.npy', mmap_mode='c')
    
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
        output_dir = f'data_index/{dataset}/{name}_dist'
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nCalculating neighbors using {name}...")
        
        for j in range(neighbor_num, neighbor_num + 1):
            print(f"  - Finding {j} nearest neighbors...")
            result = func(train_data, test_data, num_neighbors=j)
            
            output_path = f'{output_dir}/nearest_{j}_neighbors.json'
            with open(output_path, 'w') as f:
                json.dump(result, f, indent=4)
            print(f"    -> Saved results to {output_path}")