from src.dataset_index import generate_json
from src.neighbor_find import *

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
        return
    
    mean_accuracy = np.mean(all_purities) * 100
    
    print("\n" + "="*50)
    print(f"[INFO] 最近邻检索精度评估报告")
    print(f"[INFO] - 检索文件: {os.path.basename(retrieval_results_path)}")
    print("="*50)
    print(f"[INFO] 总计评估的测试样本数: {len(all_purities)}")
    print(f"[INFO] 平均检索精度 (Mean Purity @ k): {mean_accuracy:.2f}%")
    print("="*50)
    print(f"[INFO] (该指标衡量的是：对于一个测试样本，其找到的邻居有多大概率与它自己是同一类别)")




dataset='BJTU-gearbox'
dist_map = {'FIW': find_nearest_neighbors_weighted_feature}
neighbor_num = 15
# weight_DTW=0.1
# weight_feature=0.9

generate_json(dataset=dataset)
neighbor_find(dataset=dataset,
                  dist_map = dist_map,
                  neighbor_num = neighbor_num,
                  skip_labels = None,)
calculate_retrieval_accuracy(os.path.join("data_index", dataset, f"{list(dist_map.keys())[0]}_dist", f'nearest_{neighbor_num}_neighbors.json'),         os.path.join("data", "index",dataset,"test_index.json"), os.path.join("data", "index", dataset, "train_index.json"))