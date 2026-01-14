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
    results_path = os.path.join("data_index", dataset, f"test_WC{test_num}_train_WCs{train_tag}", 
                                f"{list(dist_map.keys())[0]}_dist", f'nearest_{neighbor_num}_neighbors.json')
    
    # 4. 合并训练集标签 (核心改动：因为合并后的训练集索引是连续的，需要手动合并字典)
    merged_train_labels = {}
    current_offset = 0
    for wc in train_nums:
        path = os.path.join("data", "index", dataset, f"WC{wc}", "train_index.json")
        with open(path, 'r') as f:
            labels = json.load(f)
            for item in labels:
                # 将该工况的标签存入合并字典，键为全局偏移后的索引
                merged_train_labels[current_offset] = item['label']
                current_offset += 1
    
    # 5. 加载测试集标签
    test_labels_path = os.path.join("data", "index", dataset, f"WC{test_num}", "test_index.json")
    
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
# #     pipeline()

# if __name__ == "__main__":
#     import datetime # 确保导入 datetime
    
#     # dataset = 'BJTU-gearbox'
#     # dataset = 'BJTU-motor'
#     dataset = 'BJTU-leftaxlebox'
#     dist_map_name = 'FIW'
#     dist_map = {dist_map_name: find_nearest_neighbors_weighted_feature}
#     neighbor_num = 15
#     all_wcs = [1, 2, 3, 4, 5, 6, 7, 8, 9]

#     all_train_scenarios = [
#         [1],
#         [1,2],
#         [1,2,3],
#         [1,2,3,4],
#         [1,2,3,4,5],
#         [1,2,3,4,5,6,],
#         [1,2,3,4,5,6,7],
#         [1,2,3,4,5,6,7,8],
#         # [1,2,3,4,5,7,8]
#     ]
    
#     # 用于收集所有实验结果的列表
#     experiment_logs = []
    
#     # --- 开始大循环 ---
#     for train_nums in all_train_scenarios:
#         test_wcs = [wc for wc in all_wcs if wc not in train_nums]
#         # test_wcs = [3,4,5,6,7,8,9]
        
#         print(f"\n{'='*60}")
#         print(f"🚀 大实验启动：训练集组合 = {train_nums}")
#         print(f"{'='*60}")
        
#         scenario_accuracies = []
        
#         for test_wc in test_wcs:
#             print(f"\n>>> [当前配置] 训练: {train_nums} | 测试: WC{test_wc}")
#             # 获取准确率
#             acc = pipeline(dataset, train_nums, test_wc, dist_map, neighbor_num)
            
#             # 记录单次结果
#             log_str = f"Train: {train_nums} | Test: WC{test_wc} | Accuracy: {acc:.2f}%"
#             experiment_logs.append(log_str)
#             scenario_accuracies.append(acc)
        
#         # 记录该场景的平均准确率
#         avg_acc = np.mean(scenario_accuracies) if scenario_accuracies else 0
#         experiment_logs.append(f"--- Scenario Average (Train {train_nums}): {avg_acc:.2f}% ---\n")

#     # --- 实验结束，保存汇总结果 ---
    
#     # 1. 生成文件名
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     log_dir = "result/log"
#     os.makedirs(log_dir, exist_ok=True)
#     filename = f"{dataset}_{dist_map_name}_{timestamp}.txt"
#     filepath = os.path.join(log_dir, filename)
    
#     # 2. 构建完整报告内容
#     final_report = []
#     final_report.append("="*60)
#     final_report.append(f"实验汇总报告")
#     final_report.append(f"时间: {timestamp}")
#     final_report.append(f"数据集: {dataset}")
#     final_report.append(f"距离度量: {dist_map_name}")
#     final_report.append(f"邻居数: {neighbor_num}")
#     final_report.append("="*60 + "\n")
#     final_report.extend(experiment_logs)
    
#     final_report_str = "\n".join(final_report)
    
#     # 3. 打印并保存
#     print("\n" + "#"*60)
#     print("实验全部完成！汇总结果如下：")
#     print("#"*60)
#     print(final_report_str)
    
#     with open(filepath, 'w', encoding='utf-8') as f:
#         f.write(final_report_str)
        
#     print(f"\n[INFO] 汇总日志已保存至: {filepath}")


if __name__ == "__main__":
    import datetime 
    
    # 1. 修改数据集名称 (必须与 DataGenerator 生成的文件夹名一致)
    dataset = 'Ottawa' 
    
    dist_map_name = 'FIW'
    dist_map = {dist_map_name: find_nearest_neighbors_weighted_feature}
    neighbor_num = 15
    
    # 2. 修改所有工况列表 (Ottawa 只有 A,B,C,D -> WC1, WC2, WC3, WC4)
    all_wcs = [1, 2, 3, 4]
    
    # 3. 修改训练场景组合 (注意数字不能超过 4)
    all_train_scenarios = [
        [1],          # 单工况训练 (用A测B,C,D)
        [1, 2],       # 双工况训练 (用A,B测C,D)
        [1, 3, 2],    # 三工况训练 (用A,B,C测D)
        # 也可以做反向泛化，例如用 D 测 A
        # [4] 
    ]
    
    # 用于收集所有实验结果的列表
    experiment_logs = []
    
    # --- 开始大循环 ---
    for train_nums in all_train_scenarios:
        # 自动计算测试集：在 all_wcs 里，但不在训练集里的
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

    # --- 实验结束，保存汇总结果 (代码保持不变) ---
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = "result/log"
    os.makedirs(log_dir, exist_ok=True)
    filename = f"{dataset}_{dist_map_name}_{timestamp}.txt"
    filepath = os.path.join(log_dir, filename)
    
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
    
    print("\n" + "#"*60)
    print("实验全部完成！汇总结果如下：")
    print("#"*60)
    print(final_report_str)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(final_report_str)
        
    print(f"\n[INFO] 汇总日志已保存至: {filepath}")