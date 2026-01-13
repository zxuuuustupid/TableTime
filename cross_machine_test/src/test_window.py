import pandas as pd
import numpy as np
import os
from tqdm import tqdm

class MultiClassDataGenerator:
    def __init__(self, config):
        self.cfg = config
        win_cfg = self.cfg['window']
        self.window_size = win_cfg['size']
        self.step = int(self.window_size * (1 - win_cfg['overlap_rate']))
        
        split_cfg = self.cfg['split']
        self.train_per_class = split_cfg['train_per_class']
        self.valid_per_class = split_cfg['valid_per_class']
        self.total_per_file = self.train_per_class + self.valid_per_class

    def _extract_sequential_segments(self, csv_path):
        """读取CSV，只取第一列，去掉表头"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"数据文件未找到: {csv_path}")
            
        # Pandas 默认第一行为 Header 并跳过，iloc[:, 0] 取第一列
        df = pd.read_csv(csv_path)
        data = df.iloc[:, 0].values.reshape(-1, 1)
        
        num_rows = data.shape[0]
        windows = []
        
        # 截取窗口并转置为 (1, WindowSize)
        for i in range(10000, num_rows - self.window_size + 1, self.step):
            window = data[i : i + self.window_size, :].T
            windows.append(window)
            if len(windows) >= self.total_per_file:
                break
        
        if len(windows) < self.total_per_file:
            raise ValueError(f"文件样本不足: 需{self.total_per_file}, 实际{len(windows)}")
            
        return np.array(windows)

    def process(self):
        """主处理函数"""
        all_train_x, all_valid_x = [], []
        all_train_y, all_valid_y = [], []
        
        base_path = self.cfg['base_path']
        fault_types = self.cfg['fault_types'] 
        mode = self.cfg['mode']
        condition = self.cfg['condition']

        for label_idx, f_type in enumerate(fault_types):
            try:
                # 路径匹配逻辑
                if mode == 'BJTU-leftaxlebox':
                    # BJTU 路径: cross_machine_test/raw_data/BJTU-leftaxlebox/WC1/H.csv
                    csv_path = os.path.join(base_path, condition, f"{f_type}.csv")
                else: 
                    # Ottawa 路径: cross_machine_test/raw_data/Ottawa/A/A-H.csv
                    csv_path = os.path.join(base_path, condition, f"{condition}-{f_type}.csv")

                all_x = self._extract_sequential_segments(csv_path)
                
                # 分割训练和验证
                train_x = all_x[:self.train_per_class]
                valid_x = all_x[self.train_per_class:]
                
                all_train_x.append(train_x)
                all_valid_x.append(valid_x)
                
                label_str = f"G{label_idx}"
                all_train_y.extend([label_str] * self.train_per_class)
                all_valid_y.extend([label_str] * self.valid_per_class)
                
            except Exception as e:
                print(f"[跳过] {mode}-{condition}-{f_type}: {e}")

        if not all_train_x: return

        # 合并并保存
        X_train = np.concatenate(all_train_x, axis=0)
        y_train = np.array(all_train_y)
        X_valid = np.concatenate(all_valid_x, axis=0)
        y_valid = np.array(all_valid_y)

        # 确保输出目录存在
        output_dir = self.cfg['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'X_valid.npy'), X_valid)
        np.save(os.path.join(output_dir, 'y_valid.npy'), y_valid)
        print(f"成功 -> {mode}-{condition} | X_train: {X_train.shape} | 路径: {output_dir}")

if __name__ == "__main__":
    # 统一的一级目录
    ROOT = "cross_machine_test"
    BJTU_WC_LIST_NUM = [1]
    # 故障对齐列表
    FAULT_TYPES = ['H', 'IF', 'OF', 'BF'] 
    WIN_SET = {"size": 2048, "overlap_rate": 0.0}

    # 1. 处理 BJTU (训练源)
    print("开始处理 BJTU 数据集...")
    for i in BJTU_WC_LIST_NUM:
        config = {
            "mode": "BJTU-leftaxlebox",
            "base_path": os.path.join(ROOT, "raw_data/BJTU-leftaxlebox"),
            "condition": f"WC{i}",
            "fault_types": FAULT_TYPES,
            "window": WIN_SET,
            "split": {"train_per_class": 60, "valid_per_class": 60},
            "output_dir": os.path.join(ROOT, f"data/BJTU_leftaxlebox/WC{i}")
        }
        MultiClassDataGenerator(config).process()

    # 2. 处理 Ottawa (测试目标)
    print("\n开始处理 Ottawa 数据集...")
    for cond in ['A', 'B', 'C', 'D']:
        config = {
            "mode": "Ottawa",
            "base_path": os.path.join(ROOT, "raw_data/Ottawa"),
            "condition": cond,
            "fault_types": FAULT_TYPES,
            "window": WIN_SET,
            "split": {"train_per_class": 60, "valid_per_class": 60},
            "output_dir": os.path.join(ROOT, f"data/Ottawa/{cond}")
        }
        MultiClassDataGenerator(config).process()