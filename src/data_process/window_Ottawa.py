import pandas as pd
import numpy as np
import os
from tqdm import tqdm

class MultiClassDataGenerator:
    def __init__(self, config):
        self.cfg = config
        
        # 从配置字典中加载参数
        win_cfg = self.cfg['window']
        self.window_size = win_cfg['size']
        self.step = int(self.window_size * (1 - win_cfg['overlap_rate']))
        
        split_cfg = self.cfg['split']
        self.train_per_class = split_cfg['train_per_class']
        self.valid_per_class = split_cfg['valid_per_class']
        self.total_per_file = self.train_per_class + self.valid_per_class
        self.component = self.cfg['component']  # 固定为 gearbox 组件

    def _extract_sequential_segments(self, csv_path):
        """从单个CSV文件中严格按时间顺序提取窗口"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"数据文件未找到: {csv_path}")
            
        df = pd.read_csv(csv_path)
        # df = df.iloc[:, :1]
        data = df.values
        num_rows = data.shape[0]
        
        windows = []
        for i in range(10000, num_rows - self.window_size + 1, self.step):
            window = data[i : i + self.window_size, :].T
            windows.append(window)
            if len(windows) >= self.total_per_file:
                break
        
        if len(windows) < self.total_per_file:
            raise ValueError(f"文件 {os.path.basename(csv_path)} 样本不足, 需要 {self.total_per_file}, 实际生成 {len(windows)}")
            
        return np.array(windows)
    # def process(self):
    #         """主处理函数"""
    #         all_train_x, all_valid_x = [], []
    #         all_train_y, all_valid_y = [], []
            
    #         base_path = self.cfg['base_path']
    #         fault_types = self.cfg['fault_types']
    #         samples = self.cfg['samples']
            
    #         pbar = tqdm(total=len(fault_types) * len(samples), desc="Processing files")
            
    #         for label_idx, fault_type in enumerate(fault_types):
    #             for sample_id in samples:
    #                 # --- 修改部分：动态搜索以 data_gearbox 开头的文件 ---
    #                 dir_path = os.path.join(base_path, fault_type, f"Sample_{sample_id}")
                    
    #                 try:
    #                     if not os.path.exists(dir_path):
    #                         raise FileNotFoundError(f"目录不存在: {dir_path}")
                        
    #                     # 自动寻找文件夹下符合条件的文件
    #                     matched_files = [f for f in os.listdir(dir_path) 
    #                                 if f.startswith(f"data_{self.component}") and f.endswith(".csv")]
                        
    #                     if not matched_files:
    #                         raise FileNotFoundError(f"未找到 {self.component} 相关文件")
                        
    #                     # 取匹配到的第一个文件
    #                     csv_path = os.path.join(dir_path, matched_files[0])
    #                     print(f"\n处理文件: {csv_path}")
    #                     # ----------------------------------------------
                        
    #                     all_x = self._extract_sequential_segments(csv_path)
                        
    #                     train_x = all_x[:self.train_per_class]
    #                     valid_x = all_x[self.train_per_class:]
                        
    #                     all_train_x.append(train_x)
    #                     all_valid_x.append(valid_x)
                        
    #                     label_str = f"G{label_idx}"
    #                     all_train_y.extend([label_str] * self.train_per_class)
    #                     all_valid_y.extend([label_str] * self.valid_per_class)
                        
    #                 except (FileNotFoundError, ValueError) as e:
    #                     print(f"\n[警告] 跳过文件处理: {e}")
                    
    #                 pbar.update(1)
            
    #         pbar.close()

    #         if not all_train_x:
    #             print("\n[错误] 没有成功处理任何文件，请检查路径和配置。")
    #             return
                
    #         X_train = np.concatenate(all_train_x, axis=0)
    #         y_train = np.array(all_train_y)
    #         X_valid = np.concatenate(all_valid_x, axis=0)
    #         y_valid = np.array(all_valid_y)

    #         # 保存数据
    #         output_dir = self.cfg['output_dir']
    #         os.makedirs(output_dir, exist_ok=True)
            
    #         np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    #         np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    #         np.save(os.path.join(output_dir, 'X_valid.npy'), X_valid)
    #         np.save(os.path.join(output_dir, 'y_valid.npy'), y_valid)

    #         print(f"\n--- 数据处理完成: {os.path.basename(output_dir)} ---")
    #         print(f"X_train 形状: {X_train.shape} | X_valid 形状: {X_valid.shape}")

    def process(self):
        """主处理函数"""
        all_train_x, all_valid_x = [], []
        all_train_y, all_valid_y = [], []
        
        base_path = self.cfg['base_path']
        fault_types = self.cfg['fault_types']
        samples = self.cfg['samples'] # 这里 sample 代表 A, B, C, D
        
        pbar = tqdm(total=len(fault_types) * len(samples), desc="Processing files")
        
        for label_idx, fault_type in enumerate(fault_types):
            for sample_id in samples:
                # === 修改开始：直接构建 A-BF.csv 这种文件名 ===
                # 文件名格式: {工况}-{故障}.csv，例如 A-BF.csv
                file_name = f"{sample_id}-{fault_type}.csv"
                csv_path = os.path.join(base_path, file_name)
                
                try:
                    # 检查文件是否存在
                    if not os.path.exists(csv_path):
                        raise FileNotFoundError(f"文件不存在: {csv_path}")
                    
                    print(f"\n处理文件: {csv_path}")
                    # ============================================
                    
                    all_x = self._extract_sequential_segments(csv_path)
                    
                    train_x = all_x[:self.train_per_class]
                    valid_x = all_x[self.train_per_class:]
                    
                    all_train_x.append(train_x)
                    all_valid_x.append(valid_x)
                    
                    # === 修改标签：直接使用故障名称 (H, IF, OF...) ===
                    label_str = fault_type 
                    # ============================================
                    
                    all_train_y.extend([label_str] * self.train_per_class)
                    all_valid_y.extend([label_str] * self.valid_per_class)
                    
                except (FileNotFoundError, ValueError) as e:
                    print(f"\n[警告] 跳过文件处理: {e}")
                
                pbar.update(1)
        
        pbar.close()

        # ... (后续保存代码保持不变) ...
        if not all_train_x:
            print("\n[错误] 没有成功处理任何文件，请检查路径和配置。")
            return
            
        X_train = np.concatenate(all_train_x, axis=0)
        y_train = np.array(all_train_y)
        X_valid = np.concatenate(all_valid_x, axis=0)
        y_valid = np.array(all_valid_y)

        # 保存数据
        output_dir = self.cfg['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'X_valid.npy'), X_valid)
        np.save(os.path.join(output_dir, 'y_valid.npy'), y_valid)

        print(f"\n--- 数据处理完成: {os.path.basename(output_dir)} ---")
        print(f"X_train 形状: {X_train.shape} | X_valid 形状: {X_valid.shape}")

# if __name__ == "__main__":
#     # 基础配置，删掉或忽略速度和负载的硬编码
#     CONFIG = {
#         "base_path": "F:/Project/TripletLoss/BJTU-RAO Bogie Datasets/Data/BJTU_RAO_Bogie_Datasets/",
#         "fault_types": [f"M0_G{i}_LA0_RA0" for i in [0,1,2,3]], # 快速生成G0-G8
#         # "fault_types": [f"M{i}_G0_LA0_RA0" for i in [0,1,2,3]], # 快速生成G0-G8
#         # "fault_types": [f"M0_G0_LA{i}_RA0" for i in [0,1,2,4]], # 快速生成G0-G8
#         "window": {"size": 2048, "overlap_rate": 0.0},
#         "split": {"train_per_class": 60, "valid_per_class": 60},
#         "samples": [1],  # 初始值，后续循环中会修改
#         "component": "leftaxlebox",  # 固定为 gearbox 组件
#     }
    

#     # 循环处理每个 Sample，并对应到 WC 目录
#     for s_id in range(1, 10): # 假设处理 Sample 1 到 9
#         CONFIG['samples'] = [s_id] # 每次只处理当前这一个 Sample 文件夹
#         CONFIG['output_dir'] = f"data/BJTU-{CONFIG['component']}/WC{s_id}" # 动态修改输出路径
        
#         print(f"\n开始生成工况 WC{s_id} 的数据...")
#         generator = MultiClassDataGenerator(CONFIG)
#         generator.process()

if __name__ == "__main__":
    # 基础配置
    CONFIG = {
        # 1. 修改为新数据的路径
        "base_path": r"F:\Project\CZSL\code\Disentangling-before-Composing\加拿大轴承数据集\data",
        
        # 2. 修改故障类型，对应文件名后缀 (H, IF, OF, BF, CF)
        "fault_types": ["H", "IF", "OF", "BF", "CF"], 
        
        "window": {"size": 2048, "overlap_rate": 0.0},
        "split": {"train_per_class": 60, "valid_per_class": 60},
        "samples": [], # 循环中赋值
        "component": "Ottawa", # 用于输出文件夹命名
    }
    
    # 3. 定义工况映射：A->WC1, B->WC2, ...
    conditions_map = {
        'A': 'WC1',
        'B': 'WC2',
        'C': 'WC3',
        'D': 'WC4'
    }

    # 循环处理每个工况
    for condition_code, wc_name in conditions_map.items():
        CONFIG['samples'] = [condition_code] # 例如 ['A']
        
        # 输出路径例如: data/Ottawa/WC1
        CONFIG['output_dir'] = f"data/{CONFIG['component']}/{wc_name}" 
        
        print(f"\n开始生成工况 {wc_name} (原工况 {condition_code}) 的数据...")
        generator = MultiClassDataGenerator(CONFIG)
        generator.process()