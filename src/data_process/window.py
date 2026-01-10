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

    def _extract_sequential_segments(self, csv_path):
        """从单个CSV文件中严格按时间顺序提取窗口"""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"数据文件未找到: {csv_path}")
            
        df = pd.read_csv(csv_path)
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

    def process(self):
        """主处理函数"""
        all_train_x, all_valid_x = [], []
        all_train_y, all_valid_y = [], []
        
        base_path = self.cfg['base_path']
        fault_types = self.cfg['fault_types']
        samples = self.cfg['samples']
        
        pbar = tqdm(total=len(fault_types) * len(samples), desc="Processing files")
        
        for label_idx, fault_type in enumerate(fault_types):
            for sample_id in samples:
                filename = self.cfg['filename_template'].format(
                    fault_type=fault_type,
                    speed=self.cfg['speed'],
                    load=self.cfg['load']
                )
                csv_path = os.path.join(base_path, fault_type, f"Sample_{sample_id}", filename)
                
                try:
                    all_x = self._extract_sequential_segments(csv_path)
                    
                    train_x = all_x[:self.train_per_class]
                    valid_x = all_x[self.train_per_class:]
                    
                    all_train_x.append(train_x)
                    all_valid_x.append(valid_x)
                    
                    # ============ [关键修改] ============
                    # 在这里生成 'G' + 数字 的字符串标签
                    label_str = f"G{label_idx}"
                    all_train_y.extend([label_str] * self.train_per_class)
                    all_valid_y.extend([label_str] * self.valid_per_class)
                    # ====================================
                    
                except (FileNotFoundError, ValueError) as e:
                    print(f"\n[警告] 跳过文件处理: {e}")
                
                pbar.update(1)
        
        pbar.close()

        if not all_train_x:
            print("\n[错误] 没有成功处理任何文件，请检查路径和配置。")
            return
            
        X_train = np.concatenate(all_train_x, axis=0)
        # 将列表转为 NumPy 数组，dtype 会自动设为字符串类型
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

        print("\n--- 数据处理完成 ---")
        print(f"输出目录: {output_dir}")
        print(f"总类别数: {len(fault_types)}")
        print(f"X_train 形状: {X_train.shape}")
        print(f"y_train 形状: {y_train.shape} (标签示例: {y_train[0]}, {y_train[self.train_per_class]})")
        print(f"X_valid 形状: {X_valid.shape}")
        print(f"y_valid 形状: {y_valid.shape}")

if __name__ == "__main__":
    # ================================================================
    # [配置区] 请在这里修改所有参数
    # ================================================================
    CONFIG = {
        # 1. 基础路径设置
        "base_path": "F:/Project/TripletLoss/BJTU-RAO Bogie Datasets/Data/BJTU_RAO_Bogie_Datasets/",

        # 2. 故障类型定义 (G0 必须是第一个，代表健康)
        #    格式: "M0_G{i}_LA0_RA0"
        "fault_types": [
            "M0_G0_LA0_RA0",  # 标签 0 (Health)
            "M0_G1_LA0_RA0",  # 标签 1
            "M0_G2_LA0_RA0",  # 标签 2
            "M0_G3_LA0_RA0",  # 标签 3
            "M0_G4_LA0_RA0",  # 标签 4
            "M0_G5_LA0_RA0",  # 标签 5
            "M0_G6_LA0_RA0",  # 标签 6
            "M0_G7_LA0_RA0",  # 标签 7
            "M0_G8_LA0_RA0",  # 标签 8
        ],

        # 3. 工况定义 (文件名中的 Sample_{i})
        #    你可以添加多个，比如 [1, 2, 3]
        "samples": [1],

        # 4. 文件名模板 (自动填充 {fault_type} 和 {sample_id})
        #    - {speed}: 速度，如 20Hz
        #    - {load}: 负载，如 0kN
        "filename_template": "data_gearbox_{fault_type}_{speed}_{load}.csv",
        "speed": "20Hz",
        "load": "0kN",

        # 5. 滑动窗口参数
        "window": {
            "size": 5000,
            "overlap_rate": 0.8
        },

        # 6. 数据集划分参数
        "split": {
            "train_per_class": 60, # 每类故障/健康的训练样本数
            "valid_per_class": 20   # 每类故障/健康的验证样本数
        },

        # 7. 输出目录
        "output_dir": "data/BJTU-gearbox"
    }
    # ================================================================
    
    generator = MultiClassDataGenerator(CONFIG)
    generator.process()