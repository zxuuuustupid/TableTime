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
        self.component = self.cfg['component']

    def _extract_sequential_segments(self, file_path):
        """从单个文件中严格按时间顺序提取窗口 (适配 Excel)"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"数据文件未找到: {file_path}")
            
        # === 修改点 1: 读取 Excel，选最后一个 sheet，取前3列 ===
        try:
            # engine='openpyxl' 是读取 xlsx 的标准引擎
            df = pd.read_excel(file_path, sheet_name=-1, engine='openpyxl')
            # 确保只取前3列数据 (根据你的描述)
            df = df.iloc[:, :3] 
        except Exception as e:
             raise ValueError(f"读取Excel失败 {file_path}: {e}")
        # ====================================================

        data = df.values
        num_rows = data.shape[0]
        
        windows = []
        # 注意：这里起始点设为 0 或者保留你之前的 10000 都可以，这里我保留你的逻辑
        # 如果数据量不够，可能需要把 10000 改小
        start_index = 0 # 建议改为0，防止Excel数据行数不够，或者保留 10000
        
        for i in range(start_index, num_rows - self.window_size + 1, self.step):
            window = data[i : i + self.window_size, :].T # 转置后形状 (3, 2048)
            windows.append(window)
            if len(windows) >= self.total_per_file:
                break
        
        if len(windows) < self.total_per_file:
            # 只是警告而不是报错，防止某个文件略短导致程序中断
            print(f"[警告] 文件 {os.path.basename(file_path)} 样本不足, 只有 {len(windows)}")
            
        return np.array(windows)

    def process(self):
        """主处理函数"""
        all_train_x, all_valid_x = [], []
        all_train_y, all_valid_y = [], []
        
        base_path = self.cfg['base_path']
        fault_types = self.cfg['fault_types']
        samples = self.cfg['samples'] # 这里 sample 代表 1, 2, 3, 4
        
        pbar = tqdm(total=len(fault_types) * len(samples), desc="Processing files")
        
        # 获取 base_path 下的所有子文件夹名称，用于匹配 (例如 "1-H", "2-IF")
        if os.path.exists(base_path):
            all_subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
        else:
            print(f"[错误] 路径不存在: {base_path}")
            return

        for label_idx, fault_type in enumerate(fault_types):
            
            # === 修改点 2: 自动匹配故障文件夹 ===
            # 比如 fault_type 是 "H"，我们要找到 "1-H" 文件夹
            target_folder_name = next((d for d in all_subdirs if d.endswith(f"-{fault_type}")), None)
            
            if not target_folder_name:
                print(f"[跳过] 未找到故障类型 {fault_type} 对应的文件夹")
                pbar.update(len(samples))
                continue
                
            fault_dir_path = os.path.join(base_path, target_folder_name)
            # ===================================

            for sample_id in samples:
                # === 修改点 3: 自动匹配工况文件 ===
                # 你的截图显示文件名类似 "20201104-1...", 其中 "-1" 代表工况
                try:
                    files_in_dir = os.listdir(fault_dir_path)
                    # 规则：是xlsx文件，且文件名包含 "-1" (如果sample_id是1)
                    # 加上 "-" 是为了防止匹配到 "11" 里的 "1"
                    target_file = next((f for f in files_in_dir if f.endswith('.xlsx') and f"-{sample_id}" in f), None)
                    
                    if not target_file:
                        raise FileNotFoundError(f"在 {target_folder_name} 中未找到包含 -{sample_id} 的xlsx文件")
                    
                    csv_path = os.path.join(fault_dir_path, target_file)
                    print(f"\n处理文件: {csv_path}")
                    
                    all_x = self._extract_sequential_segments(csv_path)
                    
                    # 如果样本不足，all_x 可能为空或长度不够，做个切片保护
                    current_len = len(all_x)
                    if current_len == 0: continue

                    # 分割训练验证
                    train_end = min(self.train_per_class, current_len)
                    train_x = all_x[:train_end]
                    
                    valid_end = min(self.train_per_class + self.valid_per_class, current_len)
                    valid_x = all_x[self.train_per_class:valid_end]
                    
                    # 只有当提取到了数据才添加
                    if len(train_x) > 0:
                        all_train_x.append(train_x)
                        all_train_y.extend([fault_type] * len(train_x))
                    
                    if len(valid_x) > 0:
                        all_valid_x.append(valid_x)
                        all_valid_y.extend([fault_type] * len(valid_x))
                    
                except (FileNotFoundError, ValueError, Exception) as e:
                    print(f"\n[警告] 跳过: {e}")
                
                pbar.update(1)
        
        pbar.close()

        if not all_train_x:
            print("\n[错误] 没有成功处理任何文件，请检查路径和配置。")
            return
            
        X_train = np.concatenate(all_train_x, axis=0)
        y_train = np.array(all_train_y)
        
        # 处理验证集可能为空的情况
        if all_valid_x:
            X_valid = np.concatenate(all_valid_x, axis=0)
            y_valid = np.array(all_valid_y)
        else:
            X_valid = np.array([])
            y_valid = np.array([])

        # 保存数据
        output_dir = self.cfg['output_dir']
        os.makedirs(output_dir, exist_ok=True)
        
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'X_valid.npy'), X_valid)
        np.save(os.path.join(output_dir, 'y_valid.npy'), y_valid)

        print(f"\n--- 数据处理完成: {os.path.basename(output_dir)} ---")
        print(f"X_train 形状: {X_train.shape} | X_valid 形状: {X_valid.shape}")


if __name__ == "__main__":
    # 基础配置
    CONFIG = {
        # 1. 修改为新数据的根目录路径
        "base_path": r"F:\Project\TableGPT\data\轴箱轴承",
        
        # 2. 故障类型对应文件夹后缀 (1-H, 2-IF...)
        "fault_types": ["H", "IF", "OF", "BF", "CF"], 
        
        "window": {"size": 2048, "overlap_rate": 0.0},
        "split": {"train_per_class": 60, "valid_per_class": 60},
        "samples": [], 
        "component": "SWJTU", # 自定义名称
    }
    
    # 3. 定义工况映射：文件名中的 1, 2, 3, 4 对应 WC1...
    # 截图显示文件名里有 -1, -2, -3, -4
    conditions_map = {
        1: 'WC1',
        2: 'WC2',
        3: 'WC3',
        4: 'WC4'
    }

    # 循环处理每个工况
    for condition_code, wc_name in conditions_map.items():
        CONFIG['samples'] = [condition_code] # 传入数字 1, 2...
        
        # 输出路径例如: data/AxleBox/WC1
        CONFIG['output_dir'] = f"data/{CONFIG['component']}/{wc_name}" 
        
        print(f"\n开始生成工况 {wc_name} (文件名包含 -{condition_code}) 的数据...")
        generator = MultiClassDataGenerator(CONFIG)
        generator.process()