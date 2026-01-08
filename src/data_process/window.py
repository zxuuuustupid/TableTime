import pandas as pd
import numpy as np
import os

class TrainDataGenerator:
    def __init__(self, window_size=100, overlap_rate=0.75):
        self.window_size = window_size
        self.step = int(window_size * (1 - overlap_rate))
        self.train_per_class = 150
        self.valid_per_class = 50

    def _extract_sequential_segments(self, csv_path, total_needed):
        """严格按时间顺序提取窗口"""
        df = pd.read_csv(csv_path)
        data = df.values
        num_rows, num_cols = data.shape
        
        windows = []
        # 这里的循环是按时间轴从上往下走的
        for i in range(0, num_rows - self.window_size + 1, self.step):
            # 提取 100xN 的块并转置为 Nx100，保证时间连续
            window = data[i : i + self.window_size, :].T
            windows.append(window)
            
            if len(windows) >= total_needed:
                break
        
        if len(windows) < total_needed:
            raise ValueError(f"文件 {csv_path} 太短，无法生成 {total_needed} 个样本。")
            
        return np.array(windows), num_cols

    def process_and_save(self, health_path, fault_path, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        count = self.train_per_class + self.valid_per_class # 每个文件取200个

        # 1. 顺序提取
        h_all_x, h_cols = self._extract_sequential_segments(health_path, count)
        f_all_x, f_cols = self._extract_sequential_segments(fault_path, count)

        # 2. 按照顺序切分 (前150训练，后50测试)
        # 这样保证了训练集用的是该设备早期的信号，测试集用的是后期的信号，符合工业预测逻辑
        h_train_x = h_all_x[:self.train_per_class]
        h_valid_x = h_all_x[self.train_per_class:]
        
        f_train_x = f_all_x[:self.train_per_class]
        f_valid_x = f_all_x[self.train_per_class:]

        # 3. 顺序合并
        # X_train 的内容：前150个是health，后150个是fault
        X_train = np.concatenate([h_train_x, f_train_x], axis=0)
        y_train = np.array(["health"] * 150 + ["fault"] * 150)

        # X_valid 的内容：前50个是health，后50个是fault
        X_valid = np.concatenate([h_valid_x, f_valid_x], axis=0)
        y_valid = np.array(["health"] * 50 + ["fault"] * 50)

        # 4. 直接保存，不执行 Shuffle
        np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
        np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
        np.save(os.path.join(output_dir, 'X_valid.npy'), X_valid)
        np.save(os.path.join(output_dir, 'y_valid.npy'), y_valid)

        print(f"数据处理完毕。")
        print(f"X_train 结构: {X_train.shape}")
        print(f"X_valid 结构: {X_valid.shape}")
        print(f"y_train 结构: {y_train.shape}")
        print(f"y_valid 结构: {y_valid.shape}")


health_type = "M0_G0_LA0_RA0"
fault_type = "M0_G1_LA0_RA0"

if __name__ == "__main__":
    # 请替换为你的绝对路径
    health_csv = f"F:\\Project\\TripletLoss\\BJTU-RAO Bogie Datasets\\Data\\BJTU_RAO_Bogie_Datasets\\{health_type}\\Sample_1\\data_gearbox_{health_type}_20Hz_0kN.csv"
    fault_csv = f"F:\\Project\\TripletLoss\\BJTU-RAO Bogie Datasets\\Data\\BJTU_RAO_Bogie_Datasets\\{fault_type}\\Sample_1\\data_gearbox_{fault_type}_20Hz_0kN.csv"
    target_rel_path = "data/BJTU-gearbox"

    processor = TrainDataGenerator(window_size=100, overlap_rate=0.75)
    
    try:
        processor.process_and_save(health_csv, fault_csv, target_rel_path)
    except Exception as e:
        print(f"程序运行出错: {e}")