import os
import shutil
import glob

def migrate_bjtu_data():
    # 配置路径
    src_base = r"F:\Project\TripletLoss\BJTU-RAO Bogie Datasets\Data\BJTU_RAO_Bogie_Datasets"
    dst_base = r"F:\Project\TableGPT\TableTime\cross_machine_test\raw_data\BJTU-leftaxlebox"

    # 映射关系：LA编号 -> 故障简称
    fault_map = {
        "0": "H",
        "1": "IF",
        "2": "OF",
        "3": "BF"
    }

    # 处理工况 1 到 9
    for sample_id in range(1, 10):
        wc_name = f"WC{sample_id}"
        sample_folder = f"Sample_{sample_id}"
        
        # 处理四种故障状态
        for la_id, fault_name in fault_map.items():
            # 构建源文件夹路径
            # 模式: M0_G0_LA{id}_RA0 \ Sample_{id}
            sub_dir = f"M0_G0_LA{la_id}_RA0"
            current_src_dir = os.path.join(src_base, sub_dir, sample_folder)

            if not os.path.exists(current_src_dir):
                print(f"跳过: 找不到目录 {current_src_dir}")
                continue

            # 构建匹配文件名的模式 (处理最后变动的部分)
            # data_leftaxlebox_M0_G0_LA{id}_RA0_*.csv
            file_pattern = f"data_leftaxlebox_M0_G0_LA{la_id}_RA0_*.csv"
            search_path = os.path.join(current_src_dir, file_pattern)
            
            # 查找文件
            matching_files = glob.glob(search_path)
            
            if matching_files:
                src_file = matching_files[0] # 取找到的第一个匹配文件
                
                # 构建目标路径
                # 目标格式: ...\WC{id}\{fault_name}.csv
                target_dir = os.path.join(dst_base, wc_name)
                os.makedirs(target_dir, exist_ok=True)
                dst_file = os.path.join(target_dir, f"{fault_name}.csv")

                # 执行迁移 (复制文件)
                try:
                    shutil.copy2(src_file, dst_file)
                    print(f"成功: {wc_name} | {fault_name} <- {os.path.basename(src_file)}")
                except Exception as e:
                    print(f"失败: 迁移 {src_file} 出错: {e}")
            else:
                print(f"警告: 在 {current_src_dir} 下没找到匹配 {file_pattern} 的文件")

if __name__ == "__main__":
    migrate_bjtu_data()
    print("\n所有任务处理完毕。")