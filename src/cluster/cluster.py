import os
import numpy as np
import pandas as pd
from pathlib import Path

def process_bogie_datasets():
    """
    处理转向架数据集，从原始数据路径读取，输出到程序所在目录的output文件夹
    路径完全独立：原始数据路径、程序路径、输出路径互不影响
    """
    
    # =============== 1. 路径定义（完全独立） ===============
    
    # 原始数据根目录（固定路径，与程序位置无关）
    raw_data_root = r"F:\Project\TripletLoss\BJTU-RAO Bogie Datasets\Data\BJTU_RAO_Bogie_Datasets"
    
    # 获取程序当前所在目录（与原始数据路径完全独立）
    current_dir = Path(__file__).parent.absolute() if '__file__' in globals() else Path.cwd()
    
    # 输出目录：程序所在目录下的output文件夹（与原始数据路径完全独立）
    output_dir = current_dir / "output"
    
    print(f"程序所在目录: {current_dir}")
    print(f"原始数据根目录: {raw_data_root}")
    print(f"输出目录: {output_dir}")
    print("-" * 60)
    
    # =============== 2. 创建输出目录 ===============
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"✅ 已创建/确认输出目录: {output_dir}")
    
    # =============== 3. 参数配置 ===============
    window_size = 5000
    step_size = 5000
    expected_samples_per_file = 50
    num_fault_types = 9  # G0 to G8
    num_conditions = 9   # Sample_1 to Sample_9
    
    # =============== 4. 处理主循环 ===============
    total_processed = 0
    total_failed = 0
    
    for fault_idx in range(num_fault_types):
        fault_type = f"M0_G{fault_idx}_LA0_RA0"
        fault_path = Path(raw_data_root) / fault_type
        
        print(f"\n🔍 处理故障类型 {fault_idx} (文件夹: {fault_type})")
        
        if not fault_path.exists():
            print(f"❌ 故障类型文件夹不存在: {fault_path}")
            total_failed += num_conditions
            continue
        
        for condition_idx in range(1, num_conditions + 1):
            sample_folder = f"Sample_{condition_idx}"
            sample_path = fault_path / sample_folder
            
            # 构建输出文件名
            output_filename = f"G{fault_idx}_WC{condition_idx}.npy"
            output_filepath = output_dir / output_filename
            
            print(f"  📁 处理工况 {condition_idx} (文件夹: {sample_folder}) -> {output_filename}")
            
            try:
                # =============== 5. 查找CSV文件 ===============
                if not sample_path.exists():
                    raise FileNotFoundError(f"工况文件夹不存在: {sample_path}")
                
                # 查找以data_gearbox开头的CSV文件
                csv_files = list(sample_path.glob("data_gearbox*.csv"))
                
                if not csv_files:
                    raise FileNotFoundError(f"在 {sample_path} 中未找到以 'data_gearbox' 开头的CSV文件")
                
                # 选择第一个匹配的CSV文件
                csv_file = csv_files[0]
                print(f"    📄 找到CSV文件: {csv_file.name}")
                
                # =============== 6. 读取和处理数据 ===============
                # 读取CSV，跳过第一行表头
                df = pd.read_csv(csv_file, header=0)
                
                # 验证列数
                if df.shape[1] < 6:
                    raise ValueError(f"CSV文件列数不足，预期6列，实际{df.shape[1]}列")
                
                # 只取前6列
                data = df.iloc[:, :6].values.astype(np.float32)
                print(f"    📊 数据形状: {data.shape} (行×列)")
                
                # =============== 7. 滑窗切割 ===============
                # 计算实际可切分的样本数
                max_samples = (data.shape[0] - window_size) // step_size + 1
                
                if max_samples < expected_samples_per_file:
                    print(f"    ⚠️  警告: 数据长度不足，预期{expected_samples_per_file}个样本，实际最多{max_samples}个")
                    actual_samples = max_samples
                else:
                    actual_samples = expected_samples_per_file
                
                # 创建存储数组
                samples = np.zeros((actual_samples, window_size, 6), dtype=np.float32)
                
                # 执行滑窗
                for i in range(actual_samples):
                    start_idx = i * step_size
                    end_idx = start_idx + window_size
                    samples[i] = data[start_idx:end_idx, :6]
                
                print(f"    ✂️  滑窗切割完成: {actual_samples}个样本，每个样本形状{samples[0].shape}")
                
                # =============== 8. 保存结果 ===============
                np.save(output_filepath, samples)
                print(f"    💾 保存成功: {output_filepath} (形状: {samples.shape})")
                
                total_processed += 1
                
            except Exception as e:
                print(f"    ❌ 处理失败: {str(e)}")
                total_failed += 1
                continue
    
    # =============== 9. 总结报告 ===============
    print("\n" + "="*60)
    print("处理完成总结:")
    print(f"✅ 成功处理: {total_processed} 个文件")
    print(f"❌ 处理失败: {total_failed} 个文件")
    print(f"📁 输出目录: {output_dir}")
    print(f"📝 总共应处理: {num_fault_types * num_conditions} 个文件")
    print("="*60)
    
    if total_failed > 0:
        print("⚠️  建议检查失败的文件路径和数据完整性")
    else:
        print("🎉 所有文件处理成功！")

if __name__ == "__main__":
    """
    程序入口点，确保路径完全独立：
    - 原始数据路径：固定在F盘
    - 程序路径：当前脚本所在位置
    - 输出路径：程序所在目录下的output文件夹
    """
    print("🚀 开始处理转向架数据集...")
    print("💡 路径说明:")
    print("   • 原始数据路径：与程序位置无关的固定路径")
    print("   • 程序路径：当前Python脚本所在目录")
    print("   • 输出路径：程序目录下的output文件夹（完全独立）")
    print()
    
    try:
        process_bogie_datasets()
    except KeyboardInterrupt:
        print("\n⏹️  用户中断程序执行")
    except Exception as e:
        print(f"\n💥 严重错误: {str(e)}")
        print("💡 建议检查:")
        print("   • 原始数据路径是否存在")
        print("   • 是否有文件读写权限")
        print("   • 磁盘空间是否充足")
        
        
        
# 写另一个程序做聚类，需要3个输入。第一个表明对哪种工况下（一个）的数据A做聚类，第二个表明用哪些（多个）工况下的数据B来计算邻居。第三个选择用哪几种故障类型作为聚类的对象。
# 你自己写一个很高明的计算邻居距离的聚类方法，对A的每个数据寻找B中最近的50个邻居（注意A的类型不参与自身的聚类），结果保存到当前程序所在目录下的result文件夹。聚类结束后进行测试：测试需要计算A中所有的故障类型分别的50个样本的聚类准确率，和总共的准确率