import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import signal as sig

def calculate_dimensionless_features(data):
    """
    计算无量纲指标：这些指标对绝对幅值不敏感，对冲击和形状敏感
    """
    rms = np.sqrt(np.mean(data**2)) + 1e-9
    abs_mean = np.mean(np.abs(data)) + 1e-9
    peak = np.max(np.abs(data))
    
    # 峭度 (Kurtosis): 反映冲击特征
    kurt = (np.mean((data - np.mean(data))**4) / (np.var(data)**2))
    # 峰值因子 (Crest Factor): 冲击与能量之比
    crest = peak / rms
    # 波形因子 (Shape Factor): 能量与均值之比
    shape = rms / abs_mean
    
    return kurt, crest, shape

def visualize_normalized_scheme(base_dir, channels_names, fs=64000, f_max=600):
    """
    方案1：频谱归一化可视化
    目的：通过消除能量量级差异，观察不同工况下的“频率指纹”是否一致
    """
    conditions = [
        "WC1: 20Hz/L0", "WC2: 20Hz/L+", "WC3: 20Hz/L-",
        "WC4: 40Hz/L0", "WC5: 40Hz/L+", "WC6: 40Hz/L-",
        "WC7: 60Hz/L0", "WC8: 60Hz/L+", "WC9: 60Hz/L-"
    ]
    
    fig, axes = plt.subplots(nrows=9, ncols=6, figsize=(22, 20), sharex=True)
    plt.subplots_adjust(hspace=0.6, wspace=0.3)
    
    if not os.path.exists("assets"):
        os.makedirs("assets")

    for row in range(9):
        wc_idx = row + 1
        wc_path = os.path.join(base_dir, f'WC{wc_idx}', 'X_train.npy')
        
        if not os.path.exists(wc_path):
            continue
            
        data = np.load(wc_path, mmap_mode='r')
        sample = data[0] 
        
        for col in range(6):
            ax = axes[row, col]
            sig_data = sample[col, :]
            
            # --- 核心步骤 1: 计算 PSD ---
            freqs, psd = sig.welch(sig_data, fs, nperseg=4096)
            
            # --- 核心步骤 2: 面积归一化 (去量纲化) ---
            # 使 PSD 曲线下方的面积为 1。这样 WC1 到 WC9 的纵坐标就在同一个量级了
            psd_norm = psd / np.sum(psd)
            
            # --- 核心步骤 3: 提取无量纲特征 ---
            k, c, s = calculate_dimensionless_features(sig_data)
            
            # 绘图：现在纵坐标反映的是“能量分布占比”
            ax.plot(freqs, psd_norm, color='#2E7D32', linewidth=0.8)
            ax.set_xlim(0, f_max)
            
            # 统一设置纵坐标范围，便于直观对比形状
            ax.set_ylim(0, np.percentile(psd_norm, 99.9) * 1.5) 
            
            # 标注无量纲特征
            ax.text(0.95, 0.95, f'Kurt:{k:.1f}\nCrest:{c:.1f}\nShape:{s:.1f}', 
                    transform=ax.transAxes, verticalalignment='top', horizontalalignment='right',
                    fontsize=7, bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))

            if row == 0:
                ax.set_title(channels_names[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(conditions[row], fontsize=9, rotation=0, labelpad=70, va='center')
            
            ax.grid(True, linestyle=':', alpha=0.6)

    plt.suptitle("Scheme 1: Normalized Spectral Density & Dimensionless Indicators\n(Removing Energy Scale Difference to Focus on Frequency Patterns)", 
                 fontsize=20, y=0.97)
    plt.xlabel("Frequency (Hz)", fontsize=14)
    
    output_fig = os.path.join("assets", "normalized_scheme_1.png")
    plt.savefig(output_fig, dpi=200, bbox_inches='tight')
    print(f"方案1对比图已保存至: {output_fig}")
    plt.show()

if __name__ == "__main__":
    DATA_BASE = "data/BJTU-gearbox"
    CHANNELS = ["CH11", "CH12", "CH13", "CH14", "CH15", "CH16"]
    visualize_normalized_scheme(DATA_BASE, CHANNELS, f_max=600)