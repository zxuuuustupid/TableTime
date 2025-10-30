import numpy as np
import pprint

# path = "data/FingerMovements/X_train.npy"
path = "data/FingerMovements/y_valid.npy"
data = np.load(path, allow_pickle=True)

print("📁 文件路径:", path)
print("📐 数据类型:", type(data))
print("📏 数据形状:", getattr(data, 'shape', '无 shape 属性'))
print("🔢 数据类型 (dtype):", getattr(data, 'dtype', '无 dtype 属性'))

# 如果是数组，打印前几个元素
if isinstance(data, np.ndarray):
    print("📊 数据前5个元素:\n", data[:5])
else:
    print("📊 数据内容:\n", data)
