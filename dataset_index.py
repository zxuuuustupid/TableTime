import numpy as np
import json

x_train=np.load('data/FingerMovements/X_train.npy',mmap_mode='c')
x_valid=np.load('data/FingerMovements/X_valid.npy',mmap_mode='c')
y_train=np.load('data/FingerMovements/y_train.npy',mmap_mode='c')
y_valid=np.load('data/FingerMovements/y_valid.npy',mmap_mode='c')

train_index=[]
test_index=[]

for i in range(x_train.shape[0]):train_index.append({'index':i,'label':y_train[i]})
for i in range(x_valid.shape[0]):test_index.append({'index':i,'label':y_valid[i]})

with open(r'train_index.json','w') as f:
    json.dump(train_index,f)
with open(r'test_index.json','w') as f:
    json.dump(test_index,f)
