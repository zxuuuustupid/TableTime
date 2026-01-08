import numpy as np
import json
import os

# dataset='FingerMovements'
# x_train=np.load(f'data/{dataset}/X_train.npy',mmap_mode='c')
# x_valid=np.load(f'data/{dataset}/X_valid.npy',mmap_mode='c')
# y_train=np.load(f'data/{dataset}/y_train.npy',mmap_mode='c')
# y_valid=np.load(f'data/{dataset}/y_valid.npy',mmap_mode='c')
# train_index=[]
# test_index=[]

# for i in range(x_train.shape[0]):train_index.append({'index':i,'label':y_train[i]})
# for i in range(x_valid.shape[0]):test_index.append({'index':i,'label':y_valid[i]})

# os.makedirs(f'data/index/{dataset}', exist_ok=True)
# with open(f'data/index/{dataset}/train_index.json','w') as f:
#     json.dump(train_index,f)
# with open(f'data/index/{dataset}/test_index.json','w') as f:
#     json.dump(test_index,f)

def generate_json(dataset):
    # dataset='FingerMovements'
    x_train=np.load(f'data/{dataset}/X_train.npy',mmap_mode='c')
    x_valid=np.load(f'data/{dataset}/X_valid.npy',mmap_mode='c')
    y_train=np.load(f'data/{dataset}/y_train.npy',mmap_mode='c')
    y_valid=np.load(f'data/{dataset}/y_valid.npy',mmap_mode='c')
    train_index=[]
    test_index=[]

    for i in range(x_train.shape[0]):train_index.append({'index':i,'label':y_train[i]})
    for i in range(x_valid.shape[0]):test_index.append({'index':i,'label':y_valid[i]})

    os.makedirs(f'data/index/{dataset}', exist_ok=True)
    with open(f'data/index/{dataset}/train_index.json','w') as f:
        json.dump(train_index,f)
    with open(f'data/index/{dataset}/test_index.json','w') as f:
        json.dump(test_index,f)


    