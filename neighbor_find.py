import numpy as np
import json
from dtaidistance import dtw_ndim
from tqdm import tqdm

def standardize(X):
    means = np.mean(X, axis=1, keepdims=True)
    stds = np.std(X, axis=1, keepdims=True)
    Z = (X - means) / stds
    return Z

def standard_ED(X,Y):
    X_standard=standardize(X)
    Y_standard=standardize(Y)
    return np.linalg.norm(Y_standard-X_standard)

def find_nearest_neighbors_DTW(train_data, test_data, num_neighbors=2):
    results = []
    for test_index, test_seq in tqdm(enumerate(test_data)):
        distances = [dtw_ndim.distance(test_seq, train_seq) for train_seq in train_data]
        nearest_indices = np.argsort(distances)[:num_neighbors]
        results.append({"test_index": test_index, "neighbors": nearest_indices.tolist()})
    return results

def find_nearest_neighbors_ED(train_data,test_data,num_neighbors):
    results=[]
    for test_index,test_seq in tqdm(enumerate(test_data)):
        distances = [np.linalg.norm(test_seq-train_seq) for train_seq in train_data]
        nearest_indices = np.argsort(distances)[:num_neighbors]
        results.append({"test_index": test_index, "neighbors": nearest_indices.tolist()})
    return results

def find_nearest_neighbors_standard_ED(train_data,test_data,num_neighbors):
    results=[]
    for test_index,test_seq in tqdm(enumerate(test_data)):
        distances = [standard_ED(test_seq,train_seq) for train_seq in train_data]
        nearest_indices = np.argsort(distances)[:num_neighbors]
        results.append({"test_index": test_index, "neighbors": nearest_indices.tolist()})
    return results

def find_nearest_neighbors_MAN(train_data,test_data,num_neighbors):
    result=[]
    for test_index,test_seq in tqdm(enumerate(test_data)):
        distances = [np.sum(np.abs(test_seq-train_seq)) for train_seq in train_data]
        nearest_indices = np.argsort(distances)[:num_neighbors]
        result.append({"test_index": test_index, "neighbors": nearest_indices.tolist()})
    return result

# dataset='RacketSports'
dataset = 'FingerMovements'
train_data=np.load(f'data/{dataset}/X_train.npy', mmap_mode='c')
test_data=np.load(f'data/{dataset}/X_valid.npy', mmap_mode='c')

dist={'DTW':find_nearest_neighbors_DTW,'ED':find_nearest_neighbors_ED,
      'SED':find_nearest_neighbors_standard_ED,'MAN':find_nearest_neighbors_MAN}

for i in ['DTW','ED','SED','MAN']:
    for j in [1,2,3,4,5,6,7,8,9,10]:
        result=dist[i](train_data,test_data,num_neighbors=j)
        with open(f'data_index/{dataset}/{i}_dist/nearest_{j}_neighbors.json', 'w') as f:
            json.dump(result,f,indent=4)
