from src.dataset_index import generate_json
from src.neighbor_find import *

dataset='BJTU-gearbox'
generate_json(dataset=dataset)
neighbor_find(dataset=dataset,
                  dist_map = {'DTW': find_nearest_neighbors_DTW},
                  neighbor_num = 15)
