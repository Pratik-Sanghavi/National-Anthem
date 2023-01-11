import torch
import pandas as pd
import numpy as np
from kmeans_pytorch import kmeans
from tqdm import tqdm

def pairwise_distance(data, centroid):
    dis = (data-centroid)**2
    return dis.sum().squeeze()

start = 5
stop = 20
min_dist = np.inf
opt_clusters = -1
for num_clusters in tqdm(range(start, stop)):
    print(f'Number of Clusters: {num_clusters}')
    print('#'*80)
    X = pd.read_csv('national_anthem_scrape/national_anthem_dataset/features.csv')
    x = X.select_dtypes('number').values
    x = torch.from_numpy(x)

    # kmeans
    cluster_ids_x, cluster_centers = kmeans(
        X=x, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0')
    )
    dist_list = []
    for i in range(len(x)):
        cluster_val = cluster_ids_x[i]
        data = x[i]
        centroid = cluster_centers[cluster_val]
        dist = pairwise_distance(data, centroid).item()
        dist_list.append(dist)
    print(f'Distance: {sum(dist_list)}')
    if sum(dist_list)<min_dist:
        min_dist = sum(dist_list)
        opt_clusters = num_clusters

print(opt_clusters)