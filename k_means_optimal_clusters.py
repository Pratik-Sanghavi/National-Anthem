import torch
import pandas as pd
import numpy as np
from kmeans_pytorch import kmeans
import matplotlib.pyplot as plt
from tqdm import tqdm

def pairwise_distance(data, centroid):
    dis = (data-centroid)**2
    return dis.sum().squeeze()

start = 5
stop = 30
min_dist_list = []
np.random.seed(42)
for num_clusters in tqdm(range(start, stop)):
    X = pd.read_csv('national_anthem_scrape/national_anthem_dataset/features.csv')
    x = X.select_dtypes('number').values
    x = torch.from_numpy(x)

    # kmeans
    cluster_ids_x, cluster_centers = kmeans(
        X=x, num_clusters=num_clusters, distance='euclidean', device=torch.device('cuda:0'), tqdm_flag = False
    )
    dist_list = []
    for i in range(len(x)):
        cluster_val = cluster_ids_x[i]
        data = x[i]
        centroid = cluster_centers[cluster_val]
        dist = pairwise_distance(data, centroid).item()
        dist_list.append(dist)
    min_dist_list.append(sum(dist_list))

plt.plot(np.arange(start, stop), min_dist_list)
plt.xticks(np.arange(start, stop,1))
plt.grid()
plt.savefig('./k_means_clustering/artifacts/distance_metric.png', bbox_inches='tight')