import torch
import pandas as pd
import numpy as np
from kmeans_pytorch import kmeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
from tqdm import tqdm

def pairwise_distance(data, centroid):
    dis = (data-centroid)**2
    return dis.sum().squeeze()

start = 5
stop = 30
min_dist_list = []
np.random.seed(42)
X = pd.read_csv('national_anthem_scrape/national_anthem_dataset/features.csv')
x = X.select_dtypes('number').values
x = torch.from_numpy(x)
for num_clusters in tqdm(range(start, stop)):
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

kn = KneeLocator(np.arange(start, stop), min_dist_list, curve = 'convex', direction = 'decreasing')

plt.plot(np.arange(start, stop), min_dist_list)
plt.xticks(np.arange(start, stop,1))
plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
plt.xlabel('number of clusters k')
plt.ylabel('Sum of squared distances')
plt.grid()
plt.show()
plt.savefig('./k_means_clustering/artifacts/distance_metric.png', bbox_inches='tight')