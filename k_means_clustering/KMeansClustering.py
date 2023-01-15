import torch
import pandas as pd
import numpy as np
from kmeans_pytorch import kmeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
from tqdm import tqdm

class KMeansCluster():
    def __init__(self, audio_features_path):
        self.audio_features_path = audio_features_path
        self.num_clusters = None

    def pairwise_distance(self, data, centroid):
        dis = (data-centroid)**2
        return dis.sum().squeeze()

    def optimal_cluster_search(self, save_path, start = 5, stop = 30, seed = 42):
        start = 5
        stop = 30
        min_dist_list = []
        np.random.seed(seed)
        X = pd.read_csv(self.audio_features_path)
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
                dist = self.pairwise_distance(data, centroid).item()
                dist_list.append(dist)
            min_dist_list.append(sum(dist_list))

        kn = KneeLocator(np.arange(start, stop), min_dist_list, curve = 'convex', direction = 'decreasing')
        self.num_clusters = kn.knee
        plt.plot(np.arange(start, stop), min_dist_list)
        plt.xticks(np.arange(start, stop,1))
        plt.vlines(kn.knee, plt.ylim()[0], plt.ylim()[1], linestyles='dashed')
        plt.xlabel('Number of clusters k')
        plt.ylabel('Sum of squared distances')
        plt.grid()
        plt.savefig(save_path, bbox_inches='tight')
    
