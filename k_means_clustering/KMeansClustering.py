import torch
import pandas as pd
import numpy as np
from kmeans_pytorch import kmeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
import plotly.express as px
from tqdm import tqdm

class KMeansCluster():
    def __init__(self, features_path, key_file_path, num_clusters = None):
        self.features_path = features_path
        self.key_file_path = key_file_path
        self.num_clusters = num_clusters

    def pairwise_distance(self, data, centroid):
        dis = (data-centroid)**2
        return dis.sum().squeeze()

    def optimal_cluster_search(self, save_path, start = 5, stop = 30, seed = 42):
        if self.num_clusters != None:
            return
        start = 5
        stop = 30
        min_dist_list = []
        np.random.seed(seed)
        X = pd.read_csv(self.features_path)
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

    def visualise_cluster(self, save_path):
        df_world = px.data.gapminder().query("year == 2007")
        # K MEANS CLUSTERING
        num_clusters = self.num_clusters
        features = pd.read_csv(self.features_path)
        x = features.select_dtypes('number').values
        x = torch.from_numpy(x)
        cluster_ids_x, cluster_centers = kmeans(
            X=x,
            num_clusters=num_clusters,
            distance='euclidean',
            device=torch.device('cuda:0'),
            tqdm_flag = False
        )
        features['cluster_assigned'] = cluster_ids_x.numpy()

        # KEY FILE
        mapping_df = pd.read_csv(self.key_file_path)
        mapping_with_clusters = pd.merge(mapping_df, features, left_on = "Audio_File", right_on="file_name")[["Country", "cluster_assigned"]]
        mapping_with_clusters['cluster_assigned'] = mapping_with_clusters['cluster_assigned'].astype('int')

        # FINAL DF
        df = df_world.merge(mapping_with_clusters, how = 'left', left_on = ["country"], right_on = ["Country"])
        df['cluster_assigned'] = df['cluster_assigned'].astype('Int64')
        df.drop_duplicates(subset = "country", inplace = True)

        fig = px.choropleth(
            df,
            locations = "iso_alpha",
            color="cluster_assigned",
            hover_name="country",
            color_continuous_scale=px.colors.sequential.Plasma
        )
        
        fig.write_image(save_path, format = "png")
        fig.show()