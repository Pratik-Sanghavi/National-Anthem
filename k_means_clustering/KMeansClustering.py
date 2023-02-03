import torch
import pandas as pd
import numpy as np
from kmeans_pytorch import kmeans
from kneed import KneeLocator
import matplotlib.pyplot as plt
import plotly.express as px
from tqdm import tqdm

class KMeansCluster():
    def __init__(self, 
                 audio_features_path, 
                 text_features_path, 
                 key_file_path, 
                 key_audio_file_col,
                 audio_features_file_col, 
                 key_country_col, 
                 text_feat_country_col,
                 num_clusters = None):
        self.audio_features_path = audio_features_path
        self.text_features_path = text_features_path
        self.key_file_path = key_file_path
        self.num_clusters = num_clusters
        self.key_audio_file_col = key_audio_file_col
        self.audio_features_file_col = audio_features_file_col
        self.key_country_col = key_country_col
        self.text_feat_country_col = text_feat_country_col
        self.audio_features = None

    def data_merge(self):
        aud_features = pd.read_csv(self.audio_features_path)
        text_features = pd.read_csv(self.text_features_path)
        audio_cols = aud_features.columns.to_list()
        audio_cols.remove(self.audio_features_file_col)

        key_df = pd.read_csv(self.key_file_path)
        country_aud_df = pd.merge(key_df, 
                                  aud_features, 
                                  left_on = self.key_audio_file_col, 
                                  right_on = self.audio_features_file_col, 
                                  how = 'left')

        cols_of_interest = [y for x in [[self.key_country_col], audio_cols] for y in x]
        country_aud_df = country_aud_df[cols_of_interest].fillna(0)
        self.features_df = pd.merge(country_aud_df, 
                                    text_features, 
                                    left_on = self.key_country_col, 
                                    right_on=self.text_feat_country_col, 
                                    how = 'outer').fillna(0).drop(columns=[self.text_feat_country_col])

    def pairwise_distance(self, data, centroid):
        dis = (data-centroid)**2
        return dis.sum().squeeze()

    def optimal_cluster_search(self, save_path, start = 5, stop = 30, seed = 42):
        if self.num_clusters != None:
            return
        if self.features_df is None:
            print('Run data preparation before this step!')
            return
        start = 5
        stop = 30
        min_dist_list = []
        np.random.seed(seed)
        X = self.features_df
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
        if self.features_df is None:
            print('Run data preparation before this step!')
            return
        df_world = px.data.gapminder().query("year == 2007")
        # K MEANS CLUSTERING
        num_clusters = self.num_clusters
        features = self.features_df
        x = features.select_dtypes('number').values
        x = torch.from_numpy(x)
        cluster_ids_x, _ = kmeans(
            X=x,
            num_clusters=num_clusters,
            distance='euclidean',
            device=torch.device('cuda:0'),
            tqdm_flag = False
        )
        features['cluster_assigned'] = cluster_ids_x.numpy()

        # KEY FILE
        mapping_with_clusters = features[[self.key_country_col, "cluster_assigned"]]
        mapping_with_clusters['cluster_assigned'] = mapping_with_clusters['cluster_assigned'].astype('int')

        # FINAL DF
        df = df_world.merge(mapping_with_clusters, how = 'left', left_on = ["country"], right_on = [self.key_country_col])
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