import plotly.express as px
import pandas as pd
import matplotlib.pyplot as plt
from kmeans_pytorch import kmeans
import torch

df_world = px.data.gapminder().query("year == 2007")

# K MEANS CLUSTERING
num_clusters = 12
song_features = pd.read_csv('national_anthem_scrape/national_anthem_dataset/features.csv')
x = song_features.select_dtypes('number').values
x = torch.from_numpy(x)
cluster_ids_x, cluster_centers = kmeans(
    X=x,
    num_clusters=num_clusters,
    distance='euclidean',
    device=torch.device('cuda:0'),
    tqdm_flag = False
)
song_features['cluster_assigned'] = cluster_ids_x.numpy()

# KEY FILE
mapping_df = pd.read_csv('national_anthem_scrape/national_anthem_dataset/key.csv')
mapping_with_clusters = pd.merge(mapping_df, song_features, left_on = "Audio_File", right_on="file_name")[["Country", "cluster_assigned"]]
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

fig.show()