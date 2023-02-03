from k_means_clustering.KMeansClustering import KMeansCluster
from settings import audio_feat_write_path, text_feat_write_path, BASE_PATH

k_means_cluster = KMeansCluster(audio_feat_write_path, 
                                text_feat_write_path, 
                                f'{BASE_PATH}/key.csv', 
                                'Audio_File',
                                 'file_name', 
                                 'Country', 
                                 'Country_Name')
k_means_cluster.data_merge()
k_means_cluster.optimal_cluster_search('./k_means_clustering/artifacts/distance_metric.png')
k_means_cluster.visualise_cluster('./k_means_clustering/artifacts/world_map_clustered.png')