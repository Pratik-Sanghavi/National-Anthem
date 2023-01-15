from k_means_clustering.KMeansClustering import KMeansCluster

k_means_cluster = KMeansCluster('national_anthem_scrape/national_anthem_dataset/features.csv')
k_means_cluster.optimal_cluster_search('./k_means_clustering/artifacts/distance_metric.png')