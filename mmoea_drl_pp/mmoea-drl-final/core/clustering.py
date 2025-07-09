from sklearn.cluster import KMeans
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning

# def cluster_population(flattened_population, num_clusters):
#     """
#     Args:
#         flattened_population: list of flattened task sequences (1D list with -1 separators)
#         num_clusters: int, number of clusters for k-means

#     Returns:
#         labels: list of cluster IDs for each individual
#     """
#     # Pad with zeros to equal length for k-means input
#     max_len = max(len(ind) for ind in flattened_population)
#     padded = [ind + [0] * (max_len - len(ind)) for ind in flattened_population]

#     X = np.array(padded)
#     with warnings.catch_warnings():
#         warnings.simplefilter("ignore", ConvergenceWarning)
#         kmeans = KMeans(n_clusters=num_clusters, n_init='auto', random_state=42)
#         labels = kmeans.fit_predict(X)
#     return labels

def cluster_population(flattened_population, num_clusters):
    X = np.array(flattened_population)

    # Detect number of unique individuals
    unique_rows = np.unique(X, axis=0)
    safe_n_clusters = min(num_clusters, len(unique_rows))

    # print(len(unique_rows),safe_n_clusters)

    # Only cluster if there's more than 1 unique individual
    if safe_n_clusters < 2:
        return [0] * len(X)  # Assign everyone to same cluster

    kmeans = KMeans(n_clusters=safe_n_clusters, random_state=0)
    labels = kmeans.fit_predict(X)
    return labels

# if __name__ == "__main__":
#     from encoding import generate_individual, flatten_individual

#     num_tasks = 20
#     num_robots = 6
#     population_size = 10
#     population = [generate_individual(num_tasks, num_robots) for _ in range(population_size)]
#     flat_population = [flatten_individual(ind) for ind in population]

#     cluster_labels = cluster_population(flat_population, num_clusters=4)

#     for i, (ind, label) in enumerate(zip(population, cluster_labels)):
#         print(f"Individual {i}: Cluster {label} => {ind}")