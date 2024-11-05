from sklearn.datasets import load_iris
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pandas as pd
import toolz as tz
from itertools import product

# 0. Adjust parameters
NUM_CLUSTERS = 2       # Number of clusters for K-Means (Experiment with 2, 3, 4)
MAX_ITER = 5           # Maximum number of iterations for the algorithm (Experiment with 5, 10, 20)
FEATURE_X_INDEX = 2    # Index of the feature for the x-axis (0 to 3 for Iris)
FEATURE_Y_INDEX = 3    # Index of the feature for the y-axis (0 to 3 for Iris)

iris_data = load_iris(as_frame=True)


def extract_training_data():
    """
    @return: the training data to be used for the k-means model
    """
    iris_data = load_iris(as_frame=True)
    return iris_data['frame'].drop('target', axis=1)


@tz.curry
def get_cluster_labels_and_centers(dataset: pd.DataFrame, clusters_and_iterations):
    """
    @param dataset: a dataset where each row contains the sepal and petal length/width
    of a flower
    @param clusters_and_iterations: a pair containing the number of clusters to search for
    and the number of iterations to run the k-means algorithm for
    @return: the predicted target labels of each flower when using the k-means algorithm with
    the passed number of iterations and clusters and the predicted cluster centers
    """

    num_of_clusters, num_of_iterations = clusters_and_iterations
    model = KMeans(n_clusters=num_of_clusters, max_iter=num_of_iterations).fit(dataset)
    return model.predict(dataset), model.cluster_centers_


@tz.curry
def print_graph(dataset: pd.DataFrame, number_of_iterations, flower_labels_and_clusters):
    """
    @param dataset:  a dataset where each row contains the sepal and petal length/width
    of a flower
    @param number_of_iterations: the number of iterations used when finding the clusters
    each flower belongs to
    @param flower_labels_and_clusters: the clusters and labels for the flowers in the dataset
    @return: plots a scatterplot comparing the sepal length/width with each cluster
    colored differently
    """
    def convert_cluster_labels_to_color(cluster_labels):
        cluster_label_to_color = plt.get_cmap("Dark2")
        return tz.thread_last(cluster_labels,
                              Normalize(),
                              cluster_label_to_color)

    labels, clusters = flower_labels_and_clusters
    fig = plt.figure()
    plt.scatter(dataset['petal length (cm)'],
                dataset['petal width (cm)'],
                c=convert_cluster_labels_to_color(labels))
    plt.scatter(clusters[:, 2], clusters[:, 3], c='b', marker='x')

    plt.xlabel("Petal length (cm)")
    plt.ylabel("Petal width (cm)")

    num_of_clusters = len(clusters)
    plt.title(f"Comparing {num_of_clusters} clusters of flowers by their sepal length/width by using {number_of_iterations} iterations "
              f"of k-means", {"size": 8})
    fig.savefig(f"{num_of_clusters}_clusters_{number_of_iterations}_iterations.png")


def gen_graphs(dataset: pd.DataFrame, cluster_and_max_iteration_configurations):
    """
    @param dataset: a dataset where each row contains the sepal and petal length/width
    of a flower
    @param cluster_and_max_iteration_configurations: a collection of tuples where
    each has parameters set for the number of clusters and the number of iterations
    to use in a k-means model
    @return: generates a graph of the clusters of flowers with similar petal lengths
    and widths for each cluster and iteration configuration
    """
    get_labels_and_clusters = get_cluster_labels_and_centers(dataset)
    label_and_cluster_data = list(map(get_labels_and_clusters, cluster_and_max_iteration_configurations))
    for i in range(len(label_and_cluster_data)):
        print_graph(dataset,
                    cluster_and_max_iteration_configurations[i][1],
                    label_and_cluster_data[i])


clusters_and_iteration_pairs = list(product([2, 3, 4], [5, 10, 20]))
data = extract_training_data()
gen_graphs(data, clusters_and_iteration_pairs)

