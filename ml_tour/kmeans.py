import tensorflow as tf
import numpy as np
from typeguard import typechecked
from typing import Union

@tf.function
def train_kmeans(X: Union[tf.Tensor, np.ndarray], 
    k: int, 
    n_iter: int = 10) -> tf.Tensor:
    """Computes the clusters and centroids for given input data `X`
    by performing k-means clustering algorithm for `n_iter` iterations
    on `k` number of clusters.

    Args:
        X: A tensor that is automatically casted to dtype `tf.float32`
            and necessarily be a 2D tensor of shape [m, nd].
        k: An integer indicating the number of clusters.
        n_iter: An integer depicting the number of iterations that 
            the algorithm will be run for.

    Returns:
        centroids: A tensor containing centroids
            after clustering is performed.
    """

    X = tf.convert_to_tensor(X)
    X = tf.cast(X, tf.float32)
    assert len(tf.shape(X)) == 2, "Training data X must be represented as 2D array only"
    m = tf.shape(X)[0]

    random_select = tf.random.shuffle(X)
    init_centroids = random_select[:k]

    clusters = tf.zeros([m, ], dtype=tf.int64)
    centroids = init_centroids

    for _ in tf.range(n_iter):
        clusters = predict_kmeans(X, centroids)

        selector = tf.range(k, dtype=tf.int64)[:, None] == clusters[None, :]
        centroids_array = tf.TensorArray(tf.float32, size=k)

        for c in tf.range(k):
            select = selector[c]
            points = X[select]
            mean_points = tf.reduce_mean(points, axis=0)
            centroids_array = centroids_array.write(c, mean_points)
        
        centroids = centroids_array.stack()

    return centroids

@tf.function
def predict_kmeans(X: Union[tf.Tensor, np.ndarray], 
    centroids: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
    """Predicts the clusters for given input data `X`
    based on smallest euclidean distance from given `centroids`.

    Args:
        X: A tensor that is automatically casted to dtype `tf.float32`
            and necessarily be a 2D tensor of shape [m, nd].
        centroids: A tensor containing centroid points for each
            cluster and having shape [k, nd].

    Returns:
        clusters: A tensor with shape [m, ] containing predicted 
            cluster for each data point in `X`.
    """

    X = tf.convert_to_tensor(X)
    X = tf.cast(X, tf.float32)
    assert len(tf.shape(X)) == 2, "X must be represented as 2D array only"

    centroids = tf.convert_to_tensor(centroids)
    centroids = tf.cast(centroids, tf.float32)
    assert len(tf.shape(X)) == 2, "centroids must be represented as 2D array only"

    squared_diffs = tf.square(X[None, :, :] - centroids[:, None, :])
    euclidean_dists = tf.reduce_sum(squared_diffs, axis=-1) ** 0.5
    clusters = tf.argmin(euclidean_dists, axis=0)

    return clusters

class KMeans:
    """Class used to perform k-Means clustering algorithm
    on a given dataset. The number of clusters and the number
    of iterations the algorithm will be run for is to be
    specified beforehand.

    Example:
    >>> X = tf.convert_to_tensor([[1., 2.], [3., 4.], [5., 4.], [3., 2.], [1., 3.]])
    >>> k = 2
    >>> 
    >>> kmeans = KMeans(k, n_iter=20)
    >>> clusters = kmeans.fit_predict(X)
    >>> 
    >>> clusters
    <tf.Tensor: shape=(5,), dtype=int64, numpy=array([1, 0, 0, 0, 1])>
    >>> kmeans.centroids
    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[3.6666667, 3.3333333],
           [1.       , 2.5      ]], dtype=float32)>
    
    """

    @typechecked
    def __init__(self, k: int, n_iter: int = 10):
        """
        Constructs an object of the KMeans class. It can be used
        to perform k-Means clustering using the `fit_predict` method.

        Args:
            k: An integer indicating the number of clusters.
            n_iter: An integer depicting the number of iterations that 
            the algorithm will be run for.
        """

        self.k = k
        self.n_iter = n_iter
        self.centroids = None
    
    @typechecked
    def fit(self, X: Union[tf.Tensor, np.ndarray]):
        """Perform k-Means algorithm on given dataset.

        Args:
            X: A tensor that is automatically casted to dtype `tf.float32`
                and necessarily be a 2D tensor of shape [m, nd].
        """
        
        self.centroids = train_kmeans(X, self.k, self.n_iter)

    @typechecked
    def predict(self, X: Union[tf.Tensor, np.ndarray]):
        """Predict cluster for each data point in `X`.

        Args:
            X: A tensor that is automatically casted to dtype `tf.float32`
                and necessarily be a 2D tensor of shape [m, nd].
        
        Returns:
            clusters: A tensor with shape [m, ] containing predicted cluster
                for each data point in `X`.
        """
        clusters = predict_kmeans(X, self.centroids)
        return clusters

    @typechecked
    def fit_predict(self, X: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        """Perform k-Means algorithm on given dataset and return
        a tensor with the predicted cluster for each data 
        point in `X`.

        Args:
            X: A tensor that is automatically casted to dtype `tf.float32`
                and necessarily be a 2D tensor of shape [m, nd].
        
        Returns:
            clusters: A tensor with shape [m, ] containing predicted cluster
            for each data point in `X`.
        """
        
        self.fit(X)
        return self.predict(X)