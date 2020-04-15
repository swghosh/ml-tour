import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from typeguard import typechecked
from typing import Union

from . import helpers

@tf.function
def train_pca(X: Union[tf.Tensor, np.ndarray], 
    k: int) -> (tf.Tensor, tf.Tensor, tf.Tensor):
    """Performs principal component analysis (PCA) on given
    data `X` and produces the top-`k` eigen vectors which 
    can help transform data into lower dimension. `X` is always
    normalized based on feature-wise mean and standard deviation.

    Args:
        X: A tensor that is automatically casted to dtype `tf.float32`
            and necessarily be a 2D tensor of shape [m, nd].
        k: An integer indicating the number of top eigen vectors,
            that'll be used the dimension for reduced features.

    Returns:
        Uk: A tensor containing top-`k` eigen vectors of the 
            generated covariance matrix, which can be used to produce
            features of `k` dimension. It has shape [k, nd].
        mean: Feature-wise means of input data, useful 
            for data normalisation. It has shape [nd, ].
        stdev: Feature-wise standard deviations of input data,
            useful for data normalisation. It has shape [nd, ].
    """

    X = helpers.as_float(X)
    helpers.check_2d(X, 'X')

    n = tf.shape(X)[-1]
    tf.debugging.assert_less(k, n, "value of reduced dimension k should be less than number of features n")

    # normalise X by subtracting mean and dividing by stdev
    mean, stdev = helpers.get_mean_stdev(X)
    X_normalised = helpers.normalise(X, mean, stdev)

    covar_matrix = tfp.stats.covariance(X_normalised, sample_axis=0)
    _, eigen_vectors = tf.linalg.eigh(covar_matrix)

    # select top k eigen vectors only
    Uk = eigen_vectors[:k]
    return Uk, mean, stdev

@tf.function
def transform_pca(X: Union[tf.Tensor, np.ndarray],
    mean: Union[tf.Tensor, np.ndarray],
    stdev: Union[tf.Tensor, np.ndarray],
    Uk: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
    """Performs dimensionality reduction based on already 
    computed principal components on given data `X`. The 
    data points in `X` are reduced to `k` number of features
    basis given matrix `Uk`, containing top-`k` eigen vectors.
    `X` is always normalised before computation of reduced features.

    Args:
        X: A tensor that is automatically casted to dtype `tf.float32`
            and necessarily be a 2D tensor of shape [m, nd].
        mean: Feature-wise means of input data, used 
            for data normalisation. It has shape [nd, ].
        stdev: Feature-wise standard deviations of input data, used 
            for data normalisation. It has shape [nd, ].
        Uk: A tensor containing top-`k` eigen vectors of the 
            generated covariance matrix, which can be used to produce
            features of `k` dimension. It has shape [k, nd].
        

    Returns:
        X_transformed: A tensor containing data with reduced PCA 
            features and has shape [m, k].
    """
    
    X, mean, stdev, Uk = [helpers.as_float(tensor) for tensor in [X, mean, stdev, Uk]]
    helpers.check_2d(X, 'X')
    helpers.check_1d(mean, 'mean')
    helpers.check_1d(stdev, 'stdev')
    helpers.check_2d(Uk, 'Uk')

    X_normalised = helpers.normalise(X, mean, stdev)
    X_transformed = X_normalised @ tf.transpose(Uk)
    return X_transformed

class PCA:
    """Class used to perform Principal Component Analysis (PCA)
    algorithm on a given dataset. The number of principal components
    to retain is specified beforehand.

    Example:
    >>>
    >>> k = 10
    >>> pca = PCA(k)
    >>> X_transformed = pca.fit_transform(X)
    >>>
    
    """
    @typechecked
    def __init__(self, k: int):
        """Constructs an object of the PCA class. It can be used
        to perform PCA algorithm using the `fit` or `fit_transform` methods.

        Args:
            k: An integer indicating the number of dimensions to retain
                in case of reduced features.
        """
        self.k = k
        self.mean = None
        self.stdev = None
        self.Uk = None
    
    @typechecked
    def fit(self, X: Union[tf.Tensor, np.ndarray]):
        """Perform PCA on a given dataset.

        Args:
            X: A tensor that is automatically casted to dtype `tf.float32`
                and necessarily be a 2D tensor of shape [m, nd].
        """
        self.Uk, self.mean, self.stdev = train_pca(X, self.k)

    @typechecked
    def transform(self, X: Union[tf.Tensor, np.ndarray]):
        """Transform all data points in `X` to 
        the reduced dimensionality.

        Args:
            X: A tensor that is automatically casted to dtype `tf.float32`
                and necessarily be a 2D tensor of shape [m, nd].

        Returns:
            X_transformed: A tensor containing data with reduced PCA 
                features and has shape [m, k]. 
        """
        X_transformed = transform_pca(X, self.mean, self.stdev, self.Uk)
        return X_transformed

    @typechecked
    def fit_transform(self, X: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
        """Perform PCA on a given dataset and 
        get a dataset with reduced features.

        Args:
            X: A tensor that is automatically casted to dtype `tf.float32`
                and necessarily be a 2D tensor of shape [m, nd].

        Returns:
            X_transformed: A tensor containing data with reduced PCA 
                features and has shape [m, k]. 
        """
        self.fit(X)
        return self.transform(X)