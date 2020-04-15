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
    >>> X = tf.random.uniform([15, 10], 0, 256) 
    >>> X.shape
    TensorShape([15, 10])
    >>>
    >>> k = 5
    >>> pca = PCA(k)
    >>>
    >>> X_transformed = pca.fit_transform(X)
    >>> X_transformed
    <tf.Tensor: shape=(15, 5), dtype=float32, numpy=
    array([[ 0.56003416,  1.3160728 ,  0.73852843, -0.15283212, -1.6777785 ],
        [ 0.39692688,  0.31944594,  0.7354324 , -0.06780335,  0.541604  ],
        [ 0.03630981, -0.14970237, -1.1399893 ,  1.5895145 ,  1.6312952 ],
        [-2.3734121 , -0.8616407 ,  0.11035006, -0.41071987, -0.9686606 ],
        [-0.54395545,  1.087436  , -2.1433825 ,  0.5442976 , -0.45127678],
        [ 1.4337428 ,  0.11133459,  1.307143  , -1.2419394 ,  0.01651086],
        [ 0.21285607, -0.59568626,  0.2282213 ,  1.953423  , -0.8812959 ],
        [ 0.50399494, -0.86252797, -0.4800622 , -1.6740584 ,  0.94912446],
        [-1.5081549 ,  2.2120705 ,  0.9293314 ,  0.6147077 ,  1.1135253 ],
        [-0.07887363, -1.3832867 ,  0.1410839 , -1.5719367 , -0.08556193],
        [ 0.20535675, -0.9995426 ,  1.0590649 , -1.6675174 ,  0.1789697 ],
        [-1.0916313 , -0.38031933, -0.50004673,  3.276281  ,  0.24024095],
        [ 0.7828646 ,  1.1728046 ,  0.61039066, -0.4449395 , -1.2169099 ],
        [ 0.08132294, -0.45601034, -1.1843696 , -0.64388186,  0.6929081 ],
        [ 1.3826163 , -0.5304493 , -0.41169184, -0.10259444, -0.08269954]],
        dtype=float32)>
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