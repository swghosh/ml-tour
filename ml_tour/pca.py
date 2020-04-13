import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from typeguard import typechecked
from typing import Union

from . import helpers

@tf.function
def train_pca(X: Union[tf.Tensor, np.ndarray], 
    k: int) -> (tf.Tensor, tf.Tensor, tf.Tensor):

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
    
    X, mean, stdev, Uk = [helpers.as_float(tensor) for tensor in [X, mean, stdev, Uk]]
    helpers.check_2d(X, 'X')
    helpers.check_1d(mean, 'mean')
    helpers.check_1d(stdev, 'stdev')
    helpers.check_2d(Uk, 'Uk')

    X_normalised = helpers.normalise(X, mean, stdev)
    X_transformed = X_normalised @ tf.transpose(Uk)
    return X_transformed

