import tensorflow as tf
import numpy as np
from typeguard import typechecked
from typing import Union

@typechecked
def check_1d(tensor: tf.Tensor, name: str):
    assert len(tf.shape(tensor)) == 1, "%s must be represented as a 1D vector only" % name

@typechecked
def check_2d(tensor: tf.Tensor, name: str):
    assert len(tf.shape(tensor)) == 2, "%s must be represented as a 2D array only" % name

@tf.function
def as_float(tensor: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
    tensor = tf.convert_to_tensor(tensor)
    tensor = tf.cast(tensor, tf.float32)
    return tensor

@tf.function
def get_mean_stdev(X: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    mean = tf.reduce_mean(X, axis=0)
    stdev = tf.math.reduce_std(X, axis=0)
    return mean, stdev

@tf.function
def normalise(X: tf.Tensor, mean: tf.Tensor, stdev: tf.Tensor) -> tf.Tensor:
    return (X - mean) / stdev