import tensorboard as tf
import numpy as np
from typeguard import typechecked
from typing import Union

@typechecked
def check_2d(tensor: tf.Tensor, name: str):
    assert len(tf.shape(tensor)) == 2, "%s must be represented as a 2D array only" % name

@typechecked
def as_float(tensor: Union[tf.Tensor, np.ndarray]) -> tf.Tensor:
    tensor = tf.convert_to_tensor(tensor)
    tensor = tf.cast(tensor, tf.float32)
    return tensor