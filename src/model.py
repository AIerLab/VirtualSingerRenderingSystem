import tensorflow as tf
from keras.layers import CuDNNGRU
from keras.layers.core import Dense, Activation, Dropout
from keras.models import Sequential

from setting import *


# TODO 修改模型进行优劣二分类
class GRU2One():
    """
    use for sequential information
    """

    def __init__(self):
        # 定义模型
        self.model = Sequential([
            CuDNNGRU(units=128, return_sequences=True, name="layer_01", input_shape=(seq_length, 4)),
            Dropout(0.2),
            CuDNNGRU(units=64, return_sequences=False, name="layer_02"),
            Dropout(0.2),
            Dense(units=1, name="output"),
            Activation("linear"),
        ])
        # 编译模型
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])


class GRU2Many(tf.keras.Model):
    """
    use for discrete information
    """

    def __init__(self, batch_size):
        super().__init__()
        pass

    def call(self, input):
        pass
