import tensorflow as tf


class LSTM(tf.keras.Model):
    """
    use for sequential information
    """

    def __init__(self, batch_size):
        super().__init__()
        self.batch_size = batch_size
        self.cell_01 = tf.keras.layers.LSTMCell(units=256, name="LSTM_01")
        self.cell_02 = tf.keras.layers.LSTMCell(units=128, name="LSTM_02")
        self.dense_01 = tf.keras.layers.Dense(units=50, name="DENSE_01")
        self.dense_02 = tf.keras.layers.Dense(units=1, name="DENSE_02")

    def call(self, input):
        state01 = self.cell_01.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        state02 = self.cell_02.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        x = 0
        for value in input[1:]:
            x = value
            x, state01 = self.cell_01(x, state01)
            x, state02 = self.cell_02(x, state02)
        x = self.dense_01(x)
        # output = tf.nn.softmax(x) # 测试分类方法使用
        output = self.dense_02(x)
        return output


class NN(tf.keras.Model):
    """
    use for discrete information
    """

    def __init__(self, batch_size):
        super().__init__()
