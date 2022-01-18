import os
import pickle
import time

import pandas as pd

from keras.layers.core import Dense, Activation, Dropout

from keras.layers.recurrent import GRU, LSTM
from keras.layers import CuDNNLSTM, CuDNNGRU

from keras.models import Sequential, load_model
from sklearn.model_selection import train_test_split


# define a normaliser
def normalizer(dataArray):
    if dataArray.max() - dataArray.min() == 0:
        return dataArray
    return (dataArray - dataArray.min()) / (dataArray.max() - dataArray.min())


# define a standardizer
def standardizer(dataArray):
    if dataArray.max() - dataArray.min() == 0:
        return dataArray
    return (dataArray - dataArray.mean()) / dataArray.std()

# 定义超参数
batch_size = 32
seq_length = 3

model_dir = "../model"
model_folder = "GRUv2"
continue_training = False

if __name__ == '__main__':
    # 导入数据
    with open("../asset/vsqx4_list.pickle", "rb") as f:
        vsqx4_list = pickle.load(f)

    with open("../asset/vsqx4_rank.pickle", "rb") as f:
        vsqx4_rank = pickle.load(f)

    # 数据预处理
    X = []
    y = []
    for rank, vsqx4 in zip(vsqx4_rank, vsqx4_list):
        rank = rank / 100
        if rank > 1: # 大于1作为测试集
            continue
        for track in vsqx4.vsTrack:
            notes = [[float(note.t), float(note.dur), float(note.n), float(note.v)] for part in track.vsPart
                     for note in part.VNote]

            notes_df = pd.DataFrame(notes, columns=["t", "dur", "n", "v"])
            notes_df = notes_df.apply(normalizer)

            X += [notes_df.iloc[i:i + seq_length + 1].values.tolist() for i in range(len(notes) - seq_length)]
            y += [rank for _ in range(len(notes) - seq_length)]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=7)

    if continue_training:
        # 导入模型
        list_dir = os.listdir(os.path.join(model_dir, model_folder))
        latest_saved_model_name = max(list_dir)
        model = load_model(os.path.join(model_dir, model_folder, latest_saved_model_name))
    else:
        # 新建模型
        model = Sequential([
            CuDNNGRU(units=128, return_sequences=True, name="layer_01", input_shape=(seq_length, 4)),
            Dropout(0.2),
            CuDNNGRU(units=64, return_sequences=False, name="layer_02"),
            Dropout(0.2),
            Dense(units=1, name="output"),
            Activation("linear"),
        ])

        # 编译模型
        model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mape'])

    # 开始训练
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              batch_size=batch_size, epochs=500)

    # 保存模型
    model.save(os.path.join(model_dir, model_folder, f"model{int(time.time())}.h5"), save_format='h5')
