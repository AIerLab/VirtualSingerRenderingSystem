import os

# 调整运行路
os.chdir("..")

import pickle
import time
from keras.models import load_model
from sklearn.model_selection import train_test_split

from model import GRU2One
from setting import *

if __name__ == '__main__':
    # 导入数据
    with open(input_seq_data, "rb") as f:
        X = pickle.load(f)
    with open(input_seq_label, "rb") as f:
        y = pickle.load(f)

    # 分割数据集
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=7)
    if continue_training:
        # 导入模型
        model = load_model(model_latest_path)
    else:
        # 新建模型
        model = (_ := GRU2One()).model

    # 开始训练
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              batch_size=batch_size, epochs=500)

    # 保存模型
    model.save(os.path.join(model_folder, f"model{int(time.time())}.h5"), save_format='h5')
