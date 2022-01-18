import os

# 调整运行路径
os.chdir("..")

from keras.models import load_model
from data import input_sequence_data_preparation
from setting import *
from utils.vsqxt import vsqx


class Predictor():
    def __init__(self):
        # 导入模型
        self.model = load_model(model_latest_path)

    def predict(self, vsqx4):
        # 数据预处理
        X = input_sequence_data_preparation(vsqx4)
        # 预测排名
        pred_rank = self.model.predict(X).mean()
        return pred_rank


if __name__ == '__main__':
    # 导入数据
    vsqx4 = vsqx.read(r"E:\Database\Vocaloid VSQX Ranking Database\testingData.vsqx")

    predictor = Predictor()
    pred_rank = predictor.predict(vsqx4)

    # 显示排名
    print(pred_rank)
