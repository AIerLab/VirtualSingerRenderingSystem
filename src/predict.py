import os
import pickle

import pandas as pd
from keras.models import load_model

from train import normalizer, seq_length

if __name__ == '__main__':
    model_dir = "../model"
    model_folder = "GRUv2"

    list_dir = os.listdir(os.path.join(model_dir, model_folder))
    latest_saved_model_name = max(list_dir)
    model = load_model(os.path.join(model_dir, model_folder, latest_saved_model_name))

    # 导入数据
    with open("../asset/vsqx4_list.pickle", "rb") as f:
        vsqx4_list = pickle.load(f)

    with open("../asset/vsqx4_rank.pickle", "rb") as f:
        vsqx4_rank = pickle.load(f)

    # 数据预处理+预测
    pred_ranks = []
    y = []
    for rank, vsqx4 in zip(vsqx4_rank, vsqx4_list):
        X = []
        rank = rank / 100
        y.append(rank)
        for track in vsqx4.vsTrack:
            notes = [[float(note.t), float(note.dur), float(note.n), float(note.v)]
                     for part in track.vsPart
                     for note in part.VNote]

            notes_df = pd.DataFrame(notes, columns=["t", "dur", "n", "v"])
            notes_df = notes_df.apply(normalizer)

            X += [notes_df.iloc[i:i + seq_length + 1].values.tolist() for i in range(len(notes) - seq_length)]
        pred_rank = model.predict(X).mean()
        print(f"{pred_rank} v.s. {rank}")
        pred_ranks.append(pred_rank)

    # 保存批预测结果
    with open("../asset/result02.pickle", 'wb') as f:
        pickle.dump(list(zip(y, pred_ranks)), f)
