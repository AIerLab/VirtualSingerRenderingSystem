import json
import multiprocessing
import pickle

import pandas as pd

from setting import *
from utils.preprocess.data import *
from utils.vsqxt import vsqx


def __dataset_load():
    # 读取2020json source
    with open(dataset_source_file_path[0], 'r', encoding='utf-8') as f:
        source = json.load(f)
        data_df01 = pd.DataFrame(
            [(source[sourceIndex]["rank"],
              os.path.join(dataset_dir[0], source[sourceIndex]["file"])) for sourceIndex in range(len(source))],
            columns=["rank", "path"])

    # 读取2021csv source
    source_df = pd.read_csv(dataset_source_file_path[1])
    vsqx_file_name_path_pair = []
    for name in source_df.name:
        vsqx_project_folder = os.path.join(dataset_dir[1], name)
        for file_name in os.listdir(vsqx_project_folder):
            if file_name.endswith(".vsqx"):
                vsqx_file_name_path_pair.append((name, os.path.join(vsqx_project_folder, file_name)))
    path_df = pd.DataFrame(vsqx_file_name_path_pair, columns=["name", "path"])
    source_df = source_df.merge(path_df, on=["name"])
    data_df02 = source_df[["rank", "path"]]

    # 合并source数据表格
    data_df = data_df01.append(data_df02)

    # 多进程读取数据
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    vsqx4_list = pool.map(func=vsqx.read, iterable=data_df.path)

    # 持久化存储反序列化数据
    with open(dataset_data_file, "wb") as f:
        pickle.dump(vsqx4_list, f)
    with open(dataset_label_file, "wb") as f:
        pickle.dump(data_df['rank'], f)


# TODO 使用聚类进行数据初步的数据清洗，需要重新进行数据读取分析，不能用 sequence
def __input_sequence_training_data_preparation():
    # 导入数据
    with open(dataset_data_file, "rb") as f:
        vsqx4_list = pickle.load(f)
    with open(dataset_label_file, "rb") as f:
        vsqx4_rank = pickle.load(f)

    # 数据预处理
    X, y = [], []
    for rank, vsqx4 in zip(vsqx4_rank, vsqx4_list):
        # if rank > 1:  # 大于1作为测试集
        #     continue
        X += (seq := input_sequence_data_preparation(vsqx4))
        y += [rank] * len(seq)

    # 持久化存储模型输入数据
    with open(input_seq_data, "wb") as f:
        pickle.dump(X, f)
    with open(input_seq_label, "wb") as f:
        pickle.dump(y, f)


def input_sequence_data_preparation(vsqx4):
    notes_df = pd.DataFrame()
    for track in vsqx4.vsTrack:
        notes = [[float(note.t), float(note.dur), float(note.n), float(note.v)]
                 for part in track.vsPart
                 for note in part.VNote]

        notes_df = pd.DataFrame(notes, columns=["t", "dur", "n", "v"])
        notes_df = notes_df.apply(normalizer)
    X = [notes_df.iloc[i:i + seq_length + 1].values.tolist() for i in range(len(notes_df) - seq_length)]
    return X


if __name__ == '__main__':
    __dataset_load()
    __input_sequence_training_data_preparation()
