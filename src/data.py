import json
import os
import pickle

import pandas as pd

from utils.vsqxt import vsqx

import multiprocessing

if __name__ == '__main__':
    # 设置
    database_dir = "E:\Database\Vocaloid VSQX Ranking Database"
    dataset = [{"folder": "Data2020", "source": "source.json"},
               {"folder": "Data2021", "source": "source.csv"}]

    # 合并路径
    dataset_dir = [os.path.join(database_dir, dataset["folder"]) for dataset in dataset]
    dataset_source_file_path = [os.path.join(dataset_dir[i], dataset[i]["source"]) for i in range(len(dataset_dir))]


    # 读取2020json source
    with open(dataset_source_file_path[0], 'r', encoding='utf-8') as f:
        source = json.load(f)  # source is a dictionary
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

    # 持久化存储
    with open("../asset/vsqx4_list.pickle", "wb") as f:
        pickle.dump(vsqx4_list, f)

    with open("../asset/vsqx4_rank.pickle","wb") as f:
        pickle.dump(data_df['rank'], f)
