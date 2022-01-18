######### 工程设置 #################################################################
import os
from typing import Dict, List

######### 自定义工程参数 ##############################################################
# 定义模型超参数
batch_size = 32
seq_length = 3

# 数据库路径
__database_dir: str = os.path.abspath("E:\Database\Vocaloid VSQX Ranking Database")

# 数据集描述
__dataset_desc: List[Dict[str, str]] = [{"folder": "Data2020", "source": "source.json"},
                                        {"folder": "Data2021", "source": "source.csv"}]

# 数据预处理持久化文件路径
dataset_data_file = os.path.abspath("../asset/vsqx4_list.pickle")
dataset_label_file = os.path.abspath("../asset/vsqx4_rank.pickle")
input_seq_data = os.path.abspath("../asset/formated_seq_data.pickle")
input_seq_label = os.path.abspath("../asset/formated_seq_rank.pickle")

# 模型路径
__model_dir: str = os.path.abspath("../model")

# 模型名称
model_name: str = "GRU2One"  # available model: GRU2One, GRU2Many

# 继续训练
continue_training: bool = True

######### 自动生成参数，请勿修改 #########################################################
# 获取数据集路径
dataset_dir: List[str] = [os.path.join(__database_dir, dataset["folder"]) for dataset in __dataset_desc]
dataset_source_file_path: List[str] = [os.path.join(dataset_dir[i], __dataset_desc[i]["source"]) for i in
                                       range(len(dataset_dir))]

# 模型文件夹路径创建
model_folder: str = os.path.join(__model_dir, model_name)
if not os.path.exists(model_folder):
    os.mkdir(model_folder)

# 最新模型文件路径获取
model_latest_path: str = ""
__model_file_list: List[str] = os.listdir(model_folder)
if not __model_file_list:
    __model_latest = ""
    continue_training = False  # 修正继续训练设置
else:
    __model_latest: str = max(__model_file_list)
    model_latest_path = os.path.join(model_folder, __model_latest)
