# -*- coding:utf-8 -*-
# Created by LuoJie at 11/16/19
import os
import pathlib
import json

# 获取项目根目录
root = pathlib.Path(os.path.abspath(__file__)).parent.parent
data_path = os.path.join(root, 'data')
# 训练数据路径
train_data_path = os.path.join(root, 'data', 'AutoMaster_TrainSet.csv')
# 测试数据路径
test_data_path = os.path.join(root, 'data', 'AutoMaster_TestSet.csv')
# 停用词路径
# stop_word_path = os.path.join(root, 'data', 'stopwords/哈工大停用词表.txt')
# 预处理后的训练数据
train_seg_path = os.path.join(root, 'data', 'train_seg_data.csv')
# 预处理后的测试数据
test_seg_path = os.path.join(root, 'data', 'test_seg_data.csv')
# 合并训练集测试集数据
# merger_seg_path = os.path.join(root, 'data', 'merged_train_test_seg_data.csv')

# 自定义切词表
user_dict = os.path.join(root, 'data', 'user_dict.txt')

# Vocab词频文件
vocab_train_path = os.path.join(root, 'data', 'vocab_dict.json')
vocab_train_test_path = os.path.join(root, 'data', 'vocab_train_test.json')


class Configure:
    def __init__(self):
        self.root = root
        self.data_path = data_path

    def __getitem__(self, key):
        return self.__dict__.get(key)
    
    @staticmethod
    def from_json(path='params.json'):
        with open(os.path.join(root, path)) as f:
            config = json.load(f)
        return Configure.from_dict(config)

    @staticmethod
    def from_dict(dic):
        config = Configure()
        for k, v in dic.items():
            setattr(config, k, v)
        return config
# config = Configure()
# config.min_frequency = 5
# config.embedding_dim = 256

# with open(os.path.join(data_path, 'params.json'), 'w') as f:
#     json.dump(config.__dict__, f)


# with open(os.path.join(root, 'params.json')) as f:
#     config = json.load(f)
# params = Configure.from_dict(config)
params = Configure.from_json()
