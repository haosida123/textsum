# -*- coding:utf-8 -*-
# Created by LuoJie at 11/16/19
import os
import pathlib

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

# vocab 建立词典最小词频
min_frequency = 5

embedding_dim = 256
