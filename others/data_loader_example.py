# -*- coding:utf-8 -*-
# Created by LuoJie at 11/16/19
import re
import jieba
import pandas as pd
from utils.multi_proc_utils import parallelize
from utils.config import train_seg_path, test_seg_path, merger_seg_path, user_dict
from utils.config import stop_word_path, train_data_path, test_data_path

# 自定义词表
jieba.load_userdict(user_dict)


def data_generate(train_data_path, test_data_path):
    '''
    数据加载+预处理
    :param train_data_path:训练集路径
    :param test_data_path: 测试集路径
    :return: 训练数据 测试数据  合并后的数据
    '''
    # 1.加载数据
    train_df, test_df = load_dataset(train_data_path, test_data_path)
    print('train data size {},test data size {}'.format(len(train_df), len(test_df)))

    # 2. 空值填充
    train_df.dropna(subset=['Question', 'Dialogue', 'Report'], how='any', inplace=True)
    test_df.dropna(subset=['Question', 'Dialogue'], how='any', inplace=True)

    # 3.多线程, 批量数据处理
    train_df = parallelize(train_df, data_frame_proc)
    test_df = parallelize(test_df, data_frame_proc)

    # 4.保存处理好的 训练 测试集合
    train_df.to_csv(train_seg_path, index=None, header=True)
    test_df.to_csv(test_seg_path, index=None, header=True)

    # 5. 合并训练测试集合
    train_df['merged'] = train_df[['Question', 'Dialogue', 'Report']].apply(lambda x: ' '.join(x), axis=1)
    test_df['merged'] = test_df[['Question', 'Dialogue']].apply(lambda x: ' '.join(x), axis=1)

    merged_df = pd.concat([train_df[['merged']], test_df[['merged']]], axis=0)
    print('train data size {},test data size {},merged_df data size {}'.format(len(train_df), len(test_df),
                                                                               len(merged_df)))

    # 6. 保存合并数据
    merged_df.to_csv(merger_seg_path, index=None, header=False)

    return train_df, test_df, merged_df


def load_dataset(train_data_path, test_data_path):
    '''
    数据数据集
    :param train_data_path:训练集路径
    :param test_data_path: 测试集路径
    :return:
    '''
    # 读取数据集
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    return train_data, test_data


def load_stop_words(stop_word_path):
    '''
    加载停用词
    :param stop_word_path:停用词路径
    :return: 停用词表 list
    '''
    # 打开文件
    file = open(stop_word_path, 'r', encoding='utf-8')
    # 读取所有行
    stop_words = file.readlines()
    # 去除每一个停用词前后 空格 换行符
    stop_words = [stop_word.strip() for stop_word in stop_words]
    return stop_words


# 加载停用词
stop_words = load_stop_words(stop_word_path)


def clean_sentence(sentence):
    '''
    特殊符号去除
    :param sentence: 待处理的字符串
    :return: 过滤特殊字符后的字符串
    '''
    if isinstance(sentence, str):
        return re.sub(
            r'[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好|您好',
            '', sentence)
    else:
        return ''


def filter_stopwords(words):
    '''
    过滤停用词
    :param seg_list: 切好词的列表 [word1 ,word2 .......]
    :return: 过滤后的停用词
    '''
    return [word for word in words if word not in stop_words]


def sentence_proc(sentence):
    '''
    预处理模块
    :param sentence:待处理字符串
    :return: 处理后的字符串
    '''
    # 清除无用词
    sentence = clean_sentence(sentence)
    # 切词，默认精确模式，全模式cut参数cut_all=True
    words = jieba.cut(sentence)
    # 过滤停用词
    words = filter_stopwords(words)
    # 拼接成一个字符串,按空格分隔
    return ' '.join(words)


def data_frame_proc(df):
    '''
    数据集批量处理方法
    :param df: 数据集
    :return:处理好的数据集
    '''
    # 批量预处理 训练集和测试集
    for col_name in ['Brand', 'Model', 'Question', 'Dialogue']:
        df[col_name] = df[col_name].apply(sentence_proc)

    if 'Report' in df.columns:
        # 训练集 Report 预处理
        df['Report'] = df['Report'].apply(sentence_proc)
    return df


if __name__ == '__main__':
    # 数据集批量处理
    data_generate(train_data_path, test_data_path)
