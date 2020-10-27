"""
针对于训练集来说的
此文件是对于deepfm的特征分类
分为连续值的特征以及离散的特征，以及没用的特征
总的特征为：
之后的总的特征就是userID, itemID, sim, label，'sex', 'age',
'ability'， categoryID, shopID, brandID, behavior, day
"""
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import pandas as pd
from deepfm_feature.generate_underline_test_data import get_deepfm_test_data
from deepfm_feature.generate_underline_feature import get_all_underline_feature

# 首先划分好数值特征，以及离散特征以及遗弃特征
CONTINUE_COLS = [
    'sex', 'ability', 'categoryID',
    'shopID', 'brandID', 'behavior',
    'day'
]

NUMBERS_COLS = [
    'sim', 'age'
]

IGNORE_COLS = [
    'userID', 'itemID',
    'label'
]


def get_feature_dict():
    """
    此函数是为获取所有特征的索引位置
    :return:
    """
    # 则要要将测试集以及训练集进行concat一下
    # 但是此处需要用到1号到15号的数据
    # 所以就直接用线下测试集
    recall_all = get_deepfm_test_data()
    """
    以上的数据格式就是
     userID, itemID, sim, 'sex', 'age', 
    'ability'， categoryID, shopID, brandID, behavior, day
    """
    feature_dict = {}  # 记录下特征的位置
    total_feature = 0
    for col in recall_all.columns:
        if col in IGNORE_COLS:
            continue
        elif col in NUMBERS_COLS:
            """
            当位于连续值特征的处理
            位置只有一个，feat_value就是它的本身
            """
            feature_dict[col] = total_feature
            total_feature = total_feature + 1
        else:
            """
            就是位于离散值的特征，处理就是
            先统计该特征的所有不同的值，然后记录下
            不同值得位置
            """
            unique_val = recall_all[col].unique()
            feature_dict[col] = dict(
                zip(unique_val, range(total_feature,
                                      len(unique_val) + total_feature))
            )
            total_feature = total_feature + len(unique_val)

    return feature_dict, total_feature


# feat, total = get_feature_dict()
# print(total)  # 进行处理后总共有的特征数601221
# print(feat)

"""
接下来就是对于训练集的转化，
就是转化比如以下特征
原始特征:男/10/100
索引特征:0/6/7-->其中这些数据表示这些特征的位置索引
特征值：1/10/100-->离散的特征值就是1，连续特征的话就是它原始的值
"""


def deal_underline_train_data():
    """
    针对于训练集的转化
    之后的总的特征就是['userID', 'itemID', 'sim', 'label', 'behavior', 'day', 'sex', 'age',
       'ability', 'categoryID', 'shopID', 'brandID'],
    :return:
    """
    # 获取训练集
    df_train = get_all_underline_feature()
    # 获取记录特征的位置字典
    feature_dict, _ = get_feature_dict()
    # 复制下两次，作为记录位置及其转化之后的特征值
    train_feature_index = df_train.copy()
    train_feature_value = df_train.copy()

    # 开始转化的过程
    for col in train_feature_index.columns:
        if col in IGNORE_COLS:
            continue
        elif col in NUMBERS_COLS:
            # 连续值的特征的原始值还是原来的值
            train_feature_index[col] = feature_dict[col]
        else:
            """
            使用map做指定的映射，找到具体那个离散特征的具体位置索引
            """
            train_feature_index[col] = train_feature_index[col].map(
                feature_dict[col]
            )
            train_feature_value[col] = 1

    return train_feature_index, train_feature_value


"""
以上类似的对于测试集进行转化
但是要注意特征的名字与训练集的不同。。。。
"""


def deal_underline_test_data():
    """
    对于线下的测试集进行转化做法同上
     以上的数据格式就是
     userID, itemID, sim, 'sex', 'age',
    'ability'， categoryID, shopID, brandID, behavior, day
    :return:
    """
    # 获取线下训练集
    df_test = get_deepfm_test_data()
    # 获取特征索引的位置
    feature_dict, _ = get_feature_dict()
    # 生成两个副本，一个记录索引，一个记录转化之后的真实值
    test_feature_index = df_test.copy()
    test_feature_value = df_test.copy()

    for col in test_feature_index.columns:
        if col in IGNORE_COLS:
            continue
        elif col in NUMBERS_COLS:
            test_feature_index[col] = feature_dict[col]
        else:
            test_feature_index[col] = test_feature_index[col].map(
                feature_dict[col]
            )
            test_feature_value[col] = 1

    return test_feature_index, test_feature_value


# test_data_index, test_data_value = deal_underline_test_data()
# print("转换后测试的维度是。。。")
# print(test_data_value.shape)
# print(test_data_index.shape)
