"""
针对于训练集来说的
此文件是对于deepfm的特征分类
分为连续值的特征以及离散的特征，以及没用的特征
总的特征为：
之后的总的特征就是userID, itemID, sim, label，'sex', 'age',
'ability'， categoryID, shopID, brandID, behavior, day
"""
import pandas as pd

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

