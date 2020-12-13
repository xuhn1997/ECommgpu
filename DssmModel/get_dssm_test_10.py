"""
获取DSSM的测试集
16到21号的
"""
import pandas as pd
import numpy as np
from DssmModel.gen_16_21_input import get_train_data_index

item_feature = ['itemID', 'categoryID', 'shopID', 'brandID']


def get_item_data_index():
    """
    生成item部分的特征
    :return:
    """
    data = get_train_data_index()

    item_profile = data[item_feature].drop_duplicates('itemID')

    print(item_profile.shape)  # (1189534, 4)
    print(item_profile.columns)

    return item_profile


# ii = get_item_data_index()
# print(ii['itemID'])
