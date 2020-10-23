"""
此文件是获取对于deepfm的线下测试集。
为1号到15号到数据，也就是之前整理好的线上训练集
"""
import pandas as pd
import numpy as np


def get_deepfm_test_data():
    """
    获取1到15号的数据当作训练集
    :return:
    """
    recall_test_data = pd.read_csv("../recall_list/train_data.csv")

    # 获取user数据集
    users_df = pd.read_csv("../data/user.csv", header=None)
    users_df.columns = ['userID', 'sex', 'age', 'ability']

    # 获取item数据集
    items_df = pd.read_csv("../data/df_item.csv")

    # 对于recall进行数据筛选
    recall_df = recall_test_data[['userID', 'itemID', 'sim']]
    # 然后进行数据合并的过程
    recall_df = pd.merge(left=recall_df, right=users_df, on=['userID'], how='left')
    recall_df = pd.merge(left=recall_df, right=items_df, on=['itemID'], how='left')

    """
    合并之后总的特征就是
    userID, itemID, sim, 'sex', 'age', 
    'ability'， categoryID, shopID, brandID
    """
    return recall_df


