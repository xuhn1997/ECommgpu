import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from code_file.model import recommend_users, reshape_to_dataframe, generate_pairs
from scipy.sparse import *
import pandas as pd
import numpy as np
from code_file.utils import get_logs_from, reduce_mem_usage
from collections import defaultdict


def generate_online_train_user_sets():
    """
    这个函数是用来生成线上训练集的用户集合，即给这里面的所有用户推荐商品
    包含着1到15天的用户集合
    格式为{user:([i1, i2, i3], .....)}.....其中i1为物品的ID, i2为行为的程度，i3为行为的时间段
    :return:
    """
    filename = "../data/df_behavior_train.csv"
    train_data_df = pd.read_csv(filename)  # 获取datafame的数据集
    train_data_df = reduce_mem_usage(train_data_df)
    matrix = train_data_df[['userID', 'itemID', 'behavior', 'day']].values  # 这里返回的是一个二维数组，里面的数字全部转成浮点的数字
    user_logs = dict()
    for row in matrix:
        user_logs.setdefault(int(row[0]), [])
        user_logs[int(row[0])].append((int(row[1]), int(row[2]), int(row[3])))

    return user_logs


def get_label_function():
    """
    此函数是为了获得第16天的用户行为日志，从而与1到15天的召回列表合并生成label的功能
    格式为dataframe格式(userID, itemID, behavior)
    :return:
    """
    filename = "../data/df_behavior_test.csv"
    right_data = pd.read_csv(filename)
    right_datas = right_data[['userID', 'itemID', 'behavior']]
    # 讲itemID的数据类型转化成int64类型
    right_datas = right_datas.astype({'itemID': 'int64'})
    # right_data['itemID'] =
    right_datas.loc[right_datas['behavior'] == 'pv', 'behavior'] = 1
    # loc[i, j]的意思就是对于i行j列的意思
    right_datas.loc[right_datas['behavior'] == 'fav', 'behavior'] = 2
    right_datas.loc[right_datas['behavior'] == 'cart', 'behavior'] = 3
    right_datas.loc[right_datas['behavior'] == 'buy', 'behavior'] = 4

    right_datas = reduce_mem_usage(right_datas)
    return right_datas


def recall_list_online_function():
    """
    基于1到15号的用户进行召回过程，再与16天的用户真正行为的数据进行生成线上训练集
    :return:
    返回为dataframe格式(userID, itemID, sim, label)其中sim为用户对于itemID感兴趣的程度，以及label为1的话说明
    在16号的时候用户真的有对itemID产生了行为！！！！,为0的话就没有产生行为！！！！
    """
    # 先获取用户集合，以及他的历史集合

    user_logs = generate_online_train_user_sets()

    # 获取物品协同矩阵
    filename_commend_matrix = "../commonMatrix_iuf/common_matrix.npz"
    mat = load_npz(filename_commend_matrix)
    # 现在给1到15天的用户进行召回过程
    recommends = recommend_users(N=500, user_logs=user_logs, mat=mat)  # 格式为{userID:{j, sim}, {j1, sim1}}
    # 将召回的数据转化成dataframe格式
    recall_df = reshape_to_dataframe(recommends)  # 这里召回的1到15号的信息
    # 获取16天的用户行为，然后进行合并
    data_16 = get_label_function()  # 格式为userID， itemID，behavior的dataframe格式

    tmp = pd.merge(left=recall_df, right=data_16, on=['userID', 'itemID'],
                   how='left').rename(columns={'behavior': 'label'})
    tmp.to_csv("../recall_list/train_data.csv", index=False)
    print("save successfully.......")

    return tmp


# user_logs = generate_online_train_user_sets()
# train_data = get_label_function()
# print(train_data.dtypes)
#
# print(train_data.iloc[0:5])

# # 测试1到15召回的东西
# recall_dfs = recall_list_online_function()
#
# print(recall_dfs.shape)
# print(recall_dfs.iloc[0:5])

def down_sample(df, percent=10):
    """
    对于线上训练集进行负采样的函数
    :param df:
    :param percent:正样本相对于负样本的倍数
    :return:
    """
    data1 = df[df['label'] != 0]
    data0 = df[df['label'] == 0]

    index = np.random.randint(len(data0), size=percent * len(data1))

    lower_data0 = data0.iloc[list(index)]
    return pd.concat([lower_data0, data1])


if __name__ == '__main__':
    """
       对于获取完的线上训练集进行数据预处理
       生成标签的格式
    """
    # train_data = pd.read_csv("../recall_list/train_data.csv")
    # train_data = reduce_mem_usage(train_data)
    # train_data = train_data.fillna(0)  # 给label为NaN进行补充值
    # print(train_data[0: 5])
    # data0 = train_data[train_data['label'] == 0]
    # print(data0[0:5])

    """
    对于1到15天的用户日志进行召回的过程，预测他们在16号到底会选择什么商品作
    然后与16天的数据形成线上训练集
    """
    # # 测试1到15召回的东西
    recall_dfs = recall_list_online_function()

    print(recall_dfs.shape)
    print(recall_dfs.iloc[0:5])
