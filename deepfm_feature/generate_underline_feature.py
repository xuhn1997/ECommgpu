"""
此文件是生成deepFM的线下训练集的
步骤就是首先要召回1号到14号的数据，然后与15号的标签构建成线下训练集

"""
import sys
import os

from scipy.sparse import load_npz

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import numpy as np
import pandas as pd
from collections import defaultdict
from code_file.utils import reduce_mem_usage
from code_file.model import recommend_users, reshape_to_dataframe


def get_users_log_underline():
    """
    此函数是用来获取用户在1号到14号的行为日志：
    即userID, itemID, behavior, day
    :return:
    """
    df = pd.read_csv("../data/df_behavior_train.csv")
    """
    再筛选数据集，应该是<21时的数据
    """
    df_train_data = df[df['day'] < 21]
    # 然后开始获取用户行为日志,格式为{userID, (itemID, behavior,day)}
    temp = df_train_data[['userID', 'itemID', 'behavior', 'day']].values

    users_log = dict()
    for row in temp:
        users_log.setdefault(int(row[0]), [])
        users_log[int(row[0])].append((int(row[1]), int(row[2]), int(row[3])))

    return users_log


def get_underline_labels():
    """
    此函数是获取线下训练集的标签
    则应该是返回15号的数据
    也就是21号
    格式就是['userID', 'itemID', 'behavior']
    :return:
    """
    df = pd.read_csv("../data/df_behavior_train.csv")
    df = df[df['day'] == 21]

    train_labels = df[['userID', 'itemID', 'behavior']]
    train_labels = reduce_mem_usage(train_labels)

    return train_labels


def recall_underline():
    """
    开始对于1号到14号的用户日志进行召回的过程
    :return:
    """
    # 首先先获取用户的行为日志
    user_logs = get_users_log_underline()
    # 获取物品协同矩阵
    file_matrix = "../commonMatrix_iuf/common_matrix.npz"
    mat = load_npz(file_matrix)

    # 现在进行找回的操作
    recommends = recommend_users(200, user_logs, mat)  # 返回的格式
    """
    以上的召回的返回的格式就是{userID:{itemid:sim}, {item1:sim1}, .....}
    """
    # 将召回的集合转化成dataframe格式
    recommends_df = reshape_to_dataframe(recommends)  # 格式就是['userID', 'itemID', 'sim']

    # 保存下来召回集、
    recommends_df.to_csv("../recall_list/train_underline_recall_data.csv", index=False)
    print("保存好线下召回集成功.............")


def get_underline_recall_data():
    """
    将召回的集合与标签集合进行结合
    :return:
    """
    # 获取召回数据
    recommends_df = pd.read_csv("../recall_list/train_underline_recall_data.csv")

    # 获取标签结合
    labels_df = get_underline_labels()

    # 然后就进行结合
    recall_data_df = pd.merge(left=recommends_df, right=labels_df, on=[['userID', 'itemID']],
                              how='left')
    recall_data_df = recall_data_df.rename(columns={'behavior': 'label'})
    """
    此时以上的数据格式就是
    userID, itemID, sim, label
    """
    return recall_data_df


def label_deal(x):
    """
    对于label的值进行梳理
    :param x:
    :return:
    """
    if x == 0:
        return 0
    else:
        return 1


def down_sample(df, percent=10):
    """
    对于召回的数据进行采用
    数据格式为userID, itemID, sim, label
    :param df:
    :param percent:
    :return:
    """
    data1 = df[df['label'] != 0]
    data0 = df[df['label'] == 0]

    # 关键部分，取得负采样的样本序号
    index = np.random.randint(len(data0), size=percent * len(data1))

    lower_data0 = data0.iloc[list(index)]
    # 然后将负采样的数据与真实的数据进行结合
    return pd.concat([lower_data0, data1])


def get_all_underline_feature():
    """
    此函数就是获取线下训练集的全部特征
    由于是要对于DeepFM的特征所以只要合并user以及df_item的文件即可
    :return:
    """
    # 获取user的数据
    users_df = pd.read_csv("../data/user.csv", header=None)
    users_df.columns = ['userID', 'sex', 'age', 'ability']

    # 获取item的数据集
    items_df = pd.read_csv("../data/df_item.csv")
    """
    以上item的数据格式就是itemID, categoryID, shopID, brandID
    """
    # 获取找回
    # pd.read_csv("../recall_list/")
    recall_df = get_underline_recall_data()
    # 首先要对于label中出现的NaN值进行补充0
    recall_df = recall_df.fillna(0)
    # 要对于召回集进行采用之后才合并
    recall_df = down_sample(recall_df)
    # 进行合并的过程
    recall_all = pd.merge(left=recall_df, right=users_df, on=['userID'], how='left')
    recall_all = pd.merge(left=recall_all, right=items_df, on=['itemID'], how='left')

    """
    之后的总的特征就是userID, itemID, sim, label，'sex', 'age', 
    'ability'， categoryID, shopID, brandID
    """
    # 最后还要对于给label进行一个值的转化
    recall_all['label'] = recall_all['label'].apply(label_deal)

    return recall_all


if __name__ == '__main__':
    # 先保存好召回集, 注意保存好了要注释掉
    recall_underline()

