"""
此文件是用来生成线下测试集的
全部是有1号到16号的数据
我们要做的是预测16号用户的行为数据
所以我们要利用1号到15号的数据召回，也就是获得用户可能在16号行为的商品，作为DIN的候选广告。。。
然后和1号到15号用户的行为的商品集合构建成测试集
"""

import sys
import os

from scipy.sparse import load_npz

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import numpy as np
import pandas as pd
import random

random.seed(1024)


# 所以用之前在用lightgbm时召回出来的1号到15号的数据

def get_din_underline_test_data():
    """
    构建线下测试集
    :return:
    """
    """
       userID,itemID,sim,label
    """
    df_recall = pd.read_csv("../recall_list/train_data.csv")
    df_recall = df_recall[['userID', 'itemID']]

    """
    后面要做一个映射
    {userID:item0, item1.......}
    """
    userID_re_items = dict()
    for userID, items in df_recall.groupby('userID'):
        userID_re_items.setdefault(userID, [])
        item_list = items['itemID'].tolist()
        userID_re_items[userID] = item_list

    # 现在获取1到15号的数据
    """
    userID,behavior,timestap,itemID,date,day
    """
    df_data = pd.read_csv("../data/df_behavior_train.csv")
    # 先将数据按照date进行排序
    df_data = df_data.groupby('userID').apply(lambda x: x.sort_values('date')).reset_index(drop=True)

    # 开始构建测试集，但是后面要和用户特征以及商品特征进行concat
    test_data_set = []
    for userID, hist in df_data.groupby('userID'):
        hist_list = hist['itemID'].tolist()
        for item in userID_re_items[userID]:
            test_data_set.append((userID, hist_list, item))

    random.shuffle(test_data_set)

    # 保存号测试集
    # 最后将线下训练集保存好
    test_set_file = open("../DIN_SAVE/underline_test_set.txt", 'w')
    test_set_file.write(str(test_data_set))
    test_set_file.close()

    print("save underline test dataset successfully........")


# get_din_underline_test_data()

df_data = pd.read_csv("../data/df_behavior_train.csv")
# df_recall = df_recall[['userID', 'itemID']]
print(df_data["userID"].nunique())
# print(df_recall[df_recall['userID'] == 5])

#

# # # 先将数据按照date进行排序
# # df_data = df_data.groupby('userID').apply(lambda x: x.sort_values('date')).reset_index(drop=True)
# #
# print(df_data[df_data['userID'] == 5])
#
# items_df = df_data[df_data['userID'] == 5]
# items = items_df['itemID'].tolist()
#
# # mat_user = mat.getrow(user)  # 得到一个(1*n)的行向量的稀疏矩阵
#
# # 1886644 2038025 3152753
# filename_commend_matrix = "../commonMatrix_iuf/common_matrix.npz"
# mat = load_npz(filename_commend_matrix)
# mat_user = mat.getrow(1886644)  # 注意是取当前用户行为过的物品i的相关的商品j
# # mat_user_nonzero = mat_user.nonzero() #获得非零元素的素养
# mat_user = zip(mat_user.indices, mat_user.data)  # 将这个行向量的列索引和它的值映射成一个元组
#
# for j, sim in mat_user:
#     # if j in items:
#     print(j)
#     print(">>>>>>>")
#     print(sim)
#
# print(mat_user)
