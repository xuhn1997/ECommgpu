import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from operator import itemgetter

import tensorflow as tf

import numpy as np
import pandas as pd
import pickle as pck
from tqdm import tqdm
from Generate_feature.generate_feature_16_21 import data, get_feature_dict
from DssmModel.gen_16_21_input import get_train_data_index

# with open('../data/train_set.pkl', 'rb') as f:
#     train_set = pck.load(f)
from DssmModel.gen_16_21_input import gen_model_input

with open('../data/dssm_user_embedding.pkl', 'rb') as f:
    uu = pck.load(f)

with open('../data/dssm_item_embedding.pkl', 'rb') as f:
    ii = pck.load(f)

print(uu.shape)
print(ii.shape)

"""
现在开始进行推荐商品
"""
# from DssmModel.gen_data_input import get_train_data_index


recommends = dict()

"""
注意此处要用测试集中的所有用户进行推荐
而且还要注意用户的顺序！！！！
"""
with open('../data/test_set_10_10.pkl', 'rb') as f:
    train_set = pck.load(f)

train_model_input, _ = gen_model_input(train_set=train_set)

user_temp = train_model_input['users_list']

# print(len(user_temp))
"""
注意这里面推荐的是商品的索引，不需要进行map了
"""
"""
这里面要获取用户的行为的日志
"""
user_log = dict()
max_user_log = 1e-8
for user_id, hist in tqdm(data.groupby("userID")):
    user_log.setdefault(user_id, set())

    temp_log = set(hist['itemID'].tolist())

    if len(temp_log) > max_user_log:
        max_user_log = len(temp_log)  # 注意这里面的是取得用户使用过商品的最大个数是？是set集合的不同！！！为210ge

    user_log[user_id] = temp_log
# print("急急急经济")
# print(user_log)
# 获取商品的序号
# 这里用的embedding的索引的位置
em_data = get_train_data_index()
items = em_data[['itemID']].drop_duplicates('itemID')
items = items["itemID"].tolist()
# 要和其总数进行成立一个map函数
# print(len(items))
items_dict = dict(zip(range(len(items)), items))
print(items_dict[0])
"""
利用fassi进行推荐
"""
import faiss

dim = 128
k = 300 + max_user_log  # 召回的个数是300个
index = faiss.IndexFlatIP(dim)

index.add(ii)

# 开始进行查找....
D, I = index.search(uu, k)

# print(I.shape)
"""
其中I为相似度矩阵,
里面的数值为50多万的商品embedding索引的索引
"""
# 获取好user部分的用户，进行合并转化成dataframe进行map的过程
users = user_temp[:, 0]

users = np.reshape(users, (-1, 1))

# 合并
recommends_np = np.concatenate((users, I), axis=1)

print(recommends_np.shape)
print(recommends_np[0:10])

recommends = pd.DataFrame(recommends_np, columns=[str(i + 1) for i in range(k + 1)])

print(recommends.shape)
print(recommends.head())

_, re_dict, _ = get_feature_dict(data)  # 获取恢复原始数据的map
for col in tqdm(recommends.columns):
    if col == '1':
        continue
    else:
        recommends[col] = recommends[col].map(items_dict)
        """
        还要将其商品de索引恢复成
        原来的商品号
        """

        recommends[col] = recommends[col].map(re_dict['itemID'])

recommends = np.array(recommends)  # 转化成numpy便于遍历
print("最后的结果是.....")
print(recommends.shape)
# print(recommends.head())

"""
下一步要进行去除用户的行为过
"""
user_recommends = dict()
for i in tqdm(range(recommends.shape[0])):
    user_recommends.setdefault(recommends[i, 0], [])

    recommends_log = recommends[i, 1:].tolist()

    ret_list = []

    kkk = 0
    for item in recommends_log:
        if item not in user_log[recommends[i, 0]]:
            ret_list.append(item)
            kkk = kkk + 1
            if kkk == 300:
                break

    user_recommends[recommends[i, 0]] = ret_list

with open('../DssmModel/DSSM_recommends_300.pkl', 'wb') as f:
    pck.dump(user_recommends, f, pck.HIGHEST_PROTOCOL)

print("保存成功........")
