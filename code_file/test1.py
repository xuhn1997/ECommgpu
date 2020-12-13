from operator import itemgetter

import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
import scipy
import pickle as pck

#
# with open('../DssmModel/DSSM_recommends_300.pkl', 'rb') as f:
#     re = pck.load(f)
#
# print(re[1084863])
#
#
# df1 = pd.DataFrame.from_dict(re)
#
# print(df1.shape)
# print(df1.head())
# df2 = df1[[901447, 1084863]]
# print(df2.head())
#
# temp = df2.to_dict(orient = 'list')
# # print(df2[1084863].tolist())
# print(temp)
print("召回率为: %s" % str(8))
"""
   测试稀疏矩阵的一些用法
"""

# mat = lil_matrix((10, 10), dtype=float)
#
# mat[9, 8] = 1
# mat[8, 9] = 1
#
# mat1 = lil_matrix((10, 10), dtype=float)
# mat1[8, 9] = 10
# mat1[0, 7] = 99
#
# mat22 = mat + mat1
# print(mat22)

# nnz = mat.nonzero()
# for i, j in zip(nnz[0], nnz[1]):
#     # print(i)
#     # print(j)
#     # print("-----")
#     mat[i, j] = mat[i, j] / 5
# print(nnz)
# print(mat)

# filename = "../data/df_behavior_test.csv"
# test_data = pd.read_csv(filename)
# test_data_group = test_data.groupby(['userID'])
#
# num = 0
# items = []
# for key, value in test_data_group:
#     print(key)
#     print(value['itemID'])
#     for i in value['itemID']:
#         items.append(i)
#
#     # # print(value['itemID'].shape)
#     # value = value.reset_index()
#     # print(value)
#     print("----------")
#
#     num = num + 1
#     if num == 6:
#         break
#
# print(items)


# 将用户日志中的item变为整型的类型
# df = pd.read_csv("../data/df_behavior.csv")
# df[[column]] = df[[column]].astype(type)
# import pandas as pd
# df = pd.DataFrame([['a', 1, 'c'], ['a', 3, 'a'], ['a', 2, 'b'],
#                    ['c', 3, 'a'], ['c', 2, 'b'], ['c', 1, 'c'],
#                    ['b', 2, 'b'], ['b', 3, 'a'], ['b',1, 'c']], columns=['A', 'B', 'C'])
#
# print(df)
#
# df1 = df.groupby('A', sort=False).apply(lambda x:x.sort_values('B', ascending=True)).reset_index(drop=True)
#
# print(df1)
#
# users = df['A'].unique()
# print(users)
#
# users = users.tolist()
# print(users)
#
# # it = 1
# for user in users:
#     # print(user)
#     tmp = df1[df1['A'] == user]
#     print(len(tmp))
#     print(tmp['B'])
#     break

#
# for i in range(10):
#     print("你好")
#     for j in range(8):
#         print("xuhn")
#         if j == 3:
#             break

import tensorflow as tf
import numpy as np
import pandas as pd
import faiss
# tmp = np.random.randint(100, size=[10, 4, 1])
# tmp = tf.reshape(tmp, (10, 4, 1))
#
# tmp1 = tf.reduce_sum(tmp, axis=2)

# print(tmp1.shape)
# import random
# import copy
#
# b = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# b1 = [100, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 23]
# print(set(b1))
# a = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
# print(a)
# l2 = sorted(set(a),key=a.index)
# # print(set(a))
# print(l2)
# for i in set(a):
#     print(i)
# b = np.reshape(b, (-1, 1))
# b1 = np.reshape(b1, (-1, 1))
# b = b.astype(np.float32)
# b1 = b1.astype(np.float32)
# dim = 1
#
# k = 3
# index = faiss.IndexFlatIP(dim)
#
# index.add(b1)
#
# D, I = index.search(b, k)
#
# print(I)
# print(D)
# tt = zip(a, b[0, :])

# temp = sorted(tt, key=itemgetter(1), reverse=True)[:4] # 将商品的相似度进行降序排序
# print(temp)
# train_set = []
#
# # temp = b
# train_set.append(b)
# print(train_set)
# # print(b)
#
# temp = copy.deepcopy(b)
#
# temp[-1] = 100
# train_set.append(temp)
# print(train_set)
# print(b)
# 测试mask
# b = tf.reshape(b, (9, 1))
# mask = tf.sequence_mask(b, 10, dtype=tf.float32)
# print(mask.shape)
# print(mask)
# b = np.reshape(b, (9, 1))
# a = np.reshape(a, (9, 1))
# c = {"6":a}
# c = {"dhajs": 89890}
# with open('../data/c.pkl', 'wb') as f:
#     pck.dump(c, f, pck.HIGHEST_PROTOCOL)

# fr = open('../data/a.pkl')
# inf = pickle.load(fr)
# fr.close()
#
# with open('../data/c.pkl', 'rb') as fr:
#     data = pck.load(fr)
# print(data)

# print(inf)


#
# print(b.shape)
# print(a.shape)
# # a = a.transpose(1, 0)
# # b = b.transpose(1, 0)
#
# print(a)
# print(b)
# randnum = np.random.randint(0,100)
# np.random.seed(randnum)
# np.random.shuffle(a)
# # a = a.transpose(1, 0)
# print(a)
# print(a.shape)
# np.random.seed(randnum)
# np.random.shuffle(b)
# # b = b.transpose(1, 0)
# print(b)
# print(b.shape)
