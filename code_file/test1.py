import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
import scipy as scp

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


for i in range(10):
    print("你好")
    for j in range(8):
        print("xuhn")
        if j == 3:
            break
