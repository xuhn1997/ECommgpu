import pandas as pd
import numpy as np

user = pd.read_csv("../data/user.csv", header=None)
user.columns = ['userID', 'sex', 'age', 'ability']
print(user.head())

users = list(set(user['userID']))

print(len(users))  # 1396951

filename1 = "../data/df_behavior_train.csv"
data = pd.read_csv(filename1)
print(data.head())
print(len(np.unique(data['userID'])))  # 272123

# 查看行为种item是否有不同时刻使用相同物品的情况
print(data.shape)  # 总的条数(7598193, 6)->(3524450, 6)
print(len(np.unique(data['itemID'])))  # 1189534条不同的商品

item_num = pd.read_csv("../data/item.csv", header=None)
print(item_num.shape)
print(len(np.unique(item_num[0])))  # 4318202个商品
CPU_NUMS = 4

user_groups = [users[i: i + len(users) // CPU_NUMS] for i in range(0, len(users), len(users) // CPU_NUMS)]
# 分成0到1000条，1000条到2000条，。。。。。。依次下去。。。
print(len(user_groups))  # 5组
# print(user_groups[3])

matrix = data[data['userID'].isin(user_groups[0])][['userID', 'itemID', 'behavior', 'timestap']].values  # 变成了了数组类型
print("-----")
print(matrix.shape)
print("*****")
#
# print(matrix.shape)
# print(matrix[0:10, :])
