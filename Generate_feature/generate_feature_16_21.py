from code_file.utils import reduce_mem_usage
import numpy as np
import pandas as pd
from tqdm import tqdm

# 载入原始的数据集 即7到21号的。。
data = pd.read_csv("../data/df_behavior_train.csv")  # userID,behavior,timestap,itemID,date,day
data = reduce_mem_usage(data)
# 载入用户的数据
user = pd.read_csv("../data/user.csv", header=None)
user.columns = ['userID', 'sex', 'age', 'ability']
user = reduce_mem_usage(user)

# 载入商品的数据
item = pd.read_csv("../data/df_item.csv")  # categoryID,shopID,brandID,itemID
item = reduce_mem_usage(item)

# 合并数据集
# 合并商品以及用户的数据集进行统计过程
"""
['userID', 'behavior', 'timestap', 'itemID', 'sex', 'age',
       'ability', 'categoryID', 'shopID', 'brandID']
"""
data = pd.merge(left=data, right=user, on=['userID'], how='left')
data = pd.merge(left=data, right=item, on=['itemID'], how='left')

data = data.drop(['date', 'day'], axis=1)
print(data.shape)
print(data.columns)


max_len = 1e-8
for userID, hist in tqdm(data.groupby('userID')):
    if max_len < len(hist):
        max_len = len(hist)
print(max_len) # 285
# items = data[['itemID']].drop_duplicates('itemID')

# print(items.head())
# items = items["itemID"].tolist()
# kk = 1
# for i in items:
#     print(i)
#     print("\n")
#     if kk == 5:
#         break
#     kk = kk + 1


def get_feature_dict(datas):
    """
    为1到15号的数据
    获取离散型特征的索引位置
    进行后续的embedding_lookup
    :return:
    """
    train_data = datas
    feature_dict = {}  # 记录下离散特征位置的字典
    total_feature = 1  # 留下第0个位置给多值离散特征进行填充0

    re_feature_dict = {}  # 用于恢复数据的使用
    for col in train_data.columns:
        if col in ['userID']:
            continue
        else:
            """
            位于离散特征时进行处理
            首先统计该特征的所有不同的值
            然后记录下不同值得位置
            """
            unique_val = train_data[col].unique()
            feature_dict[col] = dict(zip(unique_val, range(total_feature,
                                                           len(unique_val) + total_feature)))
            re_feature_dict[col] = dict(zip(range(total_feature, len(unique_val) + total_feature), unique_val))

            total_feature = total_feature + len(unique_val)
    """
    所以字典的形式就是：
    {itemID：{211：0， 534：1，.......}}。。嵌套两个字典
    """
    return feature_dict, re_feature_dict, train_data

# _, tot, _ = get_feature_dict(data)
#
# print(tot)  # 2847368
