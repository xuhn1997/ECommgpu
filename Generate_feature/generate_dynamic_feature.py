"""
此文件针对于原始数据集生成一些
静态的特征，比如商品的与行为交叉特征。。。。？
而用户与行为的交叉特征就是动态特征，因为用户user_behavior里面的是。。。。。？
我觉得不应该分动态和静态，应该是线下与线上之说，因为总的来说对于
不同的行为数据集特征值应该不一样的！！！！！！！！！！！！！！！
"""
from code_file.utils import reduce_mem_usage
import numpy as np
import pandas as pd

# 载入原始的数据集 即7到21号的。。
data = pd.read_csv("../data/df_behavior_train.csv")  # userID,behavior,timestap,itemID,date,day

"""
现在为了时间的，选择19到21号即可
"""
data = data[data['day'] >= 19]

data = reduce_mem_usage(data)

# 载入用户的数据
user = pd.read_csv("../data/user.csv", header=None)
user.columns = ['userID', 'sex', 'age', 'ability']
user = reduce_mem_usage(user)

# 载入商品的数据
item = pd.read_csv("../data/df_item.csv")  # categoryID,shopID,brandID,itemID
item = reduce_mem_usage(item)

# 合并商品以及用户的数据集进行统计过程
data = pd.merge(left=data, right=user, on=['userID'], how='left')
data = pd.merge(left=data, right=item, on=['itemID'], how='left')

# 首先是用户与商品的商品的交叉的特征，比如，userID，再shopID分组之类的行为的程度
for count_feature in ['categoryID', 'shopID', 'brandID']:
    data[['behavior', 'userID', count_feature]].groupby(['userID', count_feature], as_index=False) \
        .agg({"behavior": 'count'}).rename(columns={"behavior": 'user_to_' + str(count_feature) + '_count'}) \
        .to_csv("../Dynamic_feature_file/user_to_" + str(count_feature) + "_count.csv", index=False)

for count_feature in ['categoryID', 'shopID', 'brandID']:
    data[['behavior', 'userID', count_feature]].groupby(['userID', count_feature], as_index=False) \
        .agg({"behavior": 'sum'}).rename(columns={"behavior": 'user_to_' + str(count_feature) + '_sum'}) \
        .to_csv("../Dynamic_feature_file/user_to_" + str(count_feature) + "_sum.csv", index=False)

# 再求第21天的用户与商品的交叉的特征
train = data[data['day'] == 21]
for count_feature in ['categoryID', 'shopID', 'brandID']:
    train[['behavior', 'userID', count_feature]].groupby(['userID', count_feature], as_index=False) \
        .agg({"behavior": 'count'}).rename(columns={"behavior": 'user_to_' + str(count_feature) + '_count_21'}) \
        .to_csv("../Dynamic_feature_file/user_to_" + str(count_feature) + "_count_21.csv", index=False)

for count_feature in ['categoryID', 'shopID', 'brandID']:
    train[['behavior', 'userID', count_feature]].groupby(['userID', count_feature], as_index=False) \
        .agg({"behavior": 'sum'}).rename(columns={"behavior": 'user_to_' + str(count_feature) + '_sum_21'}) \
        .to_csv("../Dynamic_feature_file/user_to_" + str(count_feature) + "_sum_21.csv", index=False)

# 然后再生成一个sex与shopID,categoryID, itemID, brandID,的交叉特征。探讨性别的购买能力的影响
for count_feature in ["categoryID", "brandID", "itemID", "shopID"]:
    data[["behavior", "sex", count_feature]].groupby(['sex', count_feature], as_index=False) \
        .agg({"behavior": "sum"}).rename(columns={"behavior": 'sex_to_' + str(count_feature) + '_sum'}) \
        .to_csv("../Dynamic_feature_file/sex_to_" + str(count_feature) + "_sum.csv", index=False)
