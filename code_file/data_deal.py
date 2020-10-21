import numpy as np
import pandas as pd
import datetime
from code_file.utils import reduce_mem_usage
# import multiprocessing as mp
from sklearn import preprocessing
import collections

# 读取数据
filename_user = "../data/user.csv"
filename_item = "../data/item.csv"
filename_behavior = "../data/temp_0.csv"

df_user = pd.read_csv(filename_user, header=None)
df_user.columns = ['userID', 'gender', 'age', 'purchaseLevel']
# print(df_user.head())
print(df_user.shape)  # (1396951, 4)

"""
   要对物品的ID进行独特编码。。。。以便构造稀疏矩阵
"""
df_item = pd.read_csv(filename_item, header=None)
df_item.columns = ['itemID', 'categoryID', 'shopID', 'brandID']

le = preprocessing.LabelEncoder()
df_item['itemID_Encoding'] = le.fit_transform(df_item['itemID'])

# print(df_item.head(10))
print(df_item.shape)  # (4318202, 5)
print(df_item['itemID_Encoding'].min())  # 0
print(df_item['itemID_Encoding'].max())  # 4318202

df_behavior = pd.read_csv(filename_behavior, header=None)

# 在此处优化存储空间
df_behavior = reduce_mem_usage(df_behavior)
df_behavior = df_behavior.iloc[:, 1:5]
df_behavior.columns = ['userID', 'itemID', 'behavior', 'timestap']

df_behavior = df_behavior.merge(df_item, on='itemID', how='left')
df_behavior.drop(df_behavior[np.isnan(df_behavior['itemID_Encoding'])].index, inplace=True)  # 删除指定列含有NaN值
# 删除多余的列
df_behavior = df_behavior.drop(['itemID', 'categoryID', 'shopID', 'brandID'], axis=1)
df_behavior['itemID'] = df_behavior['itemID_Encoding']
df_behavior = df_behavior.drop(['itemID_Encoding'], axis=1)
print(df_behavior.head())  #
print(df_behavior['itemID'].min())  # 11
print(df_behavior['itemID'].max())  # 4318196
print(df_behavior.shape)  # (8047545, 4)

# 在对于df_item进行处理
df_item = df_item.drop(['itemID'], axis=1)
df_item['itemID'] = df_item['itemID_Encoding']
df_item = df_item.drop(['itemID_Encoding'], axis=1)
print(df_item.head())

# df_item.to_csv("../data/df_item.csv", index=False)  # 用于生成物品与其类别的映射
# 对于建立的数据集进行预处理
# df_user.drop_duplicates(inplace=True)
# df_user.reset_index(drop=True, inplace=True)
# print(df_user.shape)
#
# df_item.drop_duplicates(inplace=True)
# df_item.reset_index(drop=True, inplace=True)
# print(df_item.shape)

# 首先要最热门的物品统计出来，就是在用户行为这里动手, 解决用户的冷启动问题
item_statistic = df_behavior.groupby(['itemID'])[['userID']].count()
item_statistic.reset_index(inplace=True)
print(item_statistic.head())
item_statistic.columns = ['itemID', 'itemCount']

# 将行为统计的物品流量与物品的数据集进行合并
df_item = pd.merge(df_item, item_statistic)
# 帅选出流行度大于1的商品，但是商品的冷启动怎么解决？
df_item_selected = df_item[df_item['itemCount'] > 1]
df_item_selected.reset_index(drop=True, inplace=True)

print(df_item.shape)

print(df_item_selected.shape)

print(df_item_selected.head())

# 将其按照物品的流行度降序排序
df_item_sort = df_item_selected.sort_values(by=['itemCount'], ascending=False)
df_item_sort.reset_index(drop=True, inplace=True)
print(df_item_sort.head())

# 取出前500个
item_Top500 = list(df_item_sort.loc[0:499, 'itemID'])
len(item_Top500)
# 保存处理好的item文件
# df_item_sort.to_csv("../data/df_item_sort.csv", index=False)

# 对于时间处理的部分，刷选出1到15天的数据召回以及生成特征，16天的数据生成标签从而形成线上训练集。。。。再用1到16号的数据召回，以及生成特征从而形成
# 从而形成线上测试集。
print(df_behavior.head())
print(df_behavior.shape)

df_behavior['date'] = df_behavior['timestap'].apply(lambda x: datetime.datetime(2020, 9, 7) +
                                                              datetime.timedelta(seconds=x))
df_behavior['day'] = df_behavior['date'].dt.day

print(df_behavior.head())

# 开始划分数据集，分为训练集以及测试集
print(df_behavior['day'].max())
print(df_behavior['day'].min())

"""
   对于训练集的用户行为权重进行换算。。。。。。
   buy:4, cart:3, fav:2, pv:1
"""
df_behavior.loc[df_behavior['behavior'] == 'pv', 'behavior'] = 1
# loc[i, j]的意思就是对于i行j列的意思
df_behavior.loc[df_behavior['behavior'] == 'fav', 'behavior'] = 2
df_behavior.loc[df_behavior['behavior'] == 'cart', 'behavior'] = 3
df_behavior.loc[df_behavior['behavior'] == 'buy', 'behavior'] = 4

print(df_behavior.head())

# # 生成用户线上预测的训练集
df_behavior.to_csv("../data/df_behavior.csv", index=False)
#
df_behavior_train = df_behavior[df_behavior['day'] < 22]
df_behavior_test = df_behavior[df_behavior['day'] == 22]
print(df_behavior_train.shape)
print(df_behavior_test.shape)

print(len(np.unique(df_behavior_test['userID'])))

print(df_behavior_train.head())

# 保存处理好的训练集，以及前五百个最热门的商品已解决用户冷启动的问题
df_behavior_train.to_csv("../data/df_behavior_train.csv", index=False)
df_behavior_test.to_csv("../data/df_behavior_test.csv", index=False)
