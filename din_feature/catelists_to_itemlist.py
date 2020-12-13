"""
此文件就是利用item文件生成item到物品相关的特征的映射
比如item对应的种类，item对应的品牌，。。。。等等。
"""
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import pandas as pd
import numpy as np
from sklearn import preprocessing

# 获取原来的数据

df = pd.read_csv("../data/df_item.csv")

print(df.shape)  # 商品数据的维度是(4318202, 4)
print(df.head())  # categoryID   shopID  brandID   itemID其中brandID为-1表示该商品品牌不知名。。。

# 现在对于除了itemID都做一次编码，也就是labelEncoding。。
le = preprocessing.LabelEncoder()
df['brandID'] = le.fit_transform(df['brandID'])  # 范围在0到298836
print(df['brandID'].min())
print(df['brandID'].max())

le1 = preprocessing.LabelEncoder()
df['shopID'] = le1.fit_transform(df['shopID'])  # 范围在0到924304
print(df['shopID'].min())
print(df['shopID'].max())

le2 = preprocessing.LabelEncoder()
df['categoryID'] = le2.fit_transform(df['categoryID'])  # 范围在0到9724
print(df['categoryID'].min())
print(df['categoryID'].max())

# print(df["itemID"].min())  # 0
# print(df["itemID"].max())  # 4318201
# print(df["itemID"].nunique())  # 4318202

"""
编码完成之后现在生成映射的过程。。。
"""
# 将item按照itemID进行排序
df = df.sort_values('itemID')
df = df.reset_index(drop=True)

print(df.head())

df.to_csv("../data/item_map_list.csv", index=False)