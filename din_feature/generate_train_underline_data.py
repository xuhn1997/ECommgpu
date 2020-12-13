"""
此文件是用来生成构造DIN的线下训练集
一共有只有1号到16号的数据
我们采用1号到15号的数据构建线下训练集
训练的时候不需要进行召回，直接构建
所以要将用户的行为的历史行为按照时间进行排序，，，，
"""
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import numpy as np
import pandas as pd
import random

ITEM_COUNT = 4318202  # 商品的总数目

df = pd.read_csv("../data/df_behavior_train.csv")  # 这是1号到15号的数据
"""
关键部分以用户进行分组并将用户按照时间进行排序
"""
df = df.groupby('userID').apply(lambda x: x.sort_values('date')).reset_index(drop=True)
"""
userID  behavior  timestap   itemID   date  day
(7598193, 6)
"""
print(df[0:30])
print(df.shape)


# 开始构建训练集， 但是后面要和用户特征以及商品特征进行concat

def gen_negative(pos_list):
    """
    构建负样本的函数
    :param pos_list: 这个是用户的行为历史集合
    :return:
    """
    neg = pos_list[0]
    # while neg in pos_list:
    #     neg = random.randint(0, ITEM_COUNT-1) # 使用randint时范围是0到指定的位置，包含这这个位置
    neg_list = []
    for i in range(len(pos_list)):
        while neg in pos_list:
            neg = random.randint(0, ITEM_COUNT - 1)

        neg_list.append(neg)
        #  在恢复neg的值继续搜索
        neg = pos_list[0]

    return neg_list  # 最后获得负样本的集合，长度和用户行为的历史集合相当，但是里面的每个item不在用户的历史行为的集合里面


random.seed(1024)
train_set = []
for userID, hist in df.groupby('userID'):
    """
    将用户为指标进行分组...
    """
    pos_list = hist['itemID'].tolist()  # 获取用户的行为历史商品集合，并封装成list

    # 获取用户行为的负样本商品集合，长度是和用户的历史行为商品集合一样
    neg_list = gen_negative(pos_list)

    if len(pos_list) >= 2:
        """
        大于等于2个才能构建
        """
        for i in range(1, len(pos_list)):
            hist_list = pos_list[:i]
            # i 从 1到 len(pos_list) -1
            train_set.append((userID, hist_list, pos_list[i], 1))
            train_set.append((userID, hist_list, neg_list[i], 0))


# 然后对于创建好的训练集进行打乱顺序
random.shuffle(train_set)

# 最后将线下训练集保存好
train_set_file = open("../DIN_SAVE/underline_train_set.txt", 'w')
train_set_file.write(str(train_set))
train_set_file.close()

print("save underline train dataset successfully........")


