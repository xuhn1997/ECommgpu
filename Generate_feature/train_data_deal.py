"""
此文件是利用分类好的特征进行特征记录以进行
"""
import numpy as np
import pandas as pd
from tqdm import tqdm

from Generate_feature.feature_classify import IGNORE_COLS, NUMBERS_COLS, CONTINUE_COLS
from Generate_feature.generate_all_feature_7_21 import get_underline_7_21_all_feature
from code_file.utils import reduce_mem_usage

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

# """
# 这里面要获取用户的行为的日志
# """
# user_log = dict()
# max_user_log = 1e-8
# for user_id, hist in tqdm(data.groupby("userID")):
#     user_log.setdefault(user_id, set())
#
#     temp_log = set(hist['itemID'].tolist())
#
#     if len(temp_log) > max_user_log:
#         max_user_log = len(temp_log)
#
#     user_log[user_id] = temp_log
# print(max_user_log) # 102
# print("急急急经济")
# print(user_log[1084863])
# items = data[['itemID']].drop_duplicates('itemID')
# items = items["itemID"].tolist()
# items = np.array(items)
# print(items.shape)
# items = np.reshape(items, (1, -1))
# print(items[0])
# print(data['itemID'].unique())
#
# data = data.drop_duplicates('itemID')
# print(data.shape)
# print(data['itemID'])


def get_feature_dict(datas):
    """
    为1到15号的数据
    获取离散型特征的索引位置
    进行后续的embedding_lookup
    :return:
    """
    train_data = get_underline_7_21_all_feature(datas)
    feature_dict = {}  # 记录下离散特征位置的字典
    total_feature = 1  # 留下第0个位置给多值离散特征进行填充0

    for col in train_data.columns:
        if col in NUMBERS_COLS:
            continue
        elif col in IGNORE_COLS:
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
            total_feature = total_feature + len(unique_val)
    """
    所以字典的形式就是：
    {itemID：{211：0， 534：1，.......}}。。嵌套两个字典
    """
    return feature_dict, total_feature, train_data


#
# di, w, data1 = get_feature_dict(datas=data)
# print(di['itemID'])
# print(data1.columns)
# print(w)  # 2927842, 1140820
# train_data = get_underline_7_21_all_feature(data)
# mm = -1
# for _, df in train_data.groupby('userID'):
#     if mm < len(df):
#         mm = len(df)
#
# print(mm)  # 285, 174
