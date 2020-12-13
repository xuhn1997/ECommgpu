import numpy as np

import pandas as pd
from tqdm import tqdm

# from DssmModel.gen_data_input import gen_model_input
import pickle as pck

from Generate_feature.generate_feature_16_21 import data

all_items = data[['itemID']].drop_duplicates('itemID')

all_items = set(all_items["itemID"].tolist())

user_log1 = dict()
max_user_log = 1e-8
for user_id, hist in tqdm(data.groupby("userID")):
    user_log1.setdefault(user_id, set())

    temp_log = set(hist['itemID'].tolist())

    if len(temp_log) > max_user_log:
        max_user_log = len(temp_log)  # 注意这里面的是取得用户使用过商品的最大个数是？是set集合的不同！！！为210ge

    user_log1[user_id] = temp_log


def recall(recommends, tests):
    """
        计算Recall
        @param recommends:   给用户推荐的商品，recommends为一个dict，格式为 { userID : 推荐的物品 }
        @param tests:  测试集，同样为一个dict，格式为 { userID : 实际发生事务的物品 }
        @return: Recall
    """
    n_union = 0.
    recommend_sum = 0.
    for user_id, items in tqdm(recommends.items()):
        recommend_set = set(items)
        test_set = tests[user_id]
        temp_items = test_set ^ user_log1[user_id]
        # n_union += len(recommend_set & test_set)
        n_union += len(all_items & temp_items) # 2811547.0
        recommend_sum += len(recommend_set)  # 注意分母，比赛中的是预测的
    print("命中的商品数: %s\n" % str(n_union))
    print("召回的总数：%s\n" % str(recommend_sum))
    return n_union / recommend_sum


"""
现在获取真实的用户行为在22号
"""
# 获取用户出现在19到21号的用户
# import pickle as pck

with open('../DssmModel/DSSM_recommends_300.pkl', 'rb') as f:
    pred_recommends = pck.load(f)

# 获取预测集中的用户
recall_users = pred_recommends.keys()

real_data = pd.read_csv("../data/df_behavior.csv")
real_data = real_data[real_data['day'] == 22]
# 获取真实数据的user集合
real_users = real_data['userID'].tolist()
user_log = dict()
for user_id, hist in tqdm(real_data.groupby("userID")):
    if user_id in recall_users:
        user_log.setdefault(user_id, set())

        temp_log = set(hist['itemID'].tolist())

        user_log[user_id] = temp_log

"""
获取预测的用户行为
"""

# 下面求预测中的用户与真实的用户的交集
# users = list(set(real_users) & set(real_users))
users = user_log.keys()
# 将预测的值转成dataframe
df1 = pd.DataFrame.from_dict(pred_recommends)

# 然后取出交集中的用户
df2 = df1[users]

# 然后再转化成dataframe回来

recall_temp = df2.to_dict(orient='list')

recall_rate = recall(recall_temp, user_log)

print("召回率为: %s" % str(recall_rate))
