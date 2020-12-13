import numpy as np

import pandas as pd
from tqdm import tqdm

# from DssmModel.gen_data_input import gen_model_input
import pickle as pck


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
        n_union += len(recommend_set & test_set)
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
