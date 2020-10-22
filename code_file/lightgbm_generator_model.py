"""
lightgbm_model模型的加载与保存
gbm = joblib.load('dkal.pkl')
gbm.predict()......
"""

import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from code_file.model import code_start

"""
利用运行好的模型对于测试集召回的结果进行重排的过程
"""
# 载入模型
gbm = joblib.load('../lightgbm_model_save/lightgbm_model.pkl')

# 获取测试集的数据集
"""
userID, itemID, sim, sex, age, ability, categoryID, shopID, brandID category_median
category_std item_median item_std, itemID_sum, categoryID_sum, shopID_sum, brandID_sum
"""
test_data = pd.read_csv("../online_feature_data/online_test_data")
# 删选出需要的特征
features = [x for x in test_data.columns
            if x not in ['userID', 'itemID']]

data = test_data[features]

# 开始进行排序
y_pred = gbm.predict(data, num_iteration=gbm.best_iteration_)  # 输出的概率形式

test_data['pred_prob'] = y_pred
"""
利用输出的概率大小开始推荐的过程
首先要按照userID进行分组, 然后在对每个组内的prob进行排序，选择前50个。。。。
"""
# 对于userID进行分组之后对于pred_prod进行排序
# df1 = df.groupby('A', sort=False).apply(lambda x:x.sort_values('B', ascending=True)).reset_index(drop=True)

test_data1 = test_data.groupby('userID', sort=False).apply(
    lambda x: x.sort_values('pred_prob', ascending=False)).reset_index(drop=True)

# 然后从测试集中获取用户集
"""
users = df['A'].unique()
print(users)

users = users.tolist()
print(users)
"""
recall_list = dict()

users = test_data1['userID'].unique()
users = users.tolist()

for user in users:
    recall_list.setdefault(int(user), [])
    # tmp = df1[df1['A'] == user]
    tmp = test_data1[test_data1['userID'] == user]
    # recall_list[int(user)].append((tmp['itemID'], tmp['pred_prob']))

    if len(recall_list[int(user)]) < 50:
        recall_list[int(user)].append((tmp['itemID'], tmp['pred_prob']))

"""
对于不够五十商品的进行冷启动的操作
"""
for user in users:
    user = int(user)
    if len(recall_list[user]) < 50:
        """
        进行冷启动
        """
        tmp_length = len(recall_list[user])
        numbers = 50 - tmp_length
        item_lists = code_start(numbers)  # 返回的是一个链表的形式[(item, 0), (item1, 0)]
        recall_list[user].extend(item_lists)  # 将其进行合并

# 最后将召回的字典保存在txt文件中

recall_result_file = open("../recall_list/result_recall.txt", 'w+')
recall_result_file.write(str(recall_list))
recall_result_file.close()

print("save successfully........")

"""
加载召回的过程
fr = open("../recall_list/result_recall.txt", 'r+)
dic = eval(fr.read())
fr.close()
"""

