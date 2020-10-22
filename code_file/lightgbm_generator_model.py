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

"""
利用运行好的模型对于测试集召回的结果进行重排的过程
"""
# 载入模型
gbm = joblib.load("../online_feature_data/online_train_data")

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
y_pred = gbm.predict(data, num_iteration=gbm.best_iteration)  # 输出的概率形式


test_data['pred_prob'] = y_pred
"""
利用输出的概率大小开始推荐的过程
首先要按照userID进行分组, 然后在对每个组内的prob进行排序，选择前50个。。。。
"""




