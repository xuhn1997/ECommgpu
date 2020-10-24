import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import lightgbm as lgb
import pandas as pd
import numpy as np
from code_file.utils import get_logs_from, reduce_mem_usage
from collections import defaultdict

from sklearn.model_selection import train_test_split
from sklearn.externals import joblib

"""
此函数为筛选特特征，利用lightgbm进行筛选
"""

filename_recall = "../recall_list/train_data.csv"


def down_sample(df, percent=10):
    """
    对于线上训练集进行负采样的函数
    :param df:
    :param percent:正样本相对于负样本的倍数
    :return:
    """
    data1 = df[df['label'] != 0]
    data0 = df[df['label'] == 0]

    index = np.random.randint(len(data0), size=percent * len(data1))

    lower_data0 = data0.iloc[list(index)]
    return pd.concat([lower_data0, data1])


def transfer_label(x):
    """
    转化线上训练集的标签
    :param x:
    :return:
    """
    if x == 0:
        return 0
    else:
        return 1


def generate_all_feature():
    """
    获取线上训练集的总的特征
    也就是1到15天的数据特征
    :return:
    """
    # 先获取召回1到15天的数据集
    train_data = pd.read_csv(filename_recall)
    train_data = reduce_mem_usage(train_data)
    train_data = train_data.fillna(0)  # 给label为NaN进行补充值

    """
    以上train_data的格式为[userID, itemID, sim, label]
    解释：userID为用户，itemID为用户行为的商品，sim表示用户对于该商品的感兴趣的程度
    label表示的是在16号的时候该用户是否真的对这个商品使用了。。1到3表示使用了，0表示没有
    """
    # 进行负采样的过程
    recall_train = down_sample(train_data, 10)
    recall_train['label'] = recall_train['label'].apply(transfer_label)

    # 然后要跟原始的item文件以及user文件进行左连接
    user_data = pd.read_csv("../data/user.csv", header=None)
    user_data.columns = ['userID', 'sex', 'age', 'ability']

    item_data = pd.read_csv("../data/df_item.csv")  # 其中它的列名为categoryID,shopID,brandID,itemID

    recall_train = pd.merge(left=recall_train, right=user_data, on=['userID'], how='left', sort=False)
    recall_train = pd.merge(left=recall_train, right=item_data, on=['itemID'], how='left', sort=False)

    """
    以上进行合并之后的特征为：
    userID, itemID, sim, label, sex, age, ability, categoryID, shopID, brandID
    """

    """
    再把之前生成的统计统计特征进行连接
    分别为category_higher以及item.higher的统计
    """
    # 格式为['categoryID', 'category_median', 'category_std']
    category_feature = pd.read_csv('../statistics_feature/category_higher.csv')

    # 格式为['itemID', 'item_median', 'item_std']
    item_feature = pd.read_csv('../statistics_feature/item.higher.csv')

    recall_train = pd.merge(left=recall_train, right=category_feature, on=['categoryID'], how='left')
    recall_train = pd.merge(left=recall_train, right=item_feature, on=['itemID'], how='left')

    # 再跟统计的四个特征进行合并
    item_ID_feature = pd.read_csv('../statistics_feature/itemID_count.csv')
    category_ID_feature = pd.read_csv("../statistics_feature/categoryID_count.csv")
    shop_ID_feature = pd.read_csv("../statistics_feature/shopID_count.csv")
    brand_ID_feature = pd.read_csv("../statistics_feature/brandID_count.csv")

    recall_train = pd.merge(left=recall_train, right=item_ID_feature, on=["itemID"], how="left")
    recall_train = pd.merge(left=recall_train, right=category_ID_feature, on=["categoryID"], how="left")
    recall_train = pd.merge(left=recall_train, right=shop_ID_feature, on=["shopID"], how="left")
    recall_train = pd.merge(left=recall_train, right=brand_ID_feature, on=["brandID"], how="left")

    """
    以上之后的总的特征就是
    userID, itemID, sim, label, sex, age, ability, categoryID, shopID, brandID category_median
    category_std item_median item_std, itemID_sum, categoryID_sum, shopID_sum, brandID_sum
    """
    return recall_train


def lightgbm_model(recall_train):
    """
    进行light_gbm的函数
    :return:
    """
    """
    总的特征就是
    userID, itemID, sim, label, sex, age, ability, categoryID, shopID, brandID category_median
    category_std item_median item_std
    """
    # 定义评估的指标
    metric = 'binary_logloss'
    # 定义好lgbm模型
    model = lgb.LGBMClassifier(boosting_type='gbdt', n_estimators=30000,
                               objective='binary', learning_rate=0.1,
                               num_leaves=31, random_state=8082)

    # recall_train = generate_all_feature()
    # lightgbm需要分训练集以及验证集
    features = [x for x in recall_train.columns
                if x not in ['userID', 'itemID', 'label']]

    data = recall_train[features]
    y = recall_train['label']

    x_train, x_valid, y_train, y_valid = train_test_split(data, y,
                                                          test_size=0.1,
                                                          random_state=6666)
    model.fit(x_train, y_train,
              eval_metric=metric,
              eval_set=[(x_valid, y_valid)],
              early_stopping_rounds=30, verbose=1)
    # 保存好的训练好的模型
    joblib.dump(model, '../lightgbm_model_save/lightgbm_model.pkl')
    print("save successfully........")

    """
    最后生成测试集利用1到16生成测试集特征进行预测。。。。。。。还未完成。。。。。。
    """


# if __name__ == '__main__':
#     # 生成全部特征

# if __name__ == '__main__':
#     # 保存好线上训练集
#     online_train_data = generate_all_feature()
#     online_train_data.to_csv("../online_feature_data/online_train_data", index=False)
#     print("save successfully........")

if __name__ == '__main__':
    # 开始运行决策树模型
    # 获取线上训练集
    online_data = pd.read_csv("../online_feature_data/online_train_data")
    lightgbm_model(online_data)