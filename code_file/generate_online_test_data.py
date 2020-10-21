"""
此文件是生成线上测试集
利用的是1号到16号的用户日志
生成的特征要与训练集的特征保持一致
现在生成的测试集是要经过决策树网络的
"""
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import pandas as pd
import numpy as np
from code_file.utils import reduce_mem_usage
from code_file.model import recommend_users, reshape_to_dataframe, load_npz
from collections import defaultdict

# 先看一下训练集运用了哪些特征
"""
总的特征就是
userID, itemID, sim, label, sex, age, ability, categoryID, shopID, brandID
category_median category_std item_median item_std
"""
"""
所以第一步就是要对于1到16号的用户进行召回得到userID, itemID, sim特征，如召回需要输入用户的行为日志[userID, itemID, behavior, day]
第二步就是获取用户行为日志之后要进行召回。。。

"""


def get_users_log_function():
    """
    获取1号到16号的用户行为日志
    :return:
    """
    filename = "../data/df_behavior.csv"
    df_behavior = pd.read_csv(filename)
    df_behavior = reduce_mem_usage(df_behavior)

    matrix = df_behavior[['userID', 'itemID', 'behavior', 'day']].values  # 使用values时会将数据集转化成浮点数的形式
    users_log = dict()

    for row in matrix:
        users_log.setdefault(int(row[0]), [])
        users_log[int(row[0])].append((int(row[1]), int(row[2]), int(row[3])))

    return users_log


def run_recall_users_function():
    """
    利用1到16好的用户日志进行召回的过程。。。
    :return:
    """
    # 获取用户行为的日志
    users_log = get_users_log_function()

    # 获取物品协同矩阵
    filename_command_matrix = "../commonMatrix_iuf/common_matrix.npz"
    mat = load_npz(filename_command_matrix)

    # 召回的格式为{userID:{j, sim}, {j1, sim1}}, ......
    recommends = recommend_users(N=300, user_logs=users_log, mat=mat)

    # 将召回的数据集转化成dataframe的格式
    recall_df = reshape_to_dataframe(recommends)
    recall_df.to_csv("../recall_list/test_data.csv", index=False)

    print("save recall_test file successfully........")

    return recall_df  # 最后返回的格式就是[userID, itemID, sim]


# 然后召回的数据集与原始的item以及user以及两个的数据统计特征
def get_all_test_feature():
    """
    获取测试集的全部特征
    :return:
    """
    # 获取召回集所得的特征
    recall_test = pd.read_csv("../recall_list/test_data.csv")
    user_data = pd.read_csv("../data/user.csv", header=None)
    user_data.columns = ['userID', 'sex', 'age', 'ability']

    item_data = pd.read_csv("../data/df_item.csv")  # categoryID, shopID, brandID, itemID
    recall_test = pd.merge(left=recall_test, right=user_data, on=['userID'], how='left')
    recall_test = pd.merge(left=recall_test, right=item_data, on=['itemID'], how='left')

    # 再获取itemID，以及categoryID的统计特征
    # temp.columns = ['categoryID', 'category_median', 'category_std']
    category_feature = pd.read_csv('../statistics_feature/category_higher.csv')
    #  temp_itemID.columns = ['itemID', 'item_median', 'item_std']
    item_feature = pd.read_csv('../statistics_feature/item.higher.csv')

    recall_test = pd.merge(left=recall_test, right=category_feature, on=['categoryID'], how='left')
    recall_test = pd.merge(left=recall_test, right=item_feature, on=['itemID'], how='left')

    # 再跟统计的四个特征进行合并
    item_ID_feature = pd.read_csv('../statistics_feature/itemID_count.csv')
    category_ID_feature = pd.read_csv("../statistics_feature/categoryID_count.csv")
    shop_ID_feature = pd.read_csv("../statistics_feature/shopID_count.csv")
    brand_ID_feature = pd.read_csv("../statistics_feature/brandID_count.csv")

    recall_test = pd.merge(left=recall_test, right=item_ID_feature, on=["itemID"], how="left")
    recall_test = pd.merge(left=recall_test, right=category_ID_feature, on=["categoryID"], how="left")
    recall_test = pd.merge(left=recall_test, right=shop_ID_feature, on=["shopID"], how="left")
    recall_test = pd.merge(left=recall_test, right=brand_ID_feature, on=["brandID"], how="left")

    return recall_test


"""
以上之后的总的特征就是
userID, itemID, sim, sex, age, ability, categoryID, shopID, brandID category_median
category_std item_median item_std, itemID_sum, categoryID_sum, shopID_sum, brandID_sum
"""

if __name__ == '__main__':
    # 获取召回的数据集
    # recall_tests = run_recall_users_function()
    # print(recall_tests.shape)
    # print(recall_tests.iloc[0:5])

    # 保存线上测试集
    online_test_data = get_all_test_feature()
    online_test_data.to_csv("../online_feature_data/online_test_data", index=False)
    print("save online_test_data successfully..........")



