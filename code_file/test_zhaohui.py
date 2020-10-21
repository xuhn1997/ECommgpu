import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from code_file.model import recommend_users, reshape_to_dataframe, generate_pairs
from scipy.sparse import *
import pandas as pd
from code_file.utils import get_logs_from
from collections import defaultdict


def precision(recommends_, tests):
    """计算Precision
    :param recommends_: dict
        给用户推荐的商品，recommends为一个dict，格式为 { userID : 推荐的物品 }
    :param tests: dict
        测试集，同样为一个dict，格式为 { userID : 实际发生事务的物品 }
    :return: float
        Precision
    """
    n_union = 0.
    user_sum = 0.
    for user_id, items in recommends_.items():
        recommend_set = set(items)
        test_set1 = set(tests[user_id])
        n_union += len(recommend_set & test_set1)
        user_sum += len(test_set1)

    return n_union / user_sum


def item_cates_function():
    """
    生成物品与类别的映射文件
    :return:
    {itemID1: cate1}
    {itemID2: cate1}
    """
    filename = "../data/df_item.csv"
    items = pd.read_csv(filename)
    item_cates = dict()

    for itemID, categoryID in items[['itemID', 'categoryID']].values:
        item_cates.setdefault(itemID, 0)
        item_cates[itemID] = categoryID

    return item_cates


# 获取测试集函数
def test_users():
    """
    获取测试集函数
    该测试集是线下测试，讲最后一天的用户行为计算召回，准确率
    :return:
    """
    filename_user = "../data/user.csv"
    users = pd.read_csv(filename_user, header=None)
    users.columns = ['userID', 'gender', 'age', 'purchaseLevel']

    filename_test_data = "../data/df_behavior_test.csv"
    test_data = pd.read_csv(filename_test_data)
    test_data_group = test_data.groupby(['userID'])

    user_items = dict()
    user_sets = set()  # 测试用户的用户集合
    for key, value in test_data_group:
        user_sets.add(key)
        user_items.setdefault(key, set())
        # user_items[key].add(int(value['itemID']))
        for i in value['itemID']:
            user_items[key].add(i)

    # 不需要下面的代码只是线上测试，所以需要16号的用户集合即可，以便计算召回率以及准确率
    # for userid in users['userID'].values:
    #     user_sets.add(userid)
    #     if userid not in user_items.keys():
    #         user_items.setdefault(userid, set())

    return user_sets, user_items


def recall():
    # 获取物品协同过滤矩阵
    filename_commend_matrix = "../commonMatrix_iuf/common_matrix.npz"
    mat = load_npz(filename_commend_matrix)

    # 获取商品与类别的映射
    item_cates = item_cates_function()

    # path = "../full_logs/"
    # # 获取用户的行为日志，不是用这个作为推荐用户的用日志
    # users_log = None
    # for name in os.listdir('../full_logs'):
    #     if name[-3:] == 'txt':
    #         if users_log is None:
    #
    #             users_log = get_logs_from(path + name)
    #         else:
    #             users_log.update(users_log)

    # 注意获取的用户集不只是测试集里面的用户，应该是讲测试集里面的数据与用户文件进行合并才是！！！！

    user_sets, user_items = test_users()

    # recommends_train = recommend_users_cumpute_recall(users=user_sets, N=500, user_logs=users_log, mat=mat,
    #                                                   item_cates=item_cates)
    # # 返回值为{user:(1, 2, 3, 4 , 5)}，一个字典的形式
    # prediction = precision(recommends_=recommends_train, tests=user_items)  # 准确率0.53左右线下
    # print(prediction)

    # 此测试为线下预测，线上的话，要重新计算协同过滤矩阵进行重新预测
    # 获取daframe格式的候选表
    recommends = recommend_users(users=user_sets, N=500, user_logs=users_log, mat=mat,
                                 item_cates=item_cates)
    result = generate_pairs(recommends)

    recall_df = reshape_to_dataframe(result)
    recall_df.to_csv("../recall_list/generate_users_recall_list_novel_recall.csv", index=False)

    # f = open('../recall_list/generate_users_recall_list_novel_recall.txt', 'w')
    # f.write(str(recommends_train))
    # f.close()

    # f1 = open('../recall_list/novel_recall.txt', 'w')
    # f1.write(str(prediction))
    # f1.close()

    print("save recall users_lists successfully..................")


if __name__ == '__main__':
    recall()
