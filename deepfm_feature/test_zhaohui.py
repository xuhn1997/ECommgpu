"""此文件计算召回率。。。"""

import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import pandas as pd
import numpy as np

from code_file.utils import get_file_from_txt


def recall(recommends, tests):
    """
        计算Recall
        @param recommends:   给用户推荐的商品，recommends为一个dict，格式为 { userID : 推荐的物品 }
        @param tests:  测试集，同样为一个dict，格式为 { userID : 实际发生事务的物品 }
        @return: Recall
    """
    n_union = 0.
    recommend_sum = 0.
    for user_id, items in recommends.items():
        recommend_set = items
        test_set = tests[user_id]
        n_union += len(recommend_set & test_set)
        recommend_sum += len(test_set)  # 注意分母，比赛中的是真实

    return n_union / recommend_sum


# 生成一号到15号的行为日志
# 格式为userID:[]
def get_user_logs():
    path1 = "../data/df_behavior_train.csv"
    df_data = pd.read_csv(path1)
    df_users = df_data['userID'].unique()
    df_users = df_users.tolist()
    users_log1 = dict()
    matrix = df_data[['userID', 'itemID']].values
    for row1 in matrix:
        users_log1.setdefault(int(row1[0]), set())
        users_log1[int(row1[0])].add(int(row1[1]))
    return users_log1, df_users


if __name__ == '__main__':
    path = "../recall_list/result_recall_deepfm.txt"
    # 获取用户的历史行为日志
    users_log, df_train_users = get_user_logs()  # 回去用户在1到15号的用户集合
    """
       以下的数据格式是{userID:[(217, 1.0), (156235, 0.0)]}
    """
    recall_underline = get_file_from_txt(path)

    """要获取第十六天的数据"""
    file_test_data = "../data/df_behavior_test.csv"

    df_test = pd.read_csv(file_test_data)
    test_users = df_test['userID'].unique()
    test_users = test_users.tolist()

    test_matrix = df_test[['userID', 'itemID']].values  # 注意这个之后就是一个浮点数
    test_recall = dict()
    # 要获得用户1到16天的历史行为

    for row in test_matrix:

        if int(row[0]) in df_train_users:  # 防止出现keyerror
            test_recall.setdefault(int(row[0]), set())
            if int(row[1]) not in users_log[int(row[0])]:
                # 去除历史的行为日志
                test_recall[int(row[0])].add(int(row[1]))

    users = set()

    recall_reault = dict()
    # 开始计算召回率
    for userID, items_value_list in recall_underline.items():
        if userID in test_users:
            users.add(userID)  # 存储的是两个共有的用户
            recall_reault.setdefault(userID, set())
            for i, j in items_value_list:
                recall_reault[userID].add(i)

    print("开始计算召回率.....")
    numbers = recall(recall_reault, test_recall)
    print(numbers)  # 0.3535492938130298
    # n_union = 0.
    # recommend_sum = 0.
    # for user, item_set in test_recall.items():
    #     if user in users:
    #         recommend_set = recall_reault[user]  # 同下是set类型。。。。
    #         test_set = test_recall[user]
    #         n_union += len(recommend_set & test_set)
    #         recommend_sum += len(recommend_set)
    #
    # print("召回率为：")
    # print(n_union / recommend_sum)
