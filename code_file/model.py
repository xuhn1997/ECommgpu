import datetime
import math
import sys
from operator import itemgetter
import time
import os
# import scipy
from scipy.sparse import *
import pandas as pd
import numpy as np
import math
from collections import defaultdict

"""
  模型中的冷启动要注意的一点是：
  比如线上测试时，召回1到16天所有用户的所有商品推荐，所以在召回阶段不用进行冷启动操作，因为这里面的用户都会有用户行为日志
  但是给用户推荐500个有时候并没有500个，可能只有5个，所以在最后排序后还要给他们剩余进行冷启动
  在最后
"""


def calculate_matrix(mat, user_logs, alpha=0.5):
    """
    此处用来计算协同过滤矩阵
    :param alpha: 计算时间上下文的协同矩阵的衰减参数
    :param mat: 稀疏矩阵
    :param user_logs:
    :return:
    """
    count = 0
    N = defaultdict(int)  # 统计
    for u, item_logs in user_logs:
        count += 1
        if count % 1000 == 0:
            print("The %d " % count + 'users are finished')
        # 获取用户行为日志
        for i1, i2, i3 in item_logs:
            N[i1] += 1  # 统计使用该物品的用户数目
            # 其中i1为物品ID， i2为用户对该物品感兴趣的程度
            # i3 为行为的时间戳
            for j1, j2, j3 in item_logs:
                if j1 == i1:
                    continue
                # mat[i1, j1] += 1. / (1 + alpha * abs(i3 - j3))
                # mat[j1, i1] += 1. / (1 + alpha * abs(i3 - j3))

                # 考虑到活跃用对于物品的相似度应该小于不活跃的用户, 对于活跃用户的惩罚
                mat[i1, j1] += 1. / ((1 + alpha * abs((i3 - j3))) * (math.log2(1 + len(item_logs))))
                mat[j1, i1] += 1. / ((1 + alpha * abs((i3 - j3))) * (math.log2(1 + len(item_logs))))

    # 开始计算分母的部分
    mats = mat.nonzero()  # 获取非零的元组
    for i, j in zip(mats[0], mats[1]):
        mat[i, j] = mat[i, j] / math.sqrt(N[i] * N[j])

    return mat


def code_start_user(user, user_logs, N):
    """
    不够500个的用户，进行添加，要不能与历史的感兴趣的相同类别
    :param user:该用户
    :param user_logs:用户行为日志注意是几号到几号的用户日志
    :param item_cate:用户行为商品类别
    :param N:推荐几个
    :return:
    """
    item_users = dict()
    filename_item = "../data/df_item_sort.csv"
    item_pd = pd.read_csv(filename_item)
    numbers = 0
    # 获取用户行为过的商品
    user_items = user_logs[user]  # 获取到用户行为的日志，为一个链表形式([j1, j2, j3], [j2, j3, j4].....)
    items = [x for x, _, _ in user_items]  # 获取用户行为过商品的结合
    # cates = set()
    # for itemID in items:
    #     cates.add(item_cate[itemID])  # 获取用户行为过的商品类型集合
    for item, item_count in item_pd[['itemID', 'itemCount']].values:
        if item in items:
            continue
        item_users.setdefault(item, 0)
        item_users[item] = item_count
        numbers = numbers + 1
        if numbers == N:
            # 到达指定的个数则取出退出循环
            return item_users


def code_start(N):
    """
    解决用户冷启动问题，给用户推荐N个商品
    :param N:
    :return:
    返回一个链表比较好。。。。
    """
    item_users = []
    filename_item = "../data/df_item_sort.csv"
    item_pd = pd.read_csv(filename_item)
    numbers = 0
    for item, item_count in item_pd[['itemID', 'itemCount']].values:
        # item_users.setdefault(item, 0)
        # item_users[item] = 0
        item_users.append((item, 0.))
        numbers = numbers + 1
        if numbers == N:
            # 到达指定的个数则取出退出循环
            return item_users


def recommend(user, N, user_logs, mat, beta=1.0):
    """
    给每一个用户推荐N个商品
    :param item_cate: 物品以及其类别商品,形状为{itemID: cate1}.....
    :param beta:时间衰减函数
    :param mat: 物品协同矩阵
    :param user_logs: 用户行为日志注意是几号的到几号的用户行为日志？？？线上测试集是1到16天，线上训练集是1到15天
    :param user:
    :param N:
    :return:
    """
    t_now = 23
    recommends = dict()

    # if user not in user_logs.keys():
    #     # 如果用户没有行为日志时，采取冷启动方法
    #     return code_start(N)
    # else:
    # 获取用户行为过的商品
    user_items = user_logs[user]  # 获取到用户行为的日志，为一个链表形式([j1, j2, j3].....)
    items = [x for x, _, _ in user_items]  # 获取用户行为过商品的结合
    # cates = set()
    # for itemID in items:
    #     cates.add(item_cate[itemID])  # 获取用户行为过的商品类型集合
    for i1, i2, i3 in user_items:
        # 其中i1为物品ID， i2为用户对该物品感兴趣的程度
        # i3 为行为的时间戳
        # if len(user_items) == 0:
        #     return code_start(N)
        # else:
        # 正常推荐
        # mat_user = mat.getrow(user)  # 得到一个(1*n)的行向量的稀疏矩阵
        mat_user = mat.getrow(i1)  # 注意是取当前用户行为过的物品i的相关的商品j
        # mat_user_nonzero = mat_user.nonzero() #获得非零元素的素养
        mat_user = zip(mat_user.indices, mat_user.data)  # 将这个行向量的列索引和它的值映射成一个元组
        for j, sim in sorted(mat_user, key=itemgetter(1), reverse=True):  # 将这个元组进行一个排序
            if j in items:
                continue
            # if item_cate[j] in cates: #同类型的物品也是可以
            #     continue
            recommends.setdefault(j, 0)
            # 还要考虑用户行为过的时间戳，还有用户行为的程度大小
            recommends[j] += sim * (1 + beta * abs(t_now - i3)) * i2
    if len(recommends.keys()) >= N:
        # 如果推荐的个数够N个商品的话则进行正常推荐
        return dict(sorted(recommends.items(), key=itemgetter(1), reverse=True)[:N])
    else:
        # 但是推荐的商品个数要是不够N个的话则用热门商品进行补充
        # residue = N - len(recommends.keys())  # 获取还有多少个进行补充

        """
          冷启动的部分注意
          召回阶段不需要进行冷启动。。。
          排序最后阶段阶段再需要冷启动
        """
        # residue_dict = code_start_user(user, user_logs, item_cate, residue)

        recommends_dict = dict(sorted(recommends.items(), key=itemgetter(1), reverse=True))
        # recommends_dict.update(residue_dict)

        return recommends_dict


def recommend_users(N, user_logs, mat):
    """
    为一个用户集合的每一个用户推荐N个商品
    :param item_cates: 物品与类别的映射
    :param mat:协同过滤矩阵
    :param user_logs:用户行为日志表,要将分组之后的合并成一个总的字典
    :param users:
    :param N:
    :return:
    """
    recommends = dict()
    # for user in users:
    #     # user_recommends = list(recommend(user, N, user_logs, mat, item_cates).keys())
    #     user_recommends = recommend(user, N, user_logs, mat)
    #     recommends.setdefault(user, dict())
    #     recommends[user] = user_recommends

    for user, user_items in user_logs.items():
        user_recommends = recommend(user, N, user_logs, mat)
        recommends.setdefault(user, dict())
        recommends[user] = user_recommends

    return recommends


# def recommend_users_cumpute_recall(users, N, user_logs, item_cates, mat):
#     """
#     为一个用户集合的每一个用户推荐N个商品
#     :param item_cates: 物品与类别的映射
#     :param mat:协同过滤矩阵
#     :param user_logs:用户行为日志表,要将分组之后的合并成一个总的字典
#     :param users:
#     :param N:
#     :return:
#     """
#     recommends = dict()
#     for user in users:
#         user_recommends = list(recommend(user, N, user_logs, mat, item_cates).keys())
#         # user_recommends = recommend(user, N, user_logs, mat, item_cates)
#         recommends.setdefault(user, [])
#         recommends[user] = user_recommends
#
#     return recommends


def generate_pairs(recommends):
    """
    将生成的{{user:j, apio}}生成dataframe格式
    :param recommends:
    :return:
    """
    result = []
    for user, user_items in recommends.items():
        for j, sim in user_items.items():
            result.append([user, j, sim])

    return result


def reshape_to_dataframe(result):
    """
    讲推荐出来的候选表转化成dataframe格式
    :param result:
    :return:
    """
    recall_list = generate_pairs(result)
    recall_list = pd.DataFrame(recall_list)

    recall_list.columns = ['userID', 'itemID', 'sim']  # 其中sim为未来感兴趣的程度

    return recall_list
