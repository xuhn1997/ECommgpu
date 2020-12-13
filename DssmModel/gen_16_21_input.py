import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm
from Generate_feature.generate_feature_16_21 import data, get_feature_dict
import random
import pickle as pck
import copy

MAX_LEN = 11  # 最大的历史长度


def get_train_data_index():
    """
    此函数将训练集的离散值转化成对应的特征索引
    :return:
    """
    feature_dict, _, train_data = get_feature_dict(data)

    # 复制一下数据集
    train_data_index = train_data.copy()

    # 开始转化的过程
    print("开始进行转化的过程........")
    for col in tqdm(train_data_index.columns):
        if col in ['userID']:
            continue
        else:
            train_data_index[col] = train_data_index[col].map(feature_dict[col])

    return train_data_index  # 返回其索引即可


# train_data_index = get_train_data_index()  # (7598193, 48)


# print(train_data_index.shape)


def get_data_set(data, negsample=0):
    """
    构建数据集
    :param data:整理的好的48个特征的数据集
    :param negsample:负样本的倍数
    :return:
    """
    # 首先将数据集按照时间顺序
    data.sort_values("timestap", inplace=True)  # 第二个参数表示替换原来的数据集
    item_ids = data["itemID"].unique()  # 获取数据集中商品的集合，注意这里之前之前已经将itemID进行labelencoder过，

    train_set = []
    test_set = []

    # train = dict()
    # test = dict()
    print("开始进行......")
    kkk = 1
    for userID, hist in tqdm(data.groupby('userID')):
        post_item_list = hist['itemID'].tolist()

        if negsample > 0:
            """
            获取负样本的集合
            """
            candidate_set_item = list(set(item_ids) - set(post_item_list))
            neg_list_item = np.random.choice(candidate_set_item, size=len(post_item_list) * negsample,
                                             replace=True)
        """
        开始构建DSSM的数据集
        """
        hist = hist.reset_index(drop=True)
        k = 1
        hist = np.array(hist)
        indexs = hist.shape[0]

        # 控制用户的行为数
        ll = 0
        for index in tqdm(range(indexs)):
            if index == indexs - 1:
                """
                构建测试集
                ['userID', 'behavior', 'timestap', 'itemID', 'sex', 'age',
                    'ability', 'categoryID', 'shopID', 'brandID']
                """
                test_set.append((userID, hist[index, 1], hist[index, 2], hist[index, 3],
                                 hist[index, 4], hist[index, 5], hist[index, 6],
                                 hist[index, 7], hist[index, 8], hist[index, 9],
                                 post_item_list[0:k], k, 1))
                # test_set.append((userID, post_item_list[0:k], post_item_list[index],k, 1))

                break
            elif index != 0:
                train_set.append((userID, hist[index, 1], hist[index, 2], hist[index, 3],
                                  hist[index, 4], hist[index, 5], hist[index, 6],
                                  hist[index, 7], hist[index, 8], hist[index, 9],
                                  post_item_list[0:k], k, 1))

                # train_set.append((userID, post_item_list[0:k], post_item_list[index], k, 1))
                """
                添加负样本
                """
                for negi in range(negsample):
                    # 进行替换
                    """
                    注意这里涉及到list的深浅拷贝问题！！！
                    直接赋值会修改原来的值！！！
                    ['userID', 'behavior', 'timestap', 'itemID', 'sex', 'age',
                    'ability', 'categoryID', 'shopID', 'brandID']
                    """
                    train_set.append((userID, hist[index, 1], hist[index, 2], neg_list_item[k * negsample + negi],
                                      hist[index, 4], hist[index, 5], hist[index, 6],
                                      hist[index, 7], hist[index, 8], hist[index, 9],
                                      post_item_list[0:k], k, 0))

                ll = ll + 1
                if ll == 10:
                    test_set.append((userID, hist[index+1, 1], hist[index+1, 2], hist[index+1, 3],
                                     hist[index+1, 4], hist[index+1, 5], hist[index+1, 6],
                                     hist[index+1, 7], hist[index+1, 8], hist[index+1, 9],
                                     post_item_list[0:k+1], k+1, 1))
                    break
                k = k + 1

        # kkk = kkk + 1
        # if kkk == 3:
        #     break
    print("结束......")
    random.shuffle(train_set)
    random.shuffle(test_set)

    return train_set, test_set


# train_set1, test_set1 = get_data_set(data=train_data_index, negsample=2)
# print(train_set1)
# for line in train_set1:
#     print(line)
#     print("**********")
#     print('\n')

# with open('../data/train_set_10_10.pkl', 'wb') as f:
#     pck.dump(train_set1, f, pck.HIGHEST_PROTOCOL)
#
# with open('../data/test_set_10_10.pkl', 'wb') as f1:
#     pck.dump(test_set1, f1, pck.HIGHEST_PROTOCOL)


def gen_model_input(train_set):
    """
    构建训练集的输入。。。
    ['userID', 'behavior', 'timestap', 'itemID', 'sex', 'age',
    'ability', 'categoryID', 'shopID', 'brandID', post_item, length, 1]
    :param train_set:
    :return:
    """
    train_model_input = {}
    item_feature_index = [3, 7, 8, 9]
    #  现在来连接user的特征
    user_feature_index = [0, 1, 2, 4, 5, 6]
    users_list = []  # 存储用户的单值离散特征....
    user_item_list = []  # 存储用户的历史行为

    user_hist_len = []  # 存储用户历史行为的长度
    train_label = []  # 存储训练的label值

    item_list = []  # 存储商品的特征
    print("开始最后一步......")
    for line in tqdm(train_set):
        user_temp_list = []
        temp_list = []
        for i in item_feature_index:
            temp_list.append(line[i])
        item_list.append(temp_list)

        user_item_list.append(line[10])
        user_hist_len.append(int(line[11]))
        train_label.append(int(line[12]))

        for j in user_feature_index:
            user_temp_list.append(int(line[j]))

        users_list.append(user_temp_list)

    train_model_input['users_list'] = np.array(users_list)

    # 要对历史行为进行嵌入
    user_item = pad_sequences(user_item_list, maxlen=MAX_LEN, padding='post', truncating='post', value=0)
    train_model_input['user_item_list'] = user_item

    train_model_input['user_hist_len'] = np.array(user_hist_len)

    train_model_input['item_feature'] = np.array(item_list)  # 后面要看第二维度有几个？？

    train_label = np.array(train_label)  # 维度是多少？

    return train_model_input, train_label
