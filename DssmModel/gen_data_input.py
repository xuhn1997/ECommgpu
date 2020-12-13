"""
此文件为DSSM的数据预处理
生成输入的训练集以及测试集
['userID', 'behavior', 'timestap', 'itemID', 'sex', 'age', 'ability',
       'categoryID', 'shopID', 'brandID', 'itemID_count', 'categoryID_count',
       'shopID_count', 'brandID_count', 'itemID_sum', 'categoryID_sum',
       'shopID_sum', 'brandID_sum', 'categoryID_median', 'categoryID_std',
       'categoryID_skw', 'itemID_median', 'itemID_std', 'itemID_skw',
       'itemID_tosex_count', 'itemID_toage_count', 'itemID_toability_count',
       'rank', 'rank_percent', 'itemnum_oncat', 'shopnum_oncat',
       'brandnum_oncat', 'user_to_categoryID_count', 'user_to_shopID_count',
       'user_to_brandID_count', 'user_to_categoryID_sum', 'user_to_shopID_sum',
       'user_to_brandID_sum', 'user_to_categoryID_count_21',
       'user_to_shopID_count_21', 'user_to_brandID_count_21',
       'user_to_categoryID_sum_21', 'user_to_shopID_sum_21',
       'user_to_brandID_sum_21', 'sex_to_categoryID_sum', 'sex_to_brandID_sum',
       'sex_to_itemID_sum', 'sex_to_shopID_sum']
"""

import os
import sys

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tqdm import tqdm

from Generate_feature.train_data_deal import get_feature_dict, data
from Generate_feature.feature_classify import CONTINUE_COLS
import random
import pickle as pck
import copy

"""
这个DSSM的是为线下召回。。利用1到15天的数据召回用户可能在16号的使用的商品，从而
为下面的DIN的细排生成线下测试集，经过DIN之后就可以进行top50的筛选
"""

MAX_LEN = 174  # 最大的历史长度


def get_train_data_index():
    """
    此函数将训练集的离散值转化成对应的特征索引
    :return:
    """
    feature_dict, total_continue_feature, train_data = get_feature_dict(data)

    # 复制一下数据集
    train_data_index = train_data.copy()

    # 开始转化的过程
    print("开始进行转化的过程........")
    for col in train_data_index.columns:
        if col in CONTINUE_COLS:
            train_data_index[col] = train_data_index[col].map(feature_dict[col])

    return train_data_index  # 返回其索引即可


# train_data_index = get_train_data_index()  # (7598193, 48)
# print(train_data_index.columns)
#
"""
开始构建训练集以及测试集
"""


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
    # 方便在求历史的类别，店铺时进行映射
    category_ids = data['categoryID'].unique()
    shop_ids = data['shopID'].unique()
    brand_ids = data['brandID'].unique()

    train_set = []
    test_set = []
    print("开始进行......")
    kkk = 1
    for userID, hist in tqdm(data.groupby('userID')):
        post_item_list = hist['itemID'].tolist()
        post_category_list = hist['categoryID'].tolist()
        post_shop_list = hist['shopID'].tolist()
        post_brand_list = hist['brandID'].tolist()

        if negsample > 0:
            """
            获取负样本的集合
            """
            candidate_set_item = list(set(item_ids) - set(post_item_list))
            candidate_set_shop = list(set(shop_ids) - set(post_shop_list))
            candidate_set_category = list(set(category_ids) - set(post_category_list))
            candidate_set_brand = list(set(brand_ids) - set(post_brand_list))

            neg_list_item = np.random.choice(candidate_set_item, size=len(post_item_list) * negsample, replace=True)
            neg_list_shop = np.random.choice(candidate_set_shop, size=len(post_shop_list) * negsample, replace=True)
            neg_list_brand = np.random.choice(candidate_set_brand, size=len(post_brand_list) * negsample, replace=True)
            neg_list_category = np.random.choice(candidate_set_category, size=len(post_category_list) * negsample,
                                                 replace=True)

        """
        开始构建DSSM的数据集
        """
        hist = hist.reset_index(drop=True)
        k = 1
        hist = np.array(hist)
        indexs = hist.shape[0]
        print(indexs)
        for index in tqdm(range(indexs)):
            if index == indexs - 1:
                """
                构建测试集
                """
                index_list = hist[index, :]
                # index_list = hist.loc[index].tolist()
                index_list = index_list.tolist()
                index_list.append(post_item_list[0:k])  # 添加商品的历史行为集合，以及其历史的长度
                index_list.append(post_shop_list[0:k])  # shop
                index_list.append(post_brand_list[0:k])  # brand
                index_list.append(post_category_list[0:k])  # category
                index_list.append(k)

                """
                添加正样本
                """
                index_list.append(1)

                test_set.append(index_list)
                # continue
                break
            elif index != 0:
                index_list = hist[index, :].tolist()

                index_list.append(post_item_list[0:k])  # 添加商品的历史行为集合，以及其历史的长度
                index_list.append(post_shop_list[0:k])  # shop
                index_list.append(post_brand_list[0:k])  # brand
                index_list.append(post_category_list[0:k])  # category

                index_list.append(k)
                """
                添加正样本
                """
                index_list.append(1)
                train_set.append(index_list)

                """
                添加负样本
                """
                for negi in range(negsample):
                    # 进行替换
                    """
                    注意这里涉及到list的深浅拷贝问题！！！
                    直接赋值会修改原来的值！！！
                    """
                    temp = copy.deepcopy(index_list)
                    temp[3] = neg_list_item[k * negsample + negi]
                    temp[7] = neg_list_category[k * negsample + negi]
                    temp[8] = neg_list_shop[k * negsample + negi]
                    temp[9] = neg_list_brand[k * negsample + negi]

                    temp[-1] = 0

                    train_set.append(temp)
                k = k + 1

        # kkk = kkk+1
        # if kkk == 3:
        #     break
    print("结束......")
    random.shuffle(train_set)
    random.shuffle(test_set)

    return train_set, test_set  # 注意里面的整数类型已经变为浮点类型48 + 5 + 1/48+5


# train_set1, test_set1 = get_data_set(data=train_data_index, negsample=2)
# print(train_set1)
# for line in train_set1:
#     print(line)
#     print("**********")
#     print('\n')
#
# with open('../data/train_set.pkl', 'wb') as f:
#     pck.dump(train_set1, f, pck.HIGHEST_PROTOCOL)
#
# with open('../data/test_set.pkl', 'wb') as f1:
#     pck.dump(test_set1, f1, pck.HIGHEST_PROTOCOL)


def gen_model_input(train_set):
    """
    构建训练集的输入。。。
    ['userID', 'behavior', 'timestap', 'itemID', 'sex', 'age', 'ability',
       'categoryID', 'shopID', 'brandID', 'itemID_count', 'categoryID_count',
       'shopID_count', 'brandID_count', 'itemID_sum', 'categoryID_sum',
       'shopID_sum', 'brandID_sum', 'categoryID_median', 'categoryID_std',
       'categoryID_skw', 'itemID_median', 'itemID_std', 'itemID_skw',
       'itemID_tosex_count', 'itemID_toage_count', 'itemID_toability_count',
       'rank', 'rank_percent', 'itemnum_oncat', 'shopnum_oncat',
       'brandnum_oncat', 'user_to_categoryID_count', 'user_to_shopID_count',
       'user_to_brandID_count', 'user_to_categoryID_sum', 'user_to_shopID_sum',
       'user_to_brandID_sum', 'user_to_categoryID_count_21',
       'user_to_shopID_count_21', 'user_to_brandID_count_21',
       'user_to_categoryID_sum_21', 'user_to_shopID_sum_21',
       'user_to_brandID_sum_21', 'sex_to_categoryID_sum', 'sex_to_brandID_sum',
       'sex_to_itemID_sum', 'sex_to_shopID_sum']
    :param train_set:
    :return:
    """
    train_model_input = {}
    item_feature_index = [i for i in range(7, 32)]
    item_feature_index.append(3)
    # item_continue_index = [18, 19, 20, 21, 22, 23, 28]  # 连续值的位置

    #  现在来连接user的特征
    user_multiply_feature = [
        48, 49, 50, 51
    ]
    user_hist_len_label = [52, 53]
    all_feature = [i for i in range(54)]
    user_feature_index = list(set(all_feature) - set(item_feature_index))
    users_list = []
    user_item_list = []
    user_shop_list = []
    user_brand_list = []
    user_category_list = []

    user_hist_len = []
    train_label = []

    item_list = []
    print("开始最后一步......")
    for line in tqdm(train_set):
        user_temp_list = []

        # t1 = []
        # t2 = []
        temp_list = []
        for i in item_feature_index:
            temp_list.append(line[i])
        item_list.append(temp_list)

        for j in user_feature_index:
            if j in user_multiply_feature:
                # continue
                if j == 48:
                    user_item_list.append(line[j])
                if j == 49:
                    user_shop_list.append(line[j])
                if j == 50:
                    user_brand_list.append(line[j])
                if j == 51:
                    user_category_list.append(line[j])
            elif j in user_hist_len_label:
                # continue
                if j == 52:
                    user_hist_len.append(int(line[j]))
                if j == 53:
                    train_label.append(int(line[j]))
            else:
                user_temp_list.append(int(line[j]))

        users_list.append(user_temp_list)

    train_model_input['users_list'] = np.array(users_list)

    # 要对历史行为进行嵌入
    user_item = pad_sequences(user_item_list, maxlen=MAX_LEN, padding='post', truncating='post', value=0)
    train_model_input['user_item_list'] = user_item

    user_shop = pad_sequences(user_shop_list, maxlen=MAX_LEN, padding='post', truncating='post', value=0)
    train_model_input['user_shop_list'] = user_shop

    user_brand = pad_sequences(user_brand_list, maxlen=MAX_LEN, padding='post', truncating='post', value=0)
    train_model_input['user_brand_list'] = user_brand

    user_category = pad_sequences(user_category_list, maxlen=MAX_LEN, padding='post', truncating='post', value=0)
    train_model_input['user_category_list'] = user_category

    train_model_input['user_hist_len'] = np.array(user_hist_len)

    train_model_input['item_feature'] = np.array(item_list)  # 后面要看第二维度有几个？？

    # 测试集没有label
    # if train_label is not None:
    train_label = np.array(train_label)  # 维度是多少？

    return train_model_input, train_label


