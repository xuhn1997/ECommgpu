"""
此文件针对于原始数据集生成一些
静态的特征，比如商品的与行为交叉特征。。。。？
而用户与行为的交叉特征就是动态特征，因为用户user_behavior里面的是。。。。。？
我觉得不应该分动态和静态，应该是线下与线上之说，因为总的来说对于
不同的行为数据集特征值应该不一样的！！！！！！！！！！！！！！！
"""
from code_file.utils import reduce_mem_usage
import numpy as np
import pandas as pd


def get_underline_7_21_all_feature(train_data):
    """
    对于生成的交叉特征进行合并形成7到21天的线下训练集，针对于DSSM的模型训练
    :return:
    """
    # 先合并以商品为主生成的交叉特征
    """
    先捋好没有合并之前的特征的时候有
    userID,behavior,timestap,itemID,date,day,'sex', 'age', 'ability', categoryID,shopID,brandID
    然后去掉不要的特征，比如date， day
    """
    train_data = train_data.drop(['date', 'day'], axis=1)

    for count_feature in ["itemID", "categoryID", "shopID", "brandID"]:
        """
        userID,behavior,timestap,itemID,'sex', 'age', 'ability', categoryID,shopID,brandID
        (count_feature + '_count')*4
        """
        temp = pd.read_csv("../Static_feature_file/" + str(count_feature) + "_count.csv")
        train_data = pd.merge(left=train_data, right=temp, on=temp.columns[0], how='left', sort=False)

    for count_feature in ["itemID", "categoryID", "shopID", "brandID"]:
        """
        userID,behavior,timestap,itemID,'sex', 'age', 'ability', categoryID,shopID,brandID
        (count_feature + '_count')*4, (count_feature + '_sum')*4
        """
        temp = pd.read_csv("../Static_feature_file/" + str(count_feature) + "_sum.csv")
        train_data = pd.merge(left=train_data, right=temp, on=temp.columns[0], how='left', sort=False)

    for feature in ["categoryID", "itemID"]:
        """
        userID,behavior,timestap,itemID,'sex', 'age', 'ability', categoryID,shopID,brandID
        (count_feature + '_count')*4, (count_feature + '_sum')*4, 
        (feature + '_median', feature + '_std', feature + '_skw')*2,
        注意这里统计学特征我要把它变成连续值特征，所以对于NaN值我要把它填充0
        """
        temp = pd.read_csv("../Static_feature_file/" + str(feature) + "_higher.csv")
        train_data = pd.merge(left=train_data, right=temp, on=temp.columns[0], how='left', sort=False)

    train_data = train_data.fillna(0)  # 注意这里统计学特征我要把它变成连续值特征，所以对于NaN值我要把它填充0

    for count_feature in ['sex', 'age', 'ability']:
        """
        userID,behavior,timestap,itemID,'sex', 'age', 'ability', categoryID,shopID,brandID
        (count_feature + '_count')*4, (count_feature + '_sum')*4, 
        (feature + '_median', feature + '_std', feature + '_skw')*2,
        ( "itemID_to" + count_feature + '_count')*3
        """
        temp = pd.read_csv("../Dynamic_feature_file/item_to_" + str(count_feature) + "_count.csv")
        train_data = pd.merge(left=train_data, right=temp, on=list(temp.columns[0:2]), how='left', sort=False)

    # 然后连接的是商品等级的特征
    """
    userID,behavior,timestap,itemID,'sex', 'age', 'ability', categoryID,shopID,brandID
    (count_feature + '_count')*4, (count_feature + '_sum')*4, 
    (feature + '_median', feature + '_std', feature + '_skw')*2,
    ( "itemID_to" + count_feature + '_count')*3,
     "rank", "rank_percent"
    """
    temp1 = pd.read_csv("../Static_feature_file/item_rank.csv")
    train_data = pd.merge(left=train_data, right=temp1, on=['itemID'], how='left', sort=False)

    # categoryID在生成与itemID，shopID， brandID
    """
    userID,behavior,timestap,itemID,'sex', 'age', 'ability', categoryID,shopID,brandID
    (count_feature + '_count')*4, (count_feature + '_sum')*4, 
    (feature + '_median', feature + '_std', feature + '_skw')*2,
    ( "itemID_to" + count_feature + '_count')*3,
     "rank", "rank_percent",
    itemnum_oncat,shopnum_oncat,brandnum_oncat
    """
    temp2 = pd.read_csv("../Static_feature_file/category_lower.csv")
    train_data = pd.merge(left=train_data, right=temp2, on=['categoryID'], how='left', sort=False)

    # 最后合并的是以用户为主的交叉特征
    for count_feature in ['categoryID', 'shopID', 'brandID']:
        """
        userID,behavior,timestap,itemID,'sex', 'age', 'ability', categoryID,shopID,brandID
        (count_feature + '_count')*4, (count_feature + '_sum')*4, 
        (feature + '_median', feature + '_std', feature + '_skw')*2,
        ( "itemID_to" + count_feature + '_count')*3,
        "rank", "rank_percent",
        itemnum_oncat,shopnum_oncat,brandnum_oncat,
        ('user_to_' + str(count_feature) + '_count')*3
        """
        temp = pd.read_csv("../Dynamic_feature_file/user_to_" + str(count_feature) + "_count.csv")
        train_data = pd.merge(left=train_data, right=temp, on=list(temp.columns[0:2]), how='left', sort=False)

    for count_feature in ['categoryID', 'shopID', 'brandID']:
        """
        userID,behavior,timestap,itemID,'sex', 'age', 'ability', categoryID,shopID,brandID
        (count_feature + '_count')*4, (count_feature + '_sum')*4, 
        (feature + '_median', feature + '_std', feature + '_skw')*2,
        ( "itemID_to" + count_feature + '_count')*3,
        "rank", "rank_percent",
        itemnum_oncat,shopnum_oncat,brandnum_oncat,
        ('user_to_' + str(count_feature) + '_count')*3,
        ('user_to_' + str(count_feature) + '_sum')*3,
        """
        temp = pd.read_csv("../Dynamic_feature_file/user_to_" + str(count_feature) + "_sum.csv")
        train_data = pd.merge(left=train_data, right=temp, on=list(temp.columns[0:2]), how='left', sort=False)

    for count_feature in ['categoryID', 'shopID', 'brandID']:
        """
        userID,behavior,timestap,itemID,'sex', 'age', 'ability', categoryID,shopID,brandID
        (count_feature + '_count')*4, (count_feature + '_sum')*4, 
        (feature + '_median', feature + '_std', feature + '_skw')*2,
        ( "itemID_to" + count_feature + '_count')*3,
        "rank", "rank_percent",
        itemnum_oncat,shopnum_oncat,brandnum_oncat,
        ('user_to_' + str(count_feature) + '_count')*3,
        ('user_to_' + str(count_feature) + '_sum')*3,
        ('user_to_' + str(count_feature) + '_count_21')*3
        """
        temp = pd.read_csv("../Dynamic_feature_file/user_to_" + str(count_feature) + "_count_21.csv")
        train_data = pd.merge(left=train_data, right=temp, on=list(temp.columns[0:2]), how='left', sort=False)

    for count_feature in ['categoryID', 'shopID', 'brandID']:
        """
        userID,behavior,timestap,itemID,'sex', 'age', 'ability', categoryID,shopID,brandID
        (count_feature + '_count')*4, (count_feature + '_sum')*4, 
        (feature + '_median', feature + '_std', feature + '_skw')*2,
        ( "itemID_to" + count_feature + '_count')*3,
        "rank", "rank_percent",
        itemnum_oncat,shopnum_oncat,brandnum_oncat,
        ('user_to_' + str(count_feature) + '_count')*3,
        ('user_to_' + str(count_feature) + '_sum')*3,
        ('user_to_' + str(count_feature) + '_count_21')*3,
        ('user_to_' + str(count_feature) + '_sum_21')*3
        """
        temp = pd.read_csv("../Dynamic_feature_file/user_to_" + str(count_feature) + "_sum_21.csv")
        train_data = pd.merge(left=train_data, right=temp, on=list(temp.columns[0:2]), how='left', sort=False)

    for count_feature in ["categoryID", "brandID", "itemID", "shopID"]:
        """
        userID,behavior,timestap,itemID,'sex', 'age', 'ability', categoryID,shopID,brandID
        (count_feature + '_count')*4, (count_feature + '_sum')*4, 
        (feature + '_median', feature + '_std', feature + '_skw')*2,
        ( "itemID_to" + count_feature + '_count')*3,
        "rank", "rank_percent",
        itemnum_oncat,shopnum_oncat,brandnum_oncat,
        ('user_to_' + str(count_feature) + '_count')*3,
        ('user_to_' + str(count_feature) + '_sum')*3,
        ('user_to_' + str(count_feature) + '_count_21')*3,
        ('user_to_' + str(count_feature) + '_sum_21')*3,
        ('sex_to_' + str(count_feature) + '_sum')*4
        """
        temp = pd.read_csv("../Dynamic_feature_file/sex_to_" + str(count_feature) + "_sum.csv")
        train_data = pd.merge(left=train_data, right=temp, on=list(temp.columns[0:2]), how='left', sort=False)

    return train_data  # 48个特征
