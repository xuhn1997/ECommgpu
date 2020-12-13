"""
此文件是将得到训练集的特征进行分类
比如分为连续值，离散值，以及忽略值
"""
import pandas as pd
import numpy as np

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

CONTINUE_COLS = [
    'behavior', 'timestap', 'itemID', 'sex', 'ability',
    'categoryID', 'shopID', 'brandID',
    'itemID_count', 'categoryID_count', 'shopID_count', 'brandID_count',
    'itemID_sum', 'categoryID_sum', 'shopID_sum', 'brandID_sum',
    'itemID_to_sex_count', 'itemID_to_age_count', 'itemID_to_ability_count',
    'rank', 'itemnum_oncat', 'shopnum_oncat', 'brandnum_oncat',
    'user_to_categoryID_count', 'user_to_shopID_count', 'user_to_brandID_count',
    'user_to_categoryID_sum', 'user_to_shopID_sum', 'user_to_brandID_sum',
    'user_to_categoryID_count_21', 'user_to_shopID_count_21', 'user_to_brandID_count_21',
    'user_to_categoryID_sum_21', 'user_to_shopID_sum_21', 'user_to_brandID_sum_21',
    'sex_to_categoryID_sum', 'sex_to_brandID_sum', 'sex_to_itemID_sum', 'sex_to_shopID_sum',
]

NUMBERS_COLS = [
    'age', 'categoryID_median', 'categoryID_std', 'categoryID_skw',
    'itemID_median', 'itemID_std', 'itemID_skw',
    'rank_percent'
]

IGNORE_COLS = [
    'userID'
]

# 还有就是对于DSSM模型时要分类好用户部分的特征，以及商品的特征

USER_FEATURE = [
    'userID', 'behavior', 'timestap', 'sex', 'ability', 'age',
    'user_to_categoryID_count', 'user_to_shopID_count', 'user_to_brandID_count',
    'user_to_categoryID_sum', 'user_to_shopID_sum', 'user_to_brandID_sum',
    'user_to_categoryID_count_21', 'user_to_shopID_count_21', 'user_to_brandID_count_21',
    'user_to_categoryID_sum_21', 'user_to_shopID_sum_21', 'user_to_brandID_sum_21',
    'sex_to_categoryID_sum', 'sex_to_brandID_sum', 'sex_to_itemID_sum', 'sex_to_shopID_sum',

]
# print(len(USER_FEATURE))

ITEM_FEATURE = [
    'itemID', 'categoryID', 'shopID', 'brandID',
    'itemID_count', 'categoryID_count', 'shopID_count', 'brandID_count',
    'itemID_sum', 'categoryID_sum', 'shopID_sum', 'brandID_sum',
    # 'itemID_to_sex_count', 'itemID_to_age_count', 'itemID_to_ability_count',
    'rank', 'itemnum_oncat', 'shopnum_oncat', 'brandnum_oncat',
    'categoryID_median', 'categoryID_std', 'categoryID_skw',
    'itemID_median', 'itemID_std', 'itemID_skw',
    'rank_percent'
]

# print(len(ITEM_FEATURE))
#
# print(len(USER_FEATURE) + len(ITEM_FEATURE))  # 48
