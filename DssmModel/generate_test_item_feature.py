import numpy as np
import pandas as pd

from code_file.utils import reduce_mem_usage
from DssmModel.gen_data_input import get_train_data_index

"""
获取DSSM测试集的item部分的特征
   [ 'itemID',
       'categoryID', 'shopID', 'brandID', 'itemID_count', 'categoryID_count',
       'shopID_count', 'brandID_count', 'itemID_sum', 'categoryID_sum',
       'shopID_sum', 'brandID_sum', 'categoryID_median', 'categoryID_std',
       'categoryID_skw', 'itemID_median', 'itemID_std', 'itemID_skw',
       'rank', 'rank_percent', 'itemnum_oncat', 'shopnum_oncat',
       'brandnum_oncat', 
"""

item_feature = ['categoryID', 'shopID', 'brandID', 'itemID_count', 'categoryID_count',
                'shopID_count', 'brandID_count', 'itemID_sum', 'categoryID_sum',
                'shopID_sum', 'brandID_sum', 'categoryID_median', 'categoryID_std',
                'categoryID_skw', 'itemID_median', 'itemID_std', 'itemID_skw',
                'rank', 'rank_percent', 'itemnum_oncat', 'shopnum_oncat',
                'brandnum_oncat', 'itemID', ]

print(len(item_feature))


def get_item_data_index():
    """
    生成item部分的特征
    :return:
    """
    data = get_train_data_index()

    item_profile = data[item_feature].drop_duplicates('itemID')

    print(item_profile.shape) # (513140, 23)
    print(item_profile.columns)

    return item_profile


ii = get_item_data_index()
print(ii['itemID'])
