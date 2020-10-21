import pandas as pd
import numpy as np

from code_file.utils import reduce_mem_usage


def load_data(path):
    """
    获取数据，并将user文件以及item文件和行为文件进行合并
    :param path:
    :return:
    """
    user = reduce_mem_usage(pd.read_csv(path + 'user.csv', header=None))
    item = reduce_mem_usage(pd.read_csv(path + 'df_item.csv'))
    user.columns = ['userID', 'sex', 'age', 'ability']
    # 获取用户行为日志1到16天的
    data = pd.read_csv(path + 'df_behavior.csv')
    # 要将itemID的格式转化成int64类型的
    data['itemID'] = data['itemID'].astype('int64')

    data = reduce_mem_usage(data)

    # 开始合并到data中来
    data = pd.merge(left=data, right=item, on='itemID',
                    how='left')
    data = pd.merge(left=data, right=user, on='userID',
                    how='left')
    return user, item, data


#
# def generate_item_feature():
#     """
#     生成item的相关特征
#     :return:
#     """

filename_path = '../data/'
if __name__ == '__main__':

    # 获取item， user， 以及两个和行为合并的数据
    user_data, item_data, behavior_data = load_data(filename_path)
    """
      生成关于item的相关特征
    """
    for count_feature in ['itemID', 'shopID', 'categoryID', 'brandID']:
        df = behavior_data[['behavior', count_feature]].groupby(count_feature, as_index=False).agg(
            {'behavior': 'count'}).rename(columns={'behavior': str(count_feature) + '_sum'})

        df.to_csv(str(count_feature) + '_count.csv', index=False)

    # 生成category的数据特征
    temp = behavior_data[['behavior', 'categoryID']].groupby('categoryID', as_index=False).agg(
        {'behavior': ['median', 'std']})
    temp.columns = ['categoryID', 'category_median', 'category_std']
    temp.to_csv('../statistics_feature/category_higher.csv', index=False)
    print("save sucessfully.....")

    # 生成itemID的数据特征
    temp_itemID = behavior_data[['behavior', 'itemID']].groupby('itemID', as_index=False).agg(
        {'behavior': ['median', 'std']}
    )
    temp_itemID.columns = ['itemID', 'item_median', 'item_std']
    temp_itemID.to_csv('../statistics_feature/item.higher.csv', index=False)
    print("save sucessfully.....")
    
