"""
此文件针对于原始数据集生成一些
静态的特征，比如商品的与行为交叉特征。
而用户与行为的交叉特征就是动态特征，因为用户user_behavior里面的是。。。。。？
"""
from code_file.utils import reduce_mem_usage
import numpy as np
import pandas as pd

# 载入原始的数据集 即7到21号的。。
data = pd.read_csv("../data/df_behavior_train.csv")  # userID,behavior,timestap,itemID,date,day
"""
现在为了时间的，选择19到21号即可
"""
data = data[data['day'] >= 19]


data = reduce_mem_usage(data)

# 载入用户的数据
user = pd.read_csv("../data/user.csv", header=None)
user.columns = ['userID', 'sex', 'age', 'ability']
user = reduce_mem_usage(user)

# 载入商品的数据
item = pd.read_csv("../data/df_item.csv")  # categoryID,shopID,brandID,itemID
item = reduce_mem_usage(item)

# 合并商品以及用户的数据集进行统计过程
data = pd.merge(left=data, right=user, on=['userID'], how='left')
data = pd.merge(left=data, right=item, on=['itemID'], how='left')

# 开始统计商品特征交叉的count类型， 即用用户行为behavior与上面四个商品的特征的交叉
for count_feature in ["itemID", "categoryID", "shopID", "brandID"]:
    data[['behavior', count_feature]].groupby(count_feature, as_index=False).agg(
        {'behavior': 'count'}).rename(columns={'behavior': count_feature + '_count'}). \
        to_csv("../Static_feature_file/" + str(count_feature) + "_count.csv", index=False)

# 开始统计商品特征交叉的sum类型， 即用用户行为behavior与上面四个商品的特征的交叉
for count_feature in ["itemID", "categoryID", "shopID", "brandID"]:
    data[['behavior', count_feature]].groupby(count_feature, as_index=False).agg(
        {'behavior': 'sum'}).rename(columns={'behavior': count_feature + '_sum'}). \
        to_csv("../Static_feature_file/" + str(count_feature) + "_sum.csv", index=False)

# 下面开始统计itemID与category和行为程度bahavior的统计学特征
for feature in ["categoryID", "itemID"]:
    temp = data[["behavior", feature]].groupby(feature, as_index=False).agg({"behavior": ['median', 'std', 'skew']})
    temp.columns = [feature, feature + '_median', feature + '_std', feature + '_skw']
    temp.to_csv("../Static_feature_file/" + str(feature) + "_higher.csv", index=False)

# 接下生成用户的sex，age， ability与商品的交叉特征，即在itemID为1时，男的行为的统计特征。。。


for count_feature in ['sex', 'age', 'ability']:
    data[["behavior", 'itemID', count_feature]].groupby(['itemID', count_feature], as_index=False). \
        agg({"behavior": "count"}).rename(columns={"behavior": "itemID_to" + count_feature + '_count'}). \
        to_csv("../Dynamic_feature_file/item_to_" + str(count_feature) + "_count.csv", index=False)

# 接下来就是构造商品等级的特征,有商品类别衍生的。。。商品等级
item_count = pd.read_csv("../Static_feature_file/itemID_count.csv")

temp = pd.merge(left=item, right=item_count, on=['itemID'], how='left')
# 注意这里面可能有些商品用户没有行为过的，所以将填充NaNz值为0
temp = temp.fillna(0)
item_rank = []
for each_category in temp.groupby('categoryID'):
    """
    要在商品类别的基础上计算商品的等级，
    因为用户的历史感兴趣集中在商品类别更多点。。
    """
    each_category_df = each_category[1].sort_values('itemID_count', ascending=False) \
        .reset_index(drop=True)  # 在某类别对于商品的数量进行降序排序
    each_category_df['rank'] = each_category_df.index + 1
    length = each_category_df.shape[0]  # 获取当前类别的长度
    each_category_df['rank_percent'] = (each_category_df.index + 1) / length  # 在该列表所占的百分比，排序越前越低

    item_rank.append(each_category_df[["itemID", "rank", "rank_percent"]])  # 加入链表中

item_rank = pd.concat(item_rank, sort=False)
item_rank.to_csv("../Static_feature_file/item_rank.csv", index=False)


# 围绕着categoryID在生成与itemID，shopID， brandID
def unique_count(x):
    return len(set(x))


cat1 = item.groupby('categoryID', as_index=False).agg({'itemID': unique_count}).rename(columns={"itemID": 'itemnum_oncat'})
cat2 = item.groupby('categoryID', as_index=False).agg({'shopID': unique_count}).rename(columns={"shopID": 'shopnum_oncat'})
cat3 = item.groupby('categoryID', as_index=False).agg({'brandID': unique_count}).rename(columns={"brandID": 'brandnum_oncat'})

pd.concat([cat1, cat2[['shopnum_oncat']], cat3[['brandnum_oncat']]], axis=1).to_csv("../Static_feature_file/category_lower.csv", index=False)