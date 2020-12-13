# # 测试保存好的三个文件，
# import pandas as pd
# import numpy as np
#
# filename1 = "../data/df_behavior_train.csv"
# df_behavior_train = pd.read_csv(filename1, header=None)
#
# filename2 = "../data/df_behavior_test.csv"
# df_behavior_test = pd.read_csv(filename2, header=None)
#
# filename3 = "../data/df_item_sort.csv"
# df_item_sort = pd.read_csv(filename3, header=None)
#
# print(df_behavior_train.shape)
# print(df_behavior_test.shape)
# print(df_item_sort.shape)  # 334254

import pandas as pd
import numpy as np
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

df1 = pd.DataFrame({
    'A': [5, 1, 2, 2, 3, 3],
    'B': [2, 3, 4, 5, 7, 8],
    'C': [0.9, 0.8, 0.7, 0.8, 0.9, 0.8],
    'D': [1, 0, 0, 1, 0, 0],

})

df2 = pd.DataFrame({
    'A': [5, 1, 2, 2, 3, 3],
    'B': [2, 3, 4, 5, 7, 8],
    'C': [0.9, 0.8, 0.7, 0.8, 0.9, 0.8],
    'D': [1, 0, 0, 1, 0, 0]
})
df3 = pd.concat([df1, df2]).reset_index(drop=True)

print(df3)
print(df3['A'].unique())

item_profile = df3.drop_duplicates('A')
print(item_profile['A'])
# print(">>>>>>>>>>>")
# trainset = []
# for eachcat in df3.groupby('A'):
#     print(eachcat[0])
#     print(">>>>>>>>>>>>")
#     print(eachcat[1])
#     """
#     如何遍历每一行的？
#     """
#     temp = eachcat[1].reset_index(drop=True)
#     # print(temp)
#     # print(temp.index)
#     hist = temp['D'].tolist()
#     # print("******")
#     # print(hist)
#     # print("******")
#     k = 1
#     # for
#     temp = np.array(temp)
#     indexs = temp.shape[0]
#     for index in range(indexs):
#     # for
#         print(")))))))")
#         print(index)
#         print("((((((((((")
#         # print(temp.loc[index])
#         # hist.append
#         if index == 0:
#             continue
#         elif index == indexs - 1:
#             print(k)
#             # continue
#         else:
#             # index_list = np.array(temp.loc[index])
#             index_list = temp[index, :]
#             # index_list = temp.loc[index]
#             index_list = index_list.tolist()
#             # index_list = list(map(int, index_list))  # 转化成整数
#             # print(index_list)
#             # temp = pad_sequences(np.array(hist[0:k]), maxlen=4, padding='post', truncating='post', value=0)
#             t = hist[0:k]
#             for i in range(4-k):
#                 t.append(0)
#             index_list.append(t)
#             index_list.append(k)
#             trainset.append(index_list)
#             k = k + 1
#             print(index_list)
#     break;
# print("____________")
# print(trainset)


# tt = np.array([line[0:3] for line in trainset])
# tt = np.array([list(map(int, line[0:1])) for line in trainset])
# print(tt)
#
# item_feature_index = [i for i in range(7, 32)]
# item_feature_index.append(3)
# print(item_feature_index)

# list_1 = [1,2,3,4,5]
# list_2 = [2,4]
# # list_3 = [list_1[i] for i in list_2]
# # list_3 = list_1.difference(list_2)
# list_3 = list(set(list_1) - set(list_2))
# print(list_3)

# print("000")
# print()
# print(list(map(int, trainset)))

# df4 = pd.DataFrame({'A': [1]})
# print(">>>>>>>>>")
# print(df4)
# print(">>>>>>>>>")
# print(df4.agg({"A":"skew"}))
"""
测试map函数
"""
# map_dic = {1:100, 0:10000}
# df3['D'] = df3['D'].map(map_dic)

# print(df3)
# index = np.random.randint(4, size=10 * 2)
# print(index)
# print(index.shape)
# data1 = df1[df1['D'] == 1]
# data0 = df1[df1['D'] == 0]
# print(data0)
# index = np.random.randint(len(data0), size=10 * 2)
# print(len(data0))
# print(index)
# print(index.shape)

# lower_data0 = data0.iloc[list(index)]
# print(lower_data0.shape)
#
# data0_1 = pd.concat([lower_data0, data1])
# print(data0_1)
# print(data0_1.shape)
