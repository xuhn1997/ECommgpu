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

df1 = pd.DataFrame({
    'A': [1, 1, 2, 2, 3, 3],
    'B': [2, 3, 4, 5, 7, 8],
    'C': [0.9, 0.8, 0.7, 0.8, 0.9, 0.8],
    'D': [1, 0, 0, 1, 0, 0]
})

print(df1)

index = np.random.randint(4, size=10 * 2)
print(index)
print(index.shape)
data1 = df1[df1['D'] == 1]
data0 = df1[df1['D'] == 0]
print(data0)

lower_data0 = data0.iloc[list(index)]
print(lower_data0.shape)

data0_1 = pd.concat([lower_data0, data1])
print(data0_1)
print(data0_1.shape)
