"""
这一部分可以将用户分成多组
并在每个组种统计用户的行为日志
以便后面的并行化处理
"""
import multiprocessing as mp
import pandas as pd
import numpy as np
import time
from code_file.utils import generate_logs_for_each_group

# 要对于用户进行划分组。。。
user = pd.read_csv("../data/user.csv", header=None)
user.columns = ['userID', 'sex', 'age', 'ability']

data = pd.read_csv("../data/df_behavior.csv")  # 读取用户行为日志，注意这里是1到16天的数据生成用户日志！！！！生成物品协同矩阵！！！！

# 获取用户的集合
users = list(set(user['userID']))

print(len(users))
# print(users)

# 开始进行划分。。。
CPU_NUMS = 8
user_groups = [users[i: i + len(users) // CPU_NUMS]
               for i in range(0, len(users), len(users) // CPU_NUMS)]
# user_groups的最后形式就是[[0, 1,........], [783, 8042.......]]

# 查看user是否有冗余的数据
# print(len(np.unique(user['userID'])))
q = mp.Queue()  # 建立多线程的队列
for groupID in range(len(user_groups)):
    # 此时的user_groups的长度为5(0, 1, 2, 3, 4)
    matrix = data[data['userID'].isin(user_groups[groupID])][['userID', 'itemID', 'behavior', 'day']].values
    task = mp.Process(target=generate_logs_for_each_group,
                      args=(matrix, q))
    task.start()

# 计算时间，可以不用理会
start_time = time.time()
print("Waiting  for the son processing")
while q.qsize() != len(user_groups):
    pass
end_time = time.time()
print("Over, the time cost is:" + str(end_time - start_time))

# 开始对于分好组的用户日志进行分块(0, 1, 2, 3, 4)存储
for i in range(len(user_groups)):
    temp = q.get()  # 出队列
    f = open('../full_logs/user_logs_group' + str(i) + '.txt', 'w')
    f.write(str(temp))
    f.close()

print("save successfully!!!!!!!")
