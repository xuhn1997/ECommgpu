"""
此文件包的目的是用来统计每个用户组的相似度矩阵
"""
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
import scipy as scp
from code_file.utils import get_logs_from
from code_file.model import calculate_matrix

# 要想计算协同过滤矩阵，要获得物品的编号最大数
ITEM_NUM = 4318203
# 获取当前组的用户行为日志

user_logs = get_logs_from('../full_logs/user_logs_group6.txt')

# 转化成链表的形式
user_logs = list(user_logs.items())

for i in range(0, len(user_logs), 10000):
    print("The %d " % i + 'batch is started...........')
    print("--------------------------")
    mat = lil_matrix((ITEM_NUM, ITEM_NUM), dtype=float)
    mat = calculate_matrix(mat, user_logs[i: i + 10000], alpha=0.5)
    # 计算每一千条之后好之后开始存下来
    # scp.sparse.save_npz('../tmpData/sparse_matrix_%d_batch_group4.npz' % i, mat.tocsr())
    scp.sparse.save_npz('../tmpdata_iuf/sparse_matrix_%d_batch_group6.npz' % i, mat.tocsr())
    print("save successfully!!!!")
    print("************************")
