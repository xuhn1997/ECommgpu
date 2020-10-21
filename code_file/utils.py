import os
import pickle
import multiprocessing as mp
import time
import numpy as np
import pandas as pd
# import cython


def save_file(filepath, data):
    """
    存储数据文件
    :param filepath:
    :param data:
    :return:
    """

    parent_path = filepath[: filepath.rfind("/")]
    if not os.path.exists(parent_path):
        os.mkdir(parent_path)
    with open(filepath, "wb") as f:
        pickle.dump(data, f)


def load_file(filepath):
    """
    载入二维的数据
    :param filepath:
    :return:
    """

    with open(filepath, "rb") as f:
        data = pickle.load(f)
    return data


# 优化存储空间的函数。。。。
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum()
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        # else:
        #     df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum()
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def generate_logs_for_each_group(matrix, q):
    """
    为每个用户组获取他们的行为日志
    :param matrix:
    :param q:
    :return:
    """
    user_log = dict()
    for row in matrix:
        # 遍历一个矩阵，为[[user, item, be(在计算为每个用户进行推荐的时候
        # 才用到), timestap?]........]形状
        user_log.setdefault(int(row[0]), [])
        user_log[int(row[0])].append((int(row[1]), int(row[2]), int(row[3])))

    print("This bacth is finished!!!")
    q.put(user_log)


def get_logs_from(path):
    """
    获取分组好的用户的日志信息
    :param path:
    :return:
    """
    f = open(path, 'r')
    a = f.read()
    dict_name = eval(a)  # 将字符串转化成原始的数据类型
    f.close()
    return dict_name
