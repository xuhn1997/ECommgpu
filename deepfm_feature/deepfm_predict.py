"""
需要的一点就是
使用的embedding_lookup的时候需要对全部特征排列
成多个域，然后相乘到对应的值得到相应的embedding向量，
对于连续值特征，值还是原来的值，但是它的索引只有一个，所以通过
embedding_lookup取出来的embedding向量都是相同的，所以要乘以它的值
而对于离散型特征，取出的就是它的embedding向量，因为乘以它的值1是不会改变的!!!!!!!

但是对于原论文来说，只对域离散型做embedding操作，连续值做归一化就好的，然后进入
网络的时候要进行bn操作。。。。。
"""

import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import tensorflow as tf
from code_file.model import code_start

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
config.gpu_options.allow_growth = True  # 程序按需申请内存
sess = tf.Session(config=config)

import numpy as np
import pandas as pd

from code_file.utils import reduce_mem_usage
from deepfm_feature.train_data_deal import deal_underline_test_data
from sklearn import preprocessing
from deepfm_feature.generate_underline_test_data import get_deepfm_test_data

"""进行预测的过程"""

BATCH_SIZE = 1024
FEATURE_SIZE = 601221  # 此处为转化成特征之后的特征总数
FIELD_SIZE = 9  # 特征域的大小
EMBEDDING_SIZE = 8  # embedding之后的最后维度的大小
# 定义placeholder部分

iS_training = tf.placeholder(tf.bool, name="is_training")
feature_index = tf.placeholder(tf.int32, shape=[None, None], name="feature_index")
feature_value = tf.placeholder(tf.float32, shape=[None, None], name="feature_value")

label = tf.placeholder(tf.float32, shape=[None, 1], name='label')


# 建立权重函数
def weight(shape):
    w_stddev = 1. / tf.sqrt(shape[0] / 2.)  # shape[0]指矩阵的行数
    return tf.random_normal(shape=shape, stddev=w_stddev)


# 以下是deeppart部分的参数变量
D_W1 = tf.Variable(weight([FIELD_SIZE * EMBEDDING_SIZE, 256]))
D_B1 = tf.Variable(tf.zeros(shape=[1, 256]))

D_W2 = tf.Variable(weight([256, 128]))
D_B2 = tf.Variable(tf.zeros(shape=[1, 128]))

D_W3 = tf.Variable(weight([128, 32]))
D_B3 = tf.Variable(tf.zeros(shape=[1, 32]))

# 接下来就是最后一层即sigmoid的参数
"""
首先要明确最后的一层的输入维度
一阶的输出就是filed_size, 二阶的输出就是embedding_size, deep部分的输出就是32
concat之后就是filed_size + embedding_size + 32
"""
S_W = tf.Variable(weight([FIELD_SIZE + EMBEDDING_SIZE + 32, 1]))
S_B = tf.Variable(tf.zeros(shape=[1, 1]))

# embedding部分的变量
weight_embedding = tf.Variable(
    tf.random_normal([FEATURE_SIZE, EMBEDDING_SIZE], 0.0, 0.01), name="feature_embedding"
)
# 用于一阶的情况
bias_embedding = tf.Variable(tf.random_normal([FEATURE_SIZE, 1], 0.0, 1.0),
                             name="feature_embedding_bias")


# 利用上面的embeddings作为输入层
def deep_part(x_input):
    """
    这个是deep部分的网络
    这里定义了两层
    :param x_input:为输入数据，维度就是(-1, filed_size * embedding_size)
    :return:
    """
    # hidden = tf.nn.relu(tf.matmul(x_input, D_W1) + D_B1)
    hidden1 = tf.matmul(x_input, D_W1) + D_B1
    hidden1 = tf.layers.dropout(hidden1, rate=0.5, training=iS_training)
    hidden1 = tf.layers.batch_normalization(hidden1, training=iS_training)
    hidden1 = tf.nn.relu(hidden1)

    hidden2 = tf.matmul(hidden1, D_W2) + D_B2
    hidden2 = tf.layers.dropout(hidden2, rate=0.5, training=iS_training)
    hidden2 = tf.layers.batch_normalization(hidden2, training=iS_training)
    hidden2 = tf.nn.relu(hidden2)
    """进行正则化"""
    deep_out = tf.nn.relu(tf.matmul(hidden2, D_W3) + D_B3)

    return deep_out  # 此时的维度就是(-1, 32)


"""然后定义最后一层也就是sigmoid层"""


def sigmoid_part(x_input):
    """
    最后一层网络
    :param x_input:(-1， filed_size + embedding_size + 32)
    :return:
    """
    sigmoid_out = tf.matmul(x_input, S_W) + S_B

    return sigmoid_out  # 维度就是(-1, 1)


"""
embedding部分
"""
"""
这里的embedding_lookup函数取出来的就是对应特征值的embedding向量
初始化embedding的维度就是(总的特征数，要映射多大的embedding大小)
"""
embeddings = tf.nn.embedding_lookup(weight_embedding, feature_index)  # 此时的维度就是N*K*F
# N表示样本数，K表示特征域的大小，这里的话就是9， F表示embedding的大小
# 对于特征的值进行维度的转化才能与上面的的embedding进行广播相乘
reshape_feature_value = tf.reshape(feature_value, [-1, FIELD_SIZE, 1])

# 进行相乘才得到经过embedding层之后得数值
embeddings = tf.multiply(embeddings, reshape_feature_value)  # 此时的维度就是N * K * F

"""以下是FM部分的网络"""
# 查一阶时对应的embedding向量
fm_first_oder = tf.nn.embedding_lookup(bias_embedding, feature_index)  # 此时的维度为N * K * 1同上
temp = tf.multiply(fm_first_oder, reshape_feature_value)  # 此时的维度是N * K * 1, 就是一阶的xi * vi
# 然后将最后的一个维度进行求和
fm_first_oder = tf.reduce_sum(temp, axis=2)  # 之后的维度就是N*K其中k就是特征域的个数

# 然后是对于二阶进行操作 最后出来的维度就是n*f
# 先对于上面的embedding求和然后进行平方
sum_feature_emb = tf.reduce_sum(embeddings, axis=1)  # 此时的维度就是N * F
sum_square_feature_emb = tf.square(sum_feature_emb)  # 此时的维度就是N * F

# 然后在平方之后然后进行求和
square_feature_emb = tf.square(embeddings)  # 此时的维度就是N * K *F
square_sum_feature_emb = tf.reduce_sum(square_feature_emb, axis=1)  # 此时的维度就是N* F

# 此时FM的二阶部分的整理结果如下
fm_second_order = 0.5 * tf.subtract(sum_square_feature_emb, square_sum_feature_emb)
# 以上维度就是N * F， 其中N表示样本数，F表示embedding的大小

"""开始串接以上的网络"""
# 先获得deep部分得输出
deep_input = tf.reshape(embeddings, (-1, FIELD_SIZE * EMBEDDING_SIZE))

deep_out = deep_part(deep_input)  # 此时维度就是(-1, 32)

# 将所有的输出进行concat以下
sigmoid_input = tf.concat([fm_first_oder, fm_second_order, deep_out],
                          axis=1)  # 此时的维度大小就是(-1, filed_size + embedding_size + 32)
out = sigmoid_part(sigmoid_input)  # 此时的维度大小就是(-1, 1)

out = tf.nn.sigmoid(out)

out = tf.reshape(out, (-1, 1))

# 获取好测试集
test_data_index, test_data_value = deal_underline_test_data()
print("转换后测试的维度是。。。")
print(test_data_value.shape)
print(test_data_index.shape)


def predict_function(data_index, data_value):
    """
    预测模型函数，上传到testdata处理好的训练集
    :param data_value:
    :param data_index:
    :return:
    """
    STEPS = len(data_index) // BATCH_SIZE

    """开始处理测试集"""
    data_index = reduce_mem_usage(data_index)
    data_value = reduce_mem_usage(data_value)

    """漏了一部分就是数据预处理，对于连续值进行归一化操作！！！！！"""
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.1, 1))
    # 连续特征有哪些'sim', 'age'
    tmp1 = np.array(data_value['sim'])
    tmp1 = np.reshape(tmp1, (-1, 1))

    tmp2 = np.array((data_value['age']))
    tmp2 = np.reshape(tmp2, (-1, 1))

    data_value['sim'] = min_max_scaler.fit_transform(tmp1)
    data_value['age'] = min_max_scaler.fit_transform(tmp2)

    feature = [x for x in data_index.columns if x not in ['userID', 'itemID', 'label']]
    # 获得训练的训练集
    data_index = data_index[feature]
    data_index = np.array(data_index)

    data_value = data_value[feature]
    data_value = np.array(data_value)

    # 获取deepfm保存好的网络
    deepfm_model = '../deepfm_save_model/deepfm_model_saver.ckpt'
    # deepfm_model = '../deepfm_real_embeddings_save/deepfm_model_saver.ckpt'

    y_pred = None
    with tf.Session() as sess:
        saver = tf.train.Saver()
        saver.restore(sess, deepfm_model)

        # 开始进行训练
        for step in range(STEPS):
            pred_temp = sess.run(out, feed_dict={
                feature_index: data_index[step * BATCH_SIZE:(step + 1) * BATCH_SIZE],
                feature_value: data_value[step * BATCH_SIZE:(step + 1) * BATCH_SIZE],
                iS_training: False})
            """合并好数据集"""
            pred_temp = np.reshape(pred_temp, (-1, 1))
            if y_pred is None:
                y_pred = pred_temp
            else:
                y_pred = np.concatenate((y_pred, pred_temp), axis=0)
        """处理剩余的部分"""
        print(step)  # 80221, 而step的总数是80222
        pred_last = sess.run(out, feed_dict={
            feature_index: data_index[(step + 1) * BATCH_SIZE:],
            feature_value: data_value[(step + 1) * BATCH_SIZE:],
            iS_training: False})

        pred_last = np.reshape(pred_last, (-1, 1))

        y_pred = np.concatenate((y_pred, pred_last), axis=0)

        return y_pred  # 维度为(-1, 1)


# 测试一下输出的概率是？
# test_data_index = test_data_index[0:1024]
# test_data_value = test_data_value[0:1024]
#
# pred = predict_function(test_data_index, test_data_value)
# print(pred[0:300])
# pred = predict_function(test_data_index, test_data_value)
# print(pred.shape)

if __name__ == '__main__':
    """对数据集进行推荐"""
    # pd.read_csv
    # 先获取原始的测试数据集
    recall_df = get_deepfm_test_data()
    # 测试集的维度
    print(recall_df.shape)
    """
    以上的数据列为：
    合并之后总的特征就是
    userID, itemID, sim, behavior, day, 'sex', 'age',
    'ability'， categoryID, shopID, brandID,
    """
    pred = predict_function(test_data_index, test_data_value)
    print(pred.shape)
    recall_df['pred_prob'] = pred

    test_data1 = recall_df.groupby('userID', sort=False).apply(
        lambda x: x.sort_values('pred_prob', ascending=False)).reset_index(drop=True)

    # 保存好创建的召回集合
    recall_list = dict()
    users = test_data1['userID'].unique()
    users = users.tolist()

    print("begin to recommend..........")

    for user in users:
        recall_list.setdefault(int(user), [])

        tmp = test_data1[test_data1['userID'] == user]
        """
           先判断用户的商品是否足够50个？
        """
        if len(tmp) < 50:
            """
              需要进行冷启动的操作
            """
            numbers = 50 - len(tmp)
            item_lists = code_start(numbers)
            # for i in range(len(tmp)):
            # matrix = data[data['userID'].isin(user_groups[groupID])][['userID', 'itemID', 'behavior', 'day']].values
            matrix = tmp[['itemID', 'pred_prob']].values  # 这样的话数据会变成浮点数
            for row in matrix:
                recall_list[int(user)].append((int(row[0]), row[1]))
            recall_list[user].extend(item_lists)  # 将其进行合并

            continue
        else:
            """
            进行正常推荐
            """
            matrix = tmp[['itemID', 'pred_prob']].values  # 这样的话数据会变成浮点数
            for row in matrix:
                recall_list[int(user)].append((int(row[0]), row[1]))
                if len(recall_list[int(user)]) == 50:
                    break

    # # 最后将召回的字典保存在txt文件中
    #
    recall_result_file = open("../recall_list/result_recall_deepfm.txt", 'w')
    recall_result_file.write(str(recall_list))
    recall_result_file.close()

    print("save successfully........")
