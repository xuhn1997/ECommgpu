import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import pandas as pd
import numpy as np

from tensorflow import keras
from deepfm_feature.train_data_deal import deal_underline_train_data
from sklearn import preprocessing

# 定义超参数
FEATURE_SIZE = 601221  # 此处为转化成特征之后的特征总数
FIELD_SIZE = 9  # 特征域的大小
EMBEDDING_SIZE = 8  # embedding之后的最后维度的大小
BATCH_SIZE = 1024

"""获取训练集的部分"""
"""
之后的总的特征就是['userID', 'itemID', 'sim', 'label', 'behavior', 'day', 'sex', 'age',
       'ability', 'categoryID', 'shopID', 'brandID'],
"""
data_index, data_value = deal_underline_train_data()

"""漏了一部分就是数据预处理，对于连续值进行归一化操作！！！！！"""
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.1, 1))
# 连续特征有哪些'sim', 'age'
tmp1 = np.array(data_value['sim'])
tmp1 = np.reshape(tmp1, (-1, 1))

tmp2 = np.array((data_value['age']))
tmp2 = np.reshape(tmp2, (-1, 1))

data_value['sim'] = min_max_scaler.fit_transform(tmp1)
data_value['age'] = min_max_scaler.fit_transform(tmp2)

# 取得标签的值
train_y = np.array(data_index['label'])
train_y = np.reshape(train_y, (-1, 1))

feature = [x for x in data_index.columns if x not in ['userID', 'itemID', 'label']]
# 获得训练的训练集
train_data_index = data_index[feature]
train_data_index = np.array(train_data_index)

train_data_value = data_value[feature]
train_data_value = np.array(train_data_value)

num_input = train_data_value.shape[0]
STEPS = num_input // BATCH_SIZE
print(num_input)
print("**********")
print(STEPS)

# 定义placeholder部分

iS_training = tf.placeholder(tf.bool, name="is_training")
feature_index = tf.placeholder(tf.int32, shape=[None, None], name="feature_index")
feature_value = tf.placeholder(tf.float32, shape=[None, None], name="feature_value")

label = tf.placeholder(tf.float32, shape=[None, 1], name='label')


# 建立权重函数
def weight(shape):
    w_stddev = 1. / tf.sqrt(shape[0] / 2.)  # shape[0]指矩阵的行数
    return tf.random_normal(shape=shape, stddev=w_stddev)


# 定义相关的变量

# 以下是deeppart部分的参数变量
D_W1 = tf.Variable(weight([FIELD_SIZE * EMBEDDING_SIZE, 512]))
D_B1 = tf.Variable(tf.zeros(shape=[1, 512]))

D_W2 = tf.Variable(weight([512, 256]))
D_B2 = tf.Variable(tf.zeros(shape=[1, 256]))

D_W3 = tf.Variable(weight([256, 32]))
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
bias_embedding = tf.Variable(tf.random_normal([FEATURE_SIZE, 1], 0.0, 1.0), name="feature_embedding_bias")

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

"""接下来就是deep部分"""


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

"""定义loss以及优化器"""
# loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=out, logits=tf.reshape(label, (-1, 1)))  # 分类问题的损失函数
loss = tf.losses.log_loss(label, out)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999,
                                   epsilon=1e-8).minimize(loss)

"""
开始进行训练
训练之前要获得训练集？？
"""
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(10000):
        for step in range(STEPS):
            epoch_loss, _ = sess.run([loss, optimizer], feed_dict={
                feature_index: train_data_index[step * BATCH_SIZE:(step + 1) * BATCH_SIZE],
                feature_value: train_data_value[step * BATCH_SIZE:(step + 1) * BATCH_SIZE],
                label: train_y[step * BATCH_SIZE: (step + 1) * BATCH_SIZE], iS_training: True})

        if epoch % 100 == 0:
            print("epoch %s, loss is %s" % (str(epoch), str(epoch_loss)))

    """保存好训练的网络"""
    saver = tf.train.Saver()
    saver.save(sess, '../deepfm_save_model/deepfm_model_saver.ckpt')
    print("保存好DeepFM网络............")
