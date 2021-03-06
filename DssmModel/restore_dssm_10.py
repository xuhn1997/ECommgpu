import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.5  # 程序最多只能占用指定gpu50%的显存
# config.gpu_options.allow_growth = True  # 程序按需申请内存
# sess = tf.Session(config=config)

from DssmModel.DSSM_Model import SequencePoolingLayer
import pickle as pck
from DssmModel.gen_16_21_input import gen_model_input
from DssmModel.get_dssm_test_10 import get_item_data_index
import numpy as np
import pandas as pd

# 现在处理user的部分的输入特征
# 现在建立一个总的embedding的举证
# embedding部分的变量
FEATURE_SIZE = 2847369  # 总的离散特征的数目
EMBEDDING_SIZE = 10
BATCH_SIZE = 256

weight_embedding = tf.Variable(
    tf.random_normal([FEATURE_SIZE, EMBEDDING_SIZE], 0.0, 0.01), name="feature_embedding"
)
IS_training = tf.placeholder(tf.bool, name="is_training")
user_temp_input = tf.placeholder(tf.int32, shape=[None, 6], name="user_temp_input")  # 注意其中有个连续型数值

user_item_input = tf.placeholder(tf.int32, shape=[None, 11], name="user_item_input")  # 用户的历史行为

user_hist_len = tf.placeholder(tf.int32, shape=[None, 1], name="user_hist_len")  # 用户行为的历史真实长度(-1, 1)

# 接下来就是处理item部分的特征
item_input = tf.placeholder(tf.float32, shape=[None, 4], name="item_input")  # 维度是(batch_size, item_features)

label = tf.placeholder(tf.float32, shape=[None, 1], name="label")
print(">>>>>>>")
print(user_temp_input.shape)
# user_temp_input = user_temp_input[:, 1:]  # 去掉用户的id
# print(">>>>>>>")
# print(user_temp_input.shape)
user_other_emb = tf.nn.embedding_lookup(weight_embedding,
                                        user_temp_input[:, 1:])  # 此时的维度是？N * K * F(Batch_size, 特征域的长度，embedding_size)

print("user_other_emb: ")
print(user_other_emb.shape)
# 现在对于user_item_input进行处理
user_items_emb = tf.nn.embedding_lookup(weight_embedding,
                                        user_item_input)  # 此时的维度就是(Batch_size, MAX_LEN, embedding_size)
print("user_items_emb: ")
print(user_items_emb.shape)
# 此时下面的维度就是(batch_size, 1, embedding_size)
user_items_embedding = SequencePoolingLayer(user_seq_embedding=user_items_emb, user_behavior_length=user_hist_len)
print("user_items_embedding: ")
print(user_items_embedding.shape)

# 获取item离散型的变量

item_number_input = tf.to_int32(item_input)

item_number_embedding = tf.nn.embedding_lookup(weight_embedding,
                                               item_number_input)  # 维度就是(batch_size, 商品离散特征个数, embedding_size)

"""
现在开始组织特征构建网络
"""
# 构建用户的输入
# 处理离散的特征
user_other_emb = tf.reshape(user_other_emb, (-1, 5 * EMBEDDING_SIZE))

user_items_embedding = tf.reshape(user_items_embedding, (-1, EMBEDDING_SIZE))

user_dnn_input = tf.concat([user_other_emb, user_items_embedding], axis=1)

print("user的输入维度: %s" % str(user_dnn_input.shape))

item_number_embedding = tf.reshape(item_number_embedding, (-1, 4 * EMBEDDING_SIZE))

item_dnn_input = item_number_embedding
print("item的输入维度: %s" % str(item_dnn_input.shape))


def user_part(x_input):
    """
    这个是deep部分的网络
    这里定义了两层
    :param x_input:为输入数据，维度就是(-1, filed_size * embedding_size)
    :return:
    """
    x_input = tf.layers.batch_normalization(x_input, name='b1', training=IS_training, reuse=tf.AUTO_REUSE)

    u_layer1 = tf.layers.dense(x_input, 256, name="u_layer1", reuse=tf.AUTO_REUSE)
    hidden1 = tf.layers.batch_normalization(u_layer1, training=IS_training, name='b2', reuse=tf.AUTO_REUSE)
    hidden1 = tf.nn.leaky_relu(hidden1)

    user_out = tf.layers.dense(hidden1, 128, activation=tf.nn.leaky_relu, name="user_out", reuse=tf.AUTO_REUSE)

    return user_out  # 此时的维度就是(-1, 128)


def item_part(x_input):
    """
    这个是deep部分的网络
    这里定义了两层
    :param x_input:为输入数据，维度就是(-1, filed_size * embedding_size)
    :return:
    """
    x_input = tf.layers.batch_normalization(x_input, name='b3', training=IS_training, reuse=tf.AUTO_REUSE)

    i_layer1 = tf.layers.dense(x_input, 256, name="i_layer1", reuse=tf.AUTO_REUSE)
    hidden1 = tf.layers.batch_normalization(i_layer1, training=IS_training, name='b4', reuse=tf.AUTO_REUSE)
    hidden1 = tf.nn.leaky_relu(hidden1)

    item_out = tf.layers.dense(hidden1, 128, activation=tf.nn.leaky_relu, name="item_out", reuse=tf.AUTO_REUSE)

    return item_out  # 此时的维度就是(-1, 128)


user_dnn_out = user_part(user_dnn_input)
item_dnn_out = item_part(item_dnn_input)

print("............")
print("user的输出维度: %s" % str(user_dnn_out.shape))

print(">>>>>>>>>>")
print("item的输出维度: %s" % str(item_dnn_out.shape))

"""
获取数据集

"""
with open('../data/test_set_10_10.pkl', 'rb') as f:
    train_set = pck.load(f)

train_model_input, train_label = gen_model_input(train_set=train_set)

train_label = np.reshape(train_label, (-1, 1))

user_temp = train_model_input['users_list']

STEPS = user_temp.shape[0] // BATCH_SIZE

user_items = train_model_input['user_item_list']

user_behavior_length = np.reshape(train_model_input['user_hist_len'], (-1, 1))

item_feature = get_item_data_index()  # 获取所有的商品
item_feature = np.array(item_feature)

print("///////")
print(user_temp.shape)

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess, '../DSSM_SAVE_MODEL_10/dssm_model_saver_10.ckpt')

    uu, ii = sess.run([user_dnn_out, item_dnn_out],
                      feed_dict={user_temp_input: user_temp,
                                 user_item_input: user_items,
                                 user_hist_len: user_behavior_length,
                                 item_input: item_feature,
                                 IS_training: False})

    with open('../data/dssm_user_embedding.pkl', 'wb') as f:
        pck.dump(uu, f, pck.HIGHEST_PROTOCOL)
        # pck.dump(ii, f, pck.HIGHEST_PROTOCOL)

    with open('../data/dssm_item_embedding.pkl', 'wb') as f:
        pck.dump(ii, f, pck.HIGHEST_PROTOCOL)
