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
from DssmModel.gen_data_input import gen_model_input

import numpy as np
import pandas as pd


# 建立权重函数
def weight(shape):
    w_stddev = 1. / tf.sqrt(shape[0] / 2.)  # shape[0]指矩阵的行数
    return tf.random_normal(shape=shape, stddev=w_stddev)


def Similarity(user_dnn_out, item_dnn_out, type):
    """
    计算DSSM两边的向量相似的函数(A * B) / ||A|| ||B||
    :param type: 是否需要计算cos值？
    :param user_dnn_out:用户部分的输出向量(batch_size, 32)
    :param item_dnn_out:商品部分的输出向量(batch_size, 32)
    :return:输出的维度就是(batch_size, 1)
    """
    if type == 'cos':
        user_norm = tf.norm(user_dnn_out, axis=1)  # axis=1是对于行求它的范式

        item_norm = tf.norm(item_dnn_out, axis=1)

    cosine_score = tf.reduce_sum(tf.multiply(user_dnn_out, item_dnn_out), axis=-1)
    if type == 'cos':
        cosine_score = tf.divide(cosine_score, (user_norm * item_norm) + 1e-8)  # 防止除0错误
        # cosine_score = tf.clip_by_value(cosine_score, -1, 1.0)  # 限制输出的范围在-1， 1

    # cosine_score = tf.nn.sigmoid(cosine_score)
    cosine_score = tf.reshape(cosine_score, (-1, 1))
    return cosine_score  # 最后再经过sigmoid函数


# 现在处理user的部分的输入特征
# 现在建立一个总的embedding的举证
# embedding部分的变量
FEATURE_SIZE = 1140821  # 总的离散特征的数目
EMBEDDING_SIZE = 4
BATCH_SIZE = 256

# weight_embedding = tf.get_variable("weight_embedding", [FEATURE_SIZE, EMBEDDING_SIZE])
weight_embedding = tf.Variable(
    tf.random_normal([FEATURE_SIZE, EMBEDDING_SIZE], 0.0, 0.01), name="feature_embedding"
)
IS_training = tf.placeholder(tf.bool, name="is_training")
user_temp_input = tf.placeholder(tf.int32, shape=[None, 22], name="user_temp_input")  # 注意其中有个连续型数值

user_item_input = tf.placeholder(tf.int32, shape=[None, 174], name="user_item_input")  # 用户的历史行为

user_shop_input = tf.placeholder(tf.int32, shape=[None, 174], name="user_shop_input")

user_brand_input = tf.placeholder(tf.int32, shape=[None, 174], name="user_brand_input")

user_category_input = tf.placeholder(tf.int32, shape=[None, 174], name="user_category_input")

user_hist_len = tf.placeholder(tf.int32, shape=[None, 1], name="user_hist_len")  # 用户行为的历史真实长度(-1, 1)

# 接下来就是处理item部分的特征
item_input = tf.placeholder(tf.float32, shape=[None, 23], name="item_input")  # 维度是(batch_size, item_features)

label = tf.placeholder(tf.float32, shape=[None, 1], name="label")

# 现在对于user_temp_input处理，其中的age在其中第4个位置

user_age_input = tf.expand_dims(user_temp_input[:, 4], 1)  # 此时的维度就是(batch_size, 1)
user_age_input = tf.cast(user_age_input, tf.float32)

user_other_input = tf.concat([user_temp_input[:, 1:4], user_temp_input[:, 5:]], axis=1)  # 去掉userID

user_other_emb = tf.nn.embedding_lookup(weight_embedding,
                                        user_other_input)  # 此时的维度是？N * K * F(Batch_size, 特征域的长度，embedding_size)

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
# 现在对于user_shop_input进行处理
user_shops_emb = tf.nn.embedding_lookup(weight_embedding,
                                        user_shop_input)  # 此时的维度就是(Batch_size, MAX_LEN, embedding_size)
# 此时下面的维度就是(batch_size, 1, embedding_size)
user_shops_embedding = SequencePoolingLayer(user_seq_embedding=user_shops_emb, user_behavior_length=user_hist_len)

# 现在对于user_brand_input进行处理
user_brands_emb = tf.nn.embedding_lookup(weight_embedding,
                                         user_brand_input)  # 此时的维度就是(Batch_size, MAX_LEN, embedding_size)
# 此时下面的维度就是(batch_size, 1, embedding_size)
user_brands_embedding = SequencePoolingLayer(user_seq_embedding=user_brands_emb, user_behavior_length=user_hist_len)

# 现在对于user_category_input进行处理
user_categorys_emb = tf.nn.embedding_lookup(weight_embedding,
                                            user_category_input)  # 此时的维度就是(Batch_size, MAX_LEN, embedding_size)
# 此时下面的维度就是(batch_size, 1, embedding_size)
user_categorys_embedding = SequencePoolingLayer(user_seq_embedding=user_categorys_emb,
                                                user_behavior_length=user_hist_len)

# 获取离散型的变量
# item_number_input = tf.concat([item_input[:, 0:11], item_input[:, 17:21], item_input[:, 22:]], axis=1)
item_number_input = tf.concat([item_input[:, 0:11], tf.expand_dims(item_input[:, 17], 1), item_input[:, 19:]], axis=1)
# print("item_number_input: ")
# print(item_number_input.shape)
# 获取连续型的变量
temp = tf.expand_dims(item_input[:, 18], 1)
item_continues_input = tf.concat([item_input[:, 11:17], temp],
                                 axis=1)  # 维度就是(bath_size, 7)
item_number_input = tf.to_int32(item_number_input)
# print("item_continues_input:")
# print(item_continues_input.shape)

item_number_embedding = tf.nn.embedding_lookup(weight_embedding,
                                               item_number_input)  # 维度就是(batch_size, 商品离散特征个数, embedding_size)

"""
现在开始组织特征构建网络
"""
# 构建用户的输入
# 处理离散的特征


user_other_emb = tf.reshape(user_other_emb, (-1, 20 * EMBEDDING_SIZE))

user_items_embedding = tf.reshape(user_items_embedding, (-1, EMBEDDING_SIZE))

user_shops_embedding = tf.reshape(user_shops_embedding, (-1, EMBEDDING_SIZE))

user_brands_embedding = tf.reshape(user_brands_embedding, (-1, EMBEDDING_SIZE))

user_categorys_embedding = tf.reshape(user_categorys_embedding, (-1, EMBEDDING_SIZE))

user_dnn_input = tf.concat([user_other_emb, user_age_input, user_items_embedding,
                            user_shops_embedding, user_brands_embedding, user_categorys_embedding], axis=1)

print("user的输出维度: ")
print(user_dnn_input.shape)

item_number_embedding = tf.reshape(item_number_embedding, (-1, 16 * EMBEDDING_SIZE))
# print("item_number_embedding:")
# print(item_number_embedding.shape)

item_dnn_input = tf.concat([item_number_embedding, item_continues_input], axis=1)
print("item的输入维度: ")
print(item_dnn_input.shape)


def user_part(x_input):
    """
    这个是deep部分的网络
    这里定义了两层
    :param x_input:为输入数据，维度就是(-1, filed_size * embedding_size)
    :return:
    """
    x_input = tf.layers.batch_normalization(x_input, name='b1', training=IS_training, reuse=tf.AUTO_REUSE)

    u_layer1 = tf.layers.dense(x_input, 256, name="u_layer1", reuse=tf.AUTO_REUSE)
    # layer1 = tf.matmul(x_input, U_W1) + U_B1
    hidden1 = tf.layers.batch_normalization(u_layer1, training=IS_training, name='b2', reuse=tf.AUTO_REUSE)
    hidden1 = tf.nn.leaky_relu(hidden1)

    user_out = tf.layers.dense(hidden1, 128, activation=tf.nn.leaky_relu, name="user_out", reuse=tf.AUTO_REUSE)

    return user_out  # 此时的维度就是(-1, 32)


def item_part(x_input):
    """
    这个是deep部分的网络
    这里定义了两层
    :param x_input:为输入数据，维度就是(-1, filed_size * embedding_size)
    :return:
    """
    x_input = tf.layers.batch_normalization(x_input, name='b3', training=IS_training, reuse=tf.AUTO_REUSE)

    i_layer1 = tf.layers.dense(x_input, 256, name="i_layer1", reuse=tf.AUTO_REUSE)
    # layer1 = tf.matmul(x_input, I_W1) + I_B1
    hidden1 = tf.layers.batch_normalization(i_layer1, training=IS_training, name='b4', reuse=tf.AUTO_REUSE)
    hidden1 = tf.nn.leaky_relu(hidden1)

    item_out = tf.layers.dense(hidden1, 128, activation=tf.nn.leaky_relu, name="item_out", reuse=tf.AUTO_REUSE)

    return item_out  # 此时的维度就是(-1, 32)


user_dnn_out = user_part(user_dnn_input)
item_dnn_out = item_part(item_dnn_input)

print("............")
print(user_dnn_out.shape)

print(">>>>>>>>>>")
print(item_dnn_out.shape)
# 接下来就是计算余玄相似度
out = Similarity(user_dnn_out=user_dnn_out, item_dnn_out=item_dnn_out, type=None)

print("out: ")
print(out.shape)

# loss = tf.losses.log_loss(label, out)

loss = tf.reduce_mean(
    tf.nn.sigmoid_cross_entropy_with_logits(
        logits=out,
        labels=label)
)
optimizer = tf.train.AdamOptimizer(learning_rate=0.0006).minimize(loss)
"""
获取数据集

"""
with open('../data/train_set.pkl', 'rb') as f:
    train_set = pck.load(f)

train_model_input, train_label = gen_model_input(train_set=train_set)

train_label = np.reshape(train_label, (-1, 1))

user_temp = train_model_input['users_list']

STEPS = user_temp.shape[0] // BATCH_SIZE

user_items = train_model_input['user_item_list']
user_shops = train_model_input['user_shop_list']
user_brands = train_model_input['user_brand_list']
user_categorys = train_model_input['user_category_list']

user_behavior_length = np.reshape(train_model_input['user_hist_len'], (-1, 1))

item_feature = train_model_input['item_feature']

# user_temp = np.concatenate((user_temp, item_feature[:, 17:20]), axis=1)
item_feature = np.concatenate((item_feature[:, 0:17], item_feature[:, 20:]), axis=1)
# print(train_label[0:30])
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #
    # print(sess.run(out[0:10]))
    # print("**********")
    for epoch in range(50):
        for step in range(STEPS):
            epoch_loss, _ = sess.run([loss, optimizer],
                                     feed_dict={
                                         user_temp_input: user_temp[step * BATCH_SIZE:(step + 1) * BATCH_SIZE],
                                         user_item_input: user_items[step * BATCH_SIZE:(step + 1) * BATCH_SIZE],
                                         user_shop_input: user_shops[step * BATCH_SIZE:(step + 1) * BATCH_SIZE],
                                         user_brand_input: user_brands[
                                                           step * BATCH_SIZE:(step + 1) * BATCH_SIZE],
                                         user_category_input: user_categorys[
                                                              step * BATCH_SIZE:(step + 1) * BATCH_SIZE],
                                         user_hist_len: user_behavior_length[
                                                        step * BATCH_SIZE:(step + 1) * BATCH_SIZE],
                                         item_input: item_feature[step * BATCH_SIZE:(step + 1) * BATCH_SIZE],
                                         label: train_label[step * BATCH_SIZE:(step + 1) * BATCH_SIZE],
                                         IS_training: True})
        epoch_loss, _, uu = sess.run([loss, optimizer, out],
                                     feed_dict={user_temp_input: user_temp[step * BATCH_SIZE:],
                                                user_item_input: user_items[step * BATCH_SIZE:],
                                                user_shop_input: user_shops[step * BATCH_SIZE:],
                                                user_brand_input: user_brands[step * BATCH_SIZE:],
                                                user_category_input: user_categorys[step * BATCH_SIZE:],
                                                user_hist_len: user_behavior_length[step * BATCH_SIZE:],
                                                item_input: item_feature[step * BATCH_SIZE:],
                                                label: train_label[step * BATCH_SIZE:],
                                                IS_training: True})

        # print(uu[0:3])
        # print("_____________")
        # print(train_label[step * BATCH_SIZE:(step * BATCH_SIZE) + 3])
        if epoch % 1 == 0:
            print("epoch %s-------loss is %s" % (str(epoch), str(epoch_loss)))
            # print(epoch_loss)

    """保存好训练的网络"""
    saver = tf.train.Saver()
    saver.save(sess, '../DSSM_SAVE_MODEL/dssm_model_saver.ckpt')
    print("保存好DSSM网络............")
