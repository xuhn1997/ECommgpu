import tensorflow as tf
import numpy as np
import pandas as pd

# def weight(shape):
#     w_stddev = 1. / tf.sqrt(shape[0] / 2.)  # shape[0]指矩阵的行数
#     return tf.random_normal(shape=shape, stddev=w_stddev)
#
# D_W1 = tf.Variable(weight([, 256]))
# D_B1 = tf.Variable(tf.zeros(shape=[1, 256]))


# 此文件是构建DSSM的网络层
MAX_LEN = 11  # 最大的历史长度


def DNN(dnn_input, IS_TRAING):
    """
    构建网络层
    :param IS_TRAING: 表示是否是使用dropout以及bn
    :param dnn_input:输入的网络(batch_size, input_dim)
    :return:
    """

    layer1 = tf.layers.dense(dnn_input, 1024, name="layer1")
    layer1 = tf.layers.dropout(layer1, rate=0.5, training=IS_TRAING)
    layer1 = tf.layers.batch_normalization(layer1, training=IS_TRAING)
    layer1 = tf.nn.relu(layer1)

    layer2 = tf.layers.dense(layer1, 512, name="layer2")
    layer2 = tf.layers.dropout(layer2, rate=0.5, training=IS_TRAING)
    layer2 = tf.layers.batch_normalization(layer2, training=IS_TRAING)
    layer2 = tf.nn.relu(layer2)

    layer3 = tf.layers.dense(layer2, 256, name="layer3")
    layer3 = tf.layers.dropout(layer3, rate=0.5, training=IS_TRAING)
    layer3 = tf.layers.batch_normalization(layer3, training=IS_TRAING)
    layer3 = tf.nn.relu(layer3)

    layer4 = tf.layers.dense(layer3, 128, name="layer4")
    layer4 = tf.layers.dropout(layer4, rate=0.5, training=IS_TRAING)
    layer4 = tf.layers.batch_normalization(layer4, training=IS_TRAING)
    layer4 = tf.nn.relu(layer4)

    output = tf.layers.dense(layer4, 32, activation=tf.nn.relu, name="output")

    return output  # 维度就是(batch_size, 32)


def Similarity(user_dnn_out, item_dnn_out, type='cos'):
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
        cosine_score = tf.div(cosine_score, user_norm + item_norm + 1e-8)  # 防止除0错误

    cosine_score = tf.clip_by_value(cosine_score, -1, 1.0)  # 限制输出的范围在-1， 1

    # cosine_score = tf.nn.sigmoid(cosine_score)
    return cosine_score  # 最后再经过sigmoid函数


def SequencePoolingLayer(user_seq_embedding, user_behavior_length, mode='mean'):
    """
    此函数是具有变长的embedding的处理
    :param user_seq_embedding:这个是传入的embedding的行为的矩阵，维度是(batch_size, T, embedding_size)
    :param user_behavior_length:这个是实际行为的长度维度是(batch_size, 1)
    :param mode:
    :return:
    """
    # 利用最大的历史以及用户历史行为的长度建立好mask
    mask = tf.sequence_mask(user_behavior_length, MAX_LEN, dtype=tf.float32)  # 维度是(batch_size, 1, MAX_LEN)
    # 然后转化其维度以便进行矩阵相乘
    mask = tf.transpose(mask, (0, 2, 1))  # (batch_size, MAX_LEN, 1)

    embedding_size = user_seq_embedding.shape[-1]  # 获取embedding的长度

    # 现在对于mask进行维度乘倍
    tf.tile(mask, [1, 1, embedding_size])  # 转化之后维度变成(batch_size, MAX_LEN, embedding_size), MAX_LEN也就是T

    # 将mask与用户的embedding相乘，对应位置相乘。1表示用户的行为的embedding
    if mode == 'max':
        """
        若要取用户的行为embedding中的最大值那一个代表当前用户行为的embedding的话
        """
        hist = user_seq_embedding - (1 - mask) * 1e9
        return tf.reduce_max(hist, 1,
                             keep_dims=True)  # keep_dims=True, 保持原有的shape个数，也就是当前维度为(batch_size, 1, embedding_size)

    hist = tf.reduce_sum(user_seq_embedding * mask, 1, keep_dims=False)  # 对应位置相乘之后减少维度为(batch_size, embedding_size)

    if mode == 'mean':
        """
        若想取它的均值的话处理为
        """
        hist = tf.div(hist, tf.cast(user_behavior_length, tf.float32) + 1e-8)  # 除以当前用户实际的行为的长度

    # 最后统一加上一个维度返回
    hist = tf.expand_dims(hist, axis=1)  # 维度变为(B, 1, embedding_size)

    return hist
