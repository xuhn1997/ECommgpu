import tensorflow as tf
import numpy as np


# 建立权重函数
def weight(shape):
    w_stddev = 1. / tf.sqrt(shape[0] / 2.)  # shape[0]指矩阵的行数
    return tf.random_normal(shape=shape, stddev=w_stddev)


U_W1 = tf.Variable(weight([12, 1]))
U_B1 = tf.Variable(tf.zeros(shape=[1, 1]))

# p = tf.Variable(tf.random_normal([10, 3]))  # 生成10*1的张量

p = tf.get_variable(name='weights', shape=[10, 3])
b = tf.nn.embedding_lookup(p, [[0, 1, 2, 3]])  # 查找张量中的序号为1和3的
print(b.shape)
b = tf.reshape(b, (-1, 3 * 4))
y = tf.matmul(tf.cast(b, tf.float32), U_W1) + U_B1
label = [1.]
label = np.reshape(label, (-1, 1))
# loss = tf.nn.sigmoid_cross_entropy_with_logits(
#     logits=y,
#     labels=label)
loss = tf.reduce_mean(tf.square(y - label))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999,
                                   epsilon=1e-8).minimize(loss)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(p))
    print(p)
    print(type(p))

    for epoch in range(30):
        loss_ = sess.run([loss, optimizer])

        if epoch % 1 == 0:
            print(loss_)

    print(sess.run(b))
    # print(c)
    print(sess.run(p))
