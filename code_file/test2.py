import pandas as pd
import numpy as np
from scipy.sparse import lil_matrix
import scipy
import pickle as pck
# from DssmModel.gen_data_input import gen_model_input
from DssmModel.gen_16_21_input import gen_model_input
import tensorflow as tf

# a = tf.constant([[1., 2.], [3., 4.]])
#
# print(tf.norm(a, axis=1))  # L1_norm,axis=1,行
#
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#
#     tt = sess.run(tf.norm(a, axis=1))
#     print(tt)
with open('../data/test_set_10.pkl', 'rb') as f:
    train_set = pck.load(f)

print(len(train_set))  # 278216, 278216
kk = 1
for line in train_set:
    print(line)
    print("\n")
    kk = kk + 1
    if kk == 10:
        break

# print(len(train_set))  # 3753135

# with open('../data/test_set.pkl', 'rb') as f:
#     test_set = pck.load(f)

# print(test_set.shape)  # 196619
# train_model_input, train_label = gen_model_input(train_set=train_set)
# #
# train_label = np.reshape(train_label, (-1, 1))
# #
# print(train_label[:1000])

# user_temp = train_model_input['users_list']
#
#
# user_items = train_model_input['user_item_list']
# user_shops = train_model_input['user_shop_list']
# user_brands = train_model_input['user_brand_list']
# user_categorys = train_model_input['user_category_list']
# print("user_items_____________________")
# print(user_items[0:3])
#
# print("user_shops_____________________")
# print(user_shops[0:3])
#
# print("user_brands_____________________")
# print(user_brands[0:3])
#
# print("user_categorys_____________________")
# print(user_categorys[0:3])
#
#
# user_behavior_length = np.reshape(train_model_input['user_hist_len'], (-1, 1))
#
# print("user_behavior_length_____________")
# print(user_behavior_length[0:3])
#
# item_feature = train_model_input['item_feature']
# print("item_feature_____________")
# print(item_feature[0:3])

# user_temp = np.concatenate((user_temp, item_feature[:, 17:20]), axis=1)
# item_feature = np.concatenate((item_feature[:, 0:17], item_feature[:, 20:]), axis=1)
