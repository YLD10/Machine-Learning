#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 3/17/2018 15:52
# @Author   : YLD10
# @Email    : yl1315348050@yahoo.com
# @File     : KNN_k_crossValidation.py
# @Software : PyCharm
# @reference: https://zhuanlan.zhihu.com/p/20894041?refer=intelligentunit
#           : http://blog.csdn.net/dream_angel_z/article/details/47110077


import pandas as pd
import numpy as np
import matplotlib as mpl
import scipy.misc as mi
import matplotlib.pyplot as plt


# 主类
class NearestNeighbor(object):
    def __init__(self):
        self.xtr = 0
        self.ytr = 0
        pass

    def train(self, x_l, y_l):
        """ x is N x D where each row is an example. y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.xtr = x_l
        self.ytr = y_l

    def predict(self, x_l, k_l=1):
        """ x is N x D where each row is an example we wish to predict label for """
        num_test = x_l.shape[0]
        # lets make sure that the output type matches the input type
        ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        # loop over all test rows
        for i_l in range(num_test):
            # print(i_l, end=' ')
            # find the nearest training image to the i'th test image
            # using the L1 distance (sum of absolute value differences)
            distances = np.sum(np.abs(self.xtr - x_l[i_l, :]), axis=1)
            # distances = np.sqrt(np.sum(np.square(self.xtr - x[i, :]), axis=1))  # L2 distance
            # print(distances.shape)
            topn_index = np.argsort(distances)[:k_l]  # get the index with top k_l small distance
            topn_label = self.ytr[topn_index]  # indexs are converted into labels
            # print(topn_label)
            tmp_list = sorted([(np.sum(topn_label == e), e) for e in set(topn_label)])  # 统计各标签的出现次数
            # print(tmp_list)
            most_label = tmp_list[-1][1]  # 取出重复次数最多的标签
            # print(most_label)
            ypred[i_l] = most_label  # predict the label of the k_l nearest example

        return ypred


# 加载pickle压缩的包
def unpickle(file):
    import pickle

    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic


# 加载cifar-10的所有数据
def load_cifar10(file_path):
    # print('yte_l: ', yte_l.shape)
    dic_d1 = unpickle(file_path + data1_file)
    dic_d2 = unpickle(file_path + data2_file)
    dic_d3 = unpickle(file_path + data3_file)
    dic_d4 = unpickle(file_path + data4_file)
    dic_d5 = unpickle(file_path + data5_file)
    dic_t = unpickle(file_path + test_file)

    # print(dic_d1)

    xtr_l = np.concatenate(
        [dic_d1[b'data'], dic_d2[b'data'], dic_d3[b'data'], dic_d4[b'data'], dic_d5[b'data']],
        axis=0)  # 多维数组按行(竖向)拼接
    # print('xtr_l: ', xtr_l.shape)
    xte_l = dic_t[b'data']
    # print('xte_l: ', xte_l.shape)

    ytr_l = np.concatenate(
        [dic_d1[b'labels'], dic_d2[b'labels'], dic_d3[b'labels'], dic_d4[b'labels'], dic_d5[b'labels']],
        axis=0)  # 一维数组默认按列(横向)拼接
    # print('ytr_l: ', ytr_l.shape)
    yte_l = np.array(dic_t[b'labels'])

    return xtr_l, ytr_l, xte_l, yte_l


# 只取前n个图像数据进行训练和测试
def cut_x_y(xtr_l, ytr_l, xte_l, yte_l, n_l=100):
    return xtr_l[:n_l, :], ytr_l[:n_l], xte_l[:n_l, :], yte_l[:n_l]


# 划分数组
def splist(arr, c_l):
    tmp_l = []
    step = arr.shape[0] // c_l
    for i_l in range(c):
        if i_l != c - 1:
            tmp_l.append(arr[i_l * step:(i_l + 1) * step])
        else:
            tmp_l.append(arr[i_l * step:])
    # print('tmp_l:\n', tmp_l)

    return tmp_l


# # 产生交叉验证数组array([(训练集1)，(验证集1)], [(训练集2)，(验证集2)], ...)
# def get_cross_validation_set(xtr_l, ytr_l, c):
#     """训练集：xtr_c, ytr_c；验证集: xval_c, yval_c"""
#     xtr_s = splist(xtr_l, c)
#     ytr_s = splist(ytr_l, c)
#     # print('xtr_s:\n', xtr_s)
#     # print('ytr_s:\n', ytr_s)
#     x_c = np.zeros((c, 2), dtype=tuple)
#     for i in range(c):
#         xtr_c = []
#         ytr_c = []
#         # 取第i份作为验证集
#         xval_c = xtr_s[i]
#         yval_c = ytr_s[i]
#         for j in range(c):
#             # 取除第i份以外的其他份作为训练集
#             if i != j:
#                 xtr_c.extend(xtr_s[j])
#                 ytr_c.extend(ytr_s[j])
#
#         # print('xtr_c:\n', xtr_c)
#         # print('ytr_c:\n', ytr_c)
#         x_c[i] = [(np.array(xtr_c), ytr_c), (np.array(xval_c), yval_c)]
#
#     return x_c


# 产生交叉验证数集xtr_cross, ytr_cross, xval_cross, yval_cross
def get_cross_validation_set(xtr_l, ytr_l, c_l):
    """训练集：xtr_cross, ytr_cross；验证集: xval_cross, yval_cross"""
    xtr_s = splist(xtr_l, c_l)
    ytr_s = splist(ytr_l, c_l)
    xtr_cross = []
    xval_cross = []
    ytr_cross = []
    yval_cross = []
    # print('xtr_s:\n', xtr_s)
    # print('ytr_s:\n', ytr_s)
    for i_l in range(c_l):
        xtr_c = []
        ytr_c = []
        # 取第i份作为验证集
        xval_c = xtr_s[i_l]
        yval_c = ytr_s[i_l]
        for j_l in range(c_l):
            # 取除第i份以外的其他份作为训练集
            if i_l != j_l:
                xtr_c.extend(xtr_s[j_l])
                ytr_c.extend(ytr_s[j_l])

        # print('xtr_c:\n', xtr_c)
        # print('ytr_c:\n', ytr_c)
        xtr_cross.append(np.array(xtr_c))
        xval_cross.append(np.array(xval_c))
        ytr_cross.append(ytr_c)
        yval_cross.append(yval_c)

    return np.array(xtr_cross), np.array(xval_cross), np.array(ytr_cross), np.array(yval_cross)


if __name__ == '__main__':
    # 设置控制台显示宽度以及取消科学计数法
    pd.set_option('display.width', 300)
    np.set_printoptions(suppress=True)

    # 解决图表的中文以及负号的乱码问题
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    MODEL_PATH = './'
    CIFAR_PATH = 'cifar-10/cifar-10-python/cifar-10-batches-py/'

    batches = 'batches.meta'
    data1_file = 'data_batch_1'
    data2_file = 'data_batch_2'
    data3_file = 'data_batch_3'
    data4_file = 'data_batch_4'
    data5_file = 'data_batch_5'
    test_file = 'test_batch'

    # 保存第一张图片，验证图片数据是否准确提取
    # img_dict = unpickle(MODEL_PATH + CIFAR_PATH + data1_file)
    #
    # # print(img_dict[b'data'].dtype)
    #
    # r = np.zeros((32, 32), dtype=img_dict[b'data'].dtype)
    # g = np.zeros((32, 32), dtype=img_dict[b'data'].dtype)
    # b = np.zeros((32, 32), dtype=img_dict[b'data'].dtype)
    #
    # r = img_dict[b'data'][:1, :1024].reshape(32, 32)
    # g = img_dict[b'data'][:1, 1024:2048].reshape(32, 32)
    # b = img_dict[b'data'][:1, 2048:].reshape(32, 32)
    #
    # print(r)
    # print(g)
    # print(b)
    #
    # img = np.dstack([r, g, b])
    #
    # print(img.shape)
    #
    # mi.imsave('one.jpg', img)

    # 设置要训练的数据量
    n = 2000
    # 设置交叉验证折数
    c = 5

    xtr, ytr, xte, yte = load_cifar10(MODEL_PATH + CIFAR_PATH)
    xtr, ytr, xte, yte = cut_x_y(xtr, ytr, xte, yte, n)  # 减少数据量至ntr个，缩短时间

    # 产生交叉验证集
    x_train, x_vali, y_train, y_vali = get_cross_validation_set(xtr, ytr, c)
    # print(x_train)
    print('x_train: ', x_train.shape)
    print('x_vali: ', x_vali.shape)
    print('y_train: ', y_train.shape)
    print('y_vali: ', y_vali.shape)

    # find hyperparameters that work best on the validation set
    validation_accuracies = []
    # 保存每一次验证结果以供绘图
    vali_acc = []
    for k in [1, 3, 5, 7, 9, 11, 13, 15, 20, 50, 100]:
        # 保存每一次验证结果的平均值以供绘图
        validation_acc = []
        for i in range(c):
            # use a particular value of k and evaluation on validation data
            nn = NearestNeighbor()  # create a Nearest Neighbor classifier class
            nn.train(x_train[i], y_train[i])  # train the classifier on the training images and labels
            # here we assume a modified NearestNeighbor class that can take a k as input
            yval_predict = nn.predict(x_vali[i], k_l=k)  # predict labels on the validation images
            # and now print the classification accuracy, which is the average number
            # of examples that are correctly predicted (i.e. label matches)
            acc = np.mean(yval_predict == y_vali[i])
            print('k = %d, accuracy: %f' % (k, acc,))
            validation_acc.append(acc)
            # 保存验证结果
            vali_acc.append((k, acc))

        # keep track of what works on the validation set
        validation_accuracies.append((k, np.mean(validation_acc)))

    validation_accuracies = np.array(validation_accuracies)
    vali_acc = np.array(vali_acc)
    print('validation_accuracies:\n', validation_accuracies)
    print('vali_acc:\n', vali_acc)

    # 绘图看结果
    fig = plt.figure()
    fig.set(alpha=0.2)  # 设定图表颜色 alpha 参数，即透明度
    plt.title('Cross-validation on k')
    plt.xlabel('k')
    plt.ylabel('Cross-validation accuracy')
    plt.scatter(vali_acc[:, 0], vali_acc[:, 1])
    plt.plot(validation_accuracies[:, 0], validation_accuracies[:, 1], '^-', color='r', label='accuracies mean')
    plt.vlines(validation_accuracies[:, 0], 0.9 * validation_accuracies[:, 1], 1.1 * validation_accuracies[:, 1],
               colors='b', linestyles='dashed')
    plt.legend(loc='best')
    plt.show()
