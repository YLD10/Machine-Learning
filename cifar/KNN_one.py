#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time     : 3/15/2018 22:55
# @Author   : YLD10
# @Email    : yl1315348050@yahoo.com
# @File     : KNN_one.py
# @Software : PyCharm
# @reference: https://zhuanlan.zhihu.com/p/20894041?refer=intelligentunit

import pandas as pd
import numpy as np
import matplotlib as mpl
import scipy.misc as mi


class NearestNeighbor(object):
    def __init__(self):
        self.xtr = np.zeros((1, 1))
        self.ytr = np.zeros((1, 1))
        pass

    def train(self, x_l, y_l):
        """ x is N x D where each row is an example. y is 1-dimension of size N """
        # the nearest neighbor classifier simply remembers all the training data
        self.xtr = np.array(x_l)
        self.ytr = np.array(y_l)

    def predict(self, x_l, k_l=1):
        """ x is N x D where each row is an example we wish to predict label for """
        num_test = x_l.shape[0]
        # lets make sure that the output type matches the input type
        ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        # loop over all test rows
        for i_l in range(num_test):
            # print(i_l, end=' ')
            # find the nearest training image to the i_l'th test image
            # using the L1 distance (sum of absolute value differences)
            distances = np.sum(np.abs(self.xtr - x_l[i_l, :]), axis=1)
            # print(distances)
            min_index = np.argmin(distances)   # get the index with smallest distance
            # print(min_index)
            ypred[i_l] = self.ytr[min_index]  # predict the label of the nearest example

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

    print(dic_d1)

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
    n = 500

    xtr, ytr, xte, yte = load_cifar10(MODEL_PATH + CIFAR_PATH)
    xtr, ytr, xte, yte = cut_x_y(xtr, ytr, xte, yte, n)  # 减少数据量至n个，缩短时间

    print('xtr: ', xtr.shape)  # 50000 x 3072
    print('xte: ', xte.shape)  # 10000 x 3072
    print('ytr: ', ytr.shape)  # 1 x 50000
    print('yte: ', yte.shape)  # 1 x 10000

    nn = NearestNeighbor()  # create a Nearest Neighbor classifier class
    nn.train(xtr, ytr)  # train the classifier on the training images and labels
    yte_predict = nn.predict(xte)  # predict labels on the test images
    # and now print the classification accuracy, which is the average number
    # of examples that are correctly predicted (i.e. label matches)
    print('accuracy: %f' % (np.mean(yte_predict == yte)))
