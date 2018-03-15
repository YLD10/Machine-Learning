#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/15/2018 22:55
# @Author  : YLD10
# @Email   : yl1315348050@yahoo.com
# @File    : KNN.py
# @Software: PyCharm


def unpickle(file):
    import pickle

    with open(file, 'rb') as fo:
        dic = pickle.load(fo, encoding='bytes')
    return dic

