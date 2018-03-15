#!/usr/bin/python
# -*- coding:utf-8 -*-

import csv
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV
from pprint import pprint


def no_duplicate(data_copy):
    title = ['user_id', 'item_id', 'item_category']
    for t in title:
        colum = data_copy[t]
        # print('Before drop_duplicates = \n', colum)
        colum = colum.drop_duplicates()
        # print('After drop_duplicates = \n', colum)
        pd.DataFrame(colum).to_csv('F://TianChi//fresh_1//fresh_comp_offline//' + t + '_no_duplicate.csv')


if __name__ == "__main__":
    pd.set_option('display.width', 300)
    np.set_printoptions(suppress=True)
    path_data = 'F://TianChi//fresh_1//fresh_comp_offline//tianchi_fresh_comp_train_user.csv'
    path_data_merge = 'F://TianChi//fresh_1//fresh_comp_offline//user_and_item.csv'
    path_item_all = 'F://TianChi//fresh_1//fresh_comp_offline//tianchi_fresh_comp_train_item.csv'
    path_user = 'F://TianChi//fresh_1//fresh_comp_offline//user_id_no_duplicate.csv'
    path_item = 'F://TianChi//fresh_1//fresh_comp_offline//item_id_no_duplicate.csv'
    path_item_category = 'F://TianChi//fresh_1//fresh_comp_offline//item_category_no_duplicate.csv'
    path_data_fill = 'F://TianChi//fresh_1//fresh_comp_offline//tianchi_fresh_comp_train_user_fill_sort.csv'

    columns = ['user_id', 'item_id', 'behavior_type', 'user_geohash', 'item_category', 'time']

    # pandas读入

    # user_id, item_id, behavior_type, user_geohash, item_category, time
    data = pd.read_csv(path_data_fill)
    data = data.iloc[:, 1:7]

    # user_id, item_id, behavior_type, user_geohash, item_category, time, item_geohash
    # data_merge = pd.read_csv(path_data_merge)

    # no_duplicate(data)  # 分离各列数据项并做去重处理

    # 缺失值填充并排序
    # user = pd.read_csv(path_user)
    # user_id = np.array(user['user_id'])
    #
    user_all_data = data.groupby('user_id')
    # user_merge = []
    # for id in user_id:
    data = pd.DataFrame(user_all_data.get_group(10001082))
    print()
    #     one = one.sort_values('time')
    #     one = one.fillna(method='ffill')
    #     one = one.fillna(method='bfill')
    #     user_merge.append(one)

    #
    # user_all_data = pd.concat(user_merge)
    # print(user_all_data)
    # pd.DataFrame(user_all_data).to_csv(path_data_fill)
    '''data[columns[3]] = pd.Categorical(data[columns[3]]).codes
    x = data[[columns[0], columns[1], columns[4]]]
    y = data[columns[2]]
    print(x, y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.7, random_state=1)
    print(x_test.shape, y_test.shape)

    model = Lasso()
    alpha_can = np.logspace(-3, 2, 10)
    np.set_printoptions(suppress=True)
    print('alpha_can = ', alpha_can)
    lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
    lasso_model.fit(x_train, y_train)
    print('超参数：\n', lasso_model.best_params_)

    y_hat = lasso_model.predict(x_test)
    print(lasso_model.score(x_test, y_test))
    mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    print(mse, rmse)'''
    # 合并user以及item表
    # item_all = pd.read_csv(path_item_all)
    # data = pd.merge(data, item_all[['item_id', 'item_geohash']], on=['item_id'])
    # pd.DataFrame(data).to_csv(path_data_merge)

    # 读取用户标识
    # user = pd.read_csv(path_user)
    # user_id = np.array(user['user_id'])
    # user_num = user_id.size

    # 读取商品标识
    # item = pd.read_csv(path_item)
    # item_id = np.array(item['item_id'])
    # item_num = item_id.size

    # 读取商品分类标识
    # category = pd.read_csv(path_item_category)
    # item_category = np.array(category['item_category'])
    # item_category_num = item_category.size

    # mpl.rcParams['font.sans-serif'] = [u'simHei']
    # mpl.rcParams['axes.unicode_minus'] = False
    #
    # # 绘制1
    # plt.figure(facecolor='w')
    # plt.plot(user_data[0]['time'], user_data[0]['item_id'], 'ro', label='item_id')
    # plt.plot(user_data[0]['time'], user_data[0]['item_category'], 'r+', label='item_category')
    # plt.legend(loc='lower right')
    # plt.xlabel(u'行为时间', fontsize=16)
    # plt.ylabel(u'商品标识和商品分类标识', fontsize=16)
    # plt.title(u'商品与行为时间对比数据', fontsize=20)
    # plt.grid()
    # plt.show()
    #
    # # 绘制2
    # plt.figure(facecolor='w', figsize=(9, 10))
    # plt.subplot(311)
    # plt.plot(data['TV'], y, 'ro')
    # plt.title('TV')
    # plt.grid()
    # plt.subplot(312)
    # plt.plot(data['Radio'], y, 'g^')
    # plt.title('Radio')
    # plt.grid()
    # plt.subplot(313)
    # plt.plot(data['Newspaper'], y, 'b*')
    # plt.title('Newspaper')
    # plt.grid()
    # plt.tight_layout()
    # plt.show()
    #
    # x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.8, random_state=1)
    # print(type(x_test))
    # print(x_train.shape, y_train.shape)
    # linreg = LinearRegression()
    # model = linreg.fit(x_train, y_train)
    # print(model)
    # print(linreg.coef_, linreg.intercept_)
    #
    # order = y_test.argsort(axis=0)
    # y_test = y_test.values[order]
    # x_test = x_test.values[order, :]
    # y_hat = linreg.predict(x_test)
    # mse = np.average((y_hat - np.array(y_test)) ** 2)  # Mean Squared Error
    # rmse = np.sqrt(mse)  # Root Mean Squared Error
    # print('MSE = ', mse)
    # print('RMSE = ', rmse)
    # print('R2 = ', linreg.score(x_train, y_train))
    # print('R2 = ', linreg.score(x_test, y_test))
    #
    # plt.figure(facecolor='w')
    # t = np.arange(len(x_test))
    # plt.plot(t, y_test, 'r-', linewidth=2, label=u'真实数据')
    # plt.plot(t, y_hat, 'g-', linewidth=2, label=u'预测数据')
    # plt.legend(loc='upper right')
    # plt.title(u'线性回归预测销量', fontsize=18)
    # plt.grid(b=True)
    # plt.show()
