import pandas as pd
import warnings
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from datetime import datetime


# 各列数据去重并保存结果
def no_duplicate(data_copy):
    title = ['user_id', 'item_id', 'item_category']
    for t in title:
        colum = data_copy[t]
        # print('Before drop_duplicates = \n', colum)
        colum = colum.drop_duplicates()
        # print('After drop_duplicates = \n', colum)
        pd.DataFrame(colum).to_csv('F://TianChi//fresh_1//fresh_comp_offline//' + t + '_no_duplicate.csv')


# 时间排序并填充缺失值及保存结果
def sort_fill(path_user_copy, path_data_fill_copy):
    user = pd.read_csv(path_user_copy)
    user_id = np.array(user['user_id'])
    user_id_data = df.groupby('user_id')
    user_merge = []
    # i = 0
    for one_id in user_id:
        one = pd.DataFrame(user_id_data.get_group(one_id))
        one = one.sort_values('time')
        one = one.fillna(method='ffill')
        one = one.fillna(method='bfill')
        # if one.count()['user_geohash'] == 0:  # 有些用户没有位置信息
        #     i += 1
        user_merge.append(one)

    user_id_data = pd.concat(user_merge, ignore_index=True)
    # print(user_id_data.head())
    # print(user_id_data.shape)
    # print(i)
    pd.DataFrame(user_id_data).to_csv(path_data_fill_copy)


# 取前n行非升序数据
def top_n(df_copy, n=3, column='time'):
    return df_copy.sort_values(by=column, ascending=False)[:n]


# 生成指定时间的日期，格式：年-月-日
def date_list(start_date, end_date):
    date_l = [datetime.strftime(x, '%Y-%m-%d') for x in list(pd.date_range(start=start_date, end=end_date))]
    return date_l


# 获取用户每日行为系数的平均值
def get_user_behavior_daily_point(df_copy, top):
    base_df = df_copy[['time', 'behavior_type']].groupby(by='time').agg('sum')
    # print(base_df.head())
    # print(base_df.index)
    trend = pd.DataFrame(columns=['date', 'number'])
    i = 0
    for date in date_list(base_df.head(1).index.tolist()[0], base_df.tail(1).index.tolist()[0]):
        trend.loc[i] = [date, base_df[date].behavior_type.sum()]
        i += 1
    # print(trend.head())
    # print(trend.shape)

    # plt.plot(trend['date'], trend['number'], 'r.', label='time')
    # plt.xlabel(u'date', fontsize=16)
    # plt.ylabel(u'number', fontsize=16)
    # plt.title(u'用户在各个时间段的行为统计', fontsize=20)
    # plt.show()

    # 剔除双12前后的峰值求每日用户行为系数平均
    return int(trend[trend.number <= top].loc[:, 'number'].mean())


if __name__ == "__main__":
    '''
        天池新手赛：根据用户点击/收藏/购买记录预测下一阶段的消费 
    '''

    # 设置控制台显示宽度以及取消科学计数法
    pd.set_option('display.width', 300)
    np.set_printoptions(suppress=True)

    # 解决图表的中文以及负号的乱码问题
    mpl.rcParams['font.sans-serif'] = [u'simHei']
    mpl.rcParams['axes.unicode_minus'] = False

    # 忽略警告
    warnings.filterwarnings("ignore")

    # 相关文件路径
    path_data = 'F://TianChi//fresh_1//fresh_comp_offline//tianchi_fresh_comp_train_user.csv'
    path_user = 'F://TianChi//fresh_1//fresh_comp_offline//user_id_no_duplicate.csv'
    path_data_fill = 'F://TianChi//fresh_1//fresh_comp_offline//tianchi_fresh_comp_train_user_fill_sort.csv'

    # 加载库，载入原始数据
    # df = pd.read_csv(path_data, date_parser='time')
    # print(df.head())
    # print(df.shape)

    # 各列数据去重并保存结果
    # no_duplicate(df)

    # 按时间排序并尽量填充缺失值及保存结果
    # sort_fill(path_user, path_data_fill)

    # 加载库，载入完整数据
    df = pd.read_csv(path_data_fill)
    df = pd.DataFrame(df.iloc[:, 1:])
    df['time'] = pd.to_datetime(df['time'])
    # print(df.head())
    # print(df.count())
    # print(df.shape)
    # print(df.behavior_type.value_counts())
    # print(df.time.value_counts())

    # 按用户分组并取每组前几行数据
    # group_df = df.groupby('user_id').apply(top_n)
    # print(group_df.head())
    # print(group_df.shape)

    # 每天的用户行为系数均值
    # udp = get_user_behavior_daily_point(df, 42500)


