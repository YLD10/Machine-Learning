import pandas as pd  # 数学分析
import numpy as np  # 科学计算
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn.model_selection import learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

'''
    Kaggle Titanic Completition
'''


# 使用RandomForestClassifier填补缺失的年龄属性
def set_missing_ages(df_l):
    # 把已有的数值型特征提取出来丢进RandomForestRegressor中
    age_df = df_l[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()
    # print(known_age)
    # print(unknown_age)

    # y即目标年龄
    y_age = known_age[:, 0]

    # X即特征属性值
    x_age = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr_age = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr_age.fit(x_age, y_age)

    # 用得到的模型进行未知年龄的结果预测
    predicted_ages = rfr_age.predict(unknown_age[:, 1:])

    # 用得到的预测结果填补原缺失数据
    df_l.loc[df_l.Age.isnull(), 'Age'] = predicted_ages

    return df_l, rfr_age


# Cabin特征转换
def set_cabin_type(df_l):
    df_l.loc[df_l.Cabin.notnull(), 'Cabin'] = 'Yes'
    df_l.loc[df_l.Cabin.isnull(), 'Cabin'] = 'No'

    return df_l


# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
def plot_learning_curve(estimator, title, x_l, y_l, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = learning_curve(estimator, x_l, y_l, cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_sizes, verbose=verbose)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u'训练样本数')
        plt.ylabel(u'得分')
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color='b')
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1,
                         color='r')
        plt.plot(train_sizes, train_scores_mean, 'o-', color='b', label=u'训练集上得分')
        plt.plot(train_sizes, test_scores_mean, 'o-', color='r', label=u'交叉验证集上得分')

        plt.legend(loc='best')

        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])

    return midpoint, diff


# 设置控制台显示宽度以及取消科学计数法
pd.set_option('display.width', 300)
np.set_printoptions(suppress=True)

# 解决图表的中文以及负号的乱码问题
mpl.rcParams['font.sans-serif'] = [u'simHei']
mpl.rcParams['axes.unicode_minus'] = False

# ############################################## #

# 看看每个/多个属性和最后的Survived之间有着什么样的关系

data_train = pd.read_csv('train.csv')
data_train['Sex_Pclass'] = data_train.Sex + '_' + data_train.Pclass.map(str)
# print(data_train)
# print(data_train.columns)
# print(data_train.info())
# print(data_train.describe())

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色 alpha 参数，即透明度

# plt.subplot2grid((2, 3), (0, 0))  # 在一张大图里分列几个小图
# data_train.Survived.value_counts().plot(kind='bar')  # plots a bar graph of those who survied vs those who did not
# plt.title(u'获救情况（1为获救）')  # puts a title on our graph
# plt.ylabel(u'人数')
#
# plt.subplot2grid((2, 3), (0, 1))
# data_train.Pclass.value_counts().plot(kind='bar')
# plt.title(u'乘客等级分布')
# plt.ylabel(u'人数')
#
# plt.subplot2grid((2, 3), (0, 2))
# plt.scatter(data_train.Survived, data_train.Age)
# plt.ylabel(u'年龄')  # sets the y axis label
# plt.grid(b=True, which='major', axis='y')  # formats the grid line style of our graphs
# plt.title(u'按年龄看获救分布（1为获救）')
#
# plt.subplot2grid((2, 3), (1, 0), colspan=2)
# data_train.Age[data_train.Pclass == 1].plot(
#     kind='kde')  # plots a kernel desnsity estimate of the subset of the  1st class passange's age
# data_train.Age[data_train.Pclass == 2].plot(kind='kde')
# data_train.Age[data_train.Pclass == 3].plot(kind='kde')
# plt.xlabel(u'年龄')  # plots an axis label
# plt.ylabel(u'密度')
# plt.title(u'各等级的乘客年龄分布')
# plt.legend((u'头等舱', u'2等舱', u'3等舱'), loc='best')  # sets our legend for our graph
#
# plt.subplot2grid((2, 3), (1, 2))
# data_train.Embarked.value_counts().plot(kind='bar')
# plt.title(u'各登船口岸上船人数')
# plt.ylabel(u'人数')
# plt.show()

# #################################################### #

# 看看乘客等级的获救情况

# Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
# print(Survived_0)
# print(Survived_1)
# df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
# df.plot(kind='bar', stacked=True)
# plt.title(u'各乘客等级的获救情况')
# plt.xlabel(u'乘客等级')
# plt.ylabel(u'人数')
# plt.show()

# 看看各登录港口的获救情况

# Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
# print(Survived_0)
# print(Survived_1)
# df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
# df.plot(kind='bar', stacked=True)
# plt.title(u'各登录港口乘客的获救情况')
# plt.xlabel(u'登录港口')
# plt.ylabel(u'人数')
# plt.show()

# 看看各性别的获救情况

# Survived_0 = data_train.Sex[data_train.Survived == 0].value_counts()
# Survived_1 = data_train.Sex[data_train.Survived == 1].value_counts()
# print(Survived_0)
# print(Survived_1)
# df = pd.DataFrame({u'获救': Survived_1, u'未获救': Survived_0})
# df.plot(kind='bar', stacked=True)
# plt.title(u'按性别看获救情况')
# plt.xlabel(u'性别')
# plt.ylabel(u'人数')
# plt.show()

# 看看各种舱级别情况下各性别的获救情况

# plt.title(u'根据舱等级和性别的获救情况')
#
# ax1 = fig.add_subplot(141)
# cou = data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts()
# print(cou)
# cou.plot(kind='bar', label='female high class', color='#FA2479')
# ax1.set_xticklabels([u'获救', u'未获救'], rotation=0)
# ax1.legend([u'女性/高级舱'], loc='best')
#
# ax2 = fig.add_subplot(142, sharey=ax1)
# cou = data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts()
# print(cou)
# cou.plot(kind='bar', label='female low class', color='pink')
# ax2.set_xticklabels([u'获救', u'未获救'], rotation=0)
# ax2.legend([u'女性/低级舱'], loc='best')
#
# ax3 = fig.add_subplot(143, sharey=ax1)
# cou = data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts()
# print(cou)
# cou.plot(kind='bar', label='male high class', color='lightblue')
# ax3.set_xticklabels([u'未获救', u'获救'], rotation=0)
# ax3.legend([u'男性/高级舱'], loc='best')
#
# ax4 = fig.add_subplot(144, sharey=ax1)
# cou = data_train.Survived[(data_train.Sex == 'male') & (data_train.Pclass == 3)].value_counts()
# print(cou)
# cou.plot(kind='bar', label='male low class', color='steelblue')
# ax4.set_xticklabels([u'未获救', u'获救'], rotation=0)
# ax4.legend([u'男性/低级舱'], loc='best')
#
# plt.show()

# ###################################################################################################

# 有堂兄弟姐妹和父母小孩的人的获救情况

# g = data_train.groupby(['SibSp', 'Survived'])
# # print(g.count())
# df = pd.DataFrame(g.count()['PassengerId'])
# print(df)

# g = data_train.groupby(['Parch', 'Survived'])
# # print(g.count())
# df = pd.DataFrame(g.count()['PassengerId'])
# print(df)

# ################################################### #

# ticket是船票编号，具有唯一性，和最后的结果没有太大的关系，不纳入考虑的特征范畴
# cabin只有204个乘客有值，先看看它的一个分布

# print(data_train.Cabin.value_counts())

# cabin的值计数太分散了，绝大多数cabin值只出现一次。感觉上作为类目，加入特征
# 未必有效。先看看这个值的有无对于survived的分布情况的影响如何

# Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
# Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
# df = pd.DataFrame({u'有': Survived_cabin, u'无': Survived_nocabin}).transpose()
# print(df)
# df.plot(kind='bar', stacked=True)
# plt.title(u'按Cabin有无看获救情况')
# plt.xlabel(u'Cabin有无')
# plt.ylabel(u'人数')
# plt.show()

# ###################################################################################### #

# 先从最突出的数据属性开始吧，对，Cabin和Age，有丢失数据实在是对下一步工作影响太大。
# 先说Cabin，暂时就按照刚才说的，按Cabin有无数据，将这个属性处理成Yes和No两种类型吧。
#
# 再说Age：
# 通常遇到缺值的情况，我们会有几种常见的处理方式
#
# 如果缺值的样本占总数比例极高，我们可能就直接舍弃了，作为特征加入的话，可能反倒带入noise，影响最后的结果了
# 如果缺值的样本适中，而该属性非连续值特征属性(比如说类目属性)，那就把NaN作为一个新类别，加到类别特征中
# 如果缺值的样本适中，而该属性为连续值特征属性，有时候我们会考虑给定一个step(比如这里的age，我们可以考虑每隔2/3岁为一个步长)，
# 然后把它离散化，之后把NaN作为一个type加到属性类目中。
# 有些情况下，缺失的值个数并不是特别多，那我们也可以试着根据已有的值，拟合一下数据，补充上。
# 本例中，后两种处理方式应该都是可行的，我们先试试拟合补全吧(虽然说没有特别多的背景可供我们拟合，这不一定是一个多么好的选择)
# 我们这里用scikit-learn中的RandomForest来拟合一下缺失的年龄数据

# origin_data_train = data_train.copy()
data_train, rfr = set_missing_ages(data_train)
data_train = set_cabin_type(data_train)
# print(data_train)

# ############################################################################################################# #

# 因为逻辑回归建模时，需要输入的特征都是数值型特征
# 我们先对类目型的特征离散/因子化
# 以Cabin为例，原本一个属性维度，因为其取值可以是['yes','no']，而将其平展开为'Cabin_yes','Cabin_no'两个属性
# 原本Cabin取值为yes的，在此处的'Cabin_yes'下取值为1，在'Cabin_no'下取值为0
# 原本Cabin取值为no的，在此处的'Cabin_yes'下取值为0，在'Cabin_no'下取值为1
# 我们使用pandas的get_dummies来完成这个工作，并拼接在原来的data_train之上，如下所示

dummies_cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')
dummies_embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')
dummies_sex = pd.get_dummies(data_train['Sex'], prefix='Sex')
dummies_pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')
dummies_sex_pclass = pd.get_dummies(data_train['Sex_Pclass'], prefix='Sex_Pclass')

df = pd.concat([data_train, dummies_cabin, dummies_embarked, dummies_sex, dummies_pclass, dummies_sex_pclass], axis=1)
# df = pd.concat([data_train, dummies_cabin, dummies_embarked, dummies_sex, dummies_pclass], axis=1)
# print(df)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Sex_Pclass'], axis=1, inplace=True)
# df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
# print(df)

# ################################################################################################## #

# 接下来我们要接着做一些数据预处理的工作，比如scaling，将一些变化幅度较大的特征化到[-1,1]之内
# 这样可以加速logistic regression的收敛
scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'].values.reshape(-1, 1))
df['Age_scaled'] = scaler.fit_transform(df['Age'].values.reshape(-1, 1))  # , age_scale_param)
fare_scale_param = scaler.fit(df['Fare'].values.reshape(-1, 1))
df['Fare_scaled'] = scaler.fit_transform(df['Fare'].values.reshape(-1, 1))  # , fare_scale_param)
# print(df)

# 我们把需要的feature字段取出来，转成numpy格式，使用scikit-learn中的LogisticRegression建模
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()
# print(train_df)
# print(train_np)

# y即Survival结果
y = train_np[:, 0]

# x即特征属性值
x = train_np[:, 1:]

# fit到LogisticRegression之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(x, y)

print(clf)
print(x.shape)

# 以上是训练集的模型训练
# ************************************************************************************ #
# 接下来是用上面训练好的模型进行测试

# 读取测试集数据
data_test = pd.read_csv('test.csv')
data_test.loc[data_test.Fare.isnull(), 'Fare'] = 0
data_test['Sex_Pclass'] = data_test.Sex + '_' + data_test.Pclass.map(str)
# print(data_test)
# print(data_test.columns)
# print(data_test.info())
# print(data_test.describe())

# 接着我们对test_data做和train_data中一致的特征变换
# 用scikit-learn中的RandomForest来拟合一下缺失的年龄数据
tmd_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

null_age = tmd_df[data_test.Age.isnull()].as_matrix()
x = null_age[:, 1:]
predicted_t_ages = rfr.predict(x)
data_test.loc[data_test.Age.isnull(), 'Age'] = predicted_t_ages

data_test = set_cabin_type(data_test)
# print(data_test)

# 我们先对类目型的特征离散/因子化
dummies_cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
dummies_embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
dummies_sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
dummies_pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')
dummies_sex_pclass = pd.get_dummies(data_test['Sex_Pclass'], prefix='Sex_Pclass')

df_test = pd.concat([data_test, dummies_cabin, dummies_embarked, dummies_sex, dummies_pclass, dummies_sex_pclass], axis=1)
# df_test = pd.concat([data_test, dummies_cabin, dummies_embarked, dummies_sex, dummies_pclass], axis=1)
# print(df_test)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked', 'Sex_Pclass'], axis=1, inplace=True)
# df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
# print(df_test)

# 接着将一些变化幅度较大的特征化到[-1,1]之内
# age_scale_param = scaler.fit(df_test['Age'].values.reshape(-1, 1))
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'].values.reshape(-1, 1), age_scale_param)
# fare_scale_param = scaler.fit(df_test['Fare'].values.reshape(-1, 1))
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'].values.reshape(-1, 1), fare_scale_param)
# print(df_test)

test = df_test.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
result.to_csv('logistic_regression_predictions2.csv', index=False)
# result.to_csv('logistic_regression_predictions.csv', index=False)

# 以上是baseline model,sorce=0.76555
# -------------------------------------------------------------------------------------------------------------------- #
# 下面用scikit-learn里面的learning_curve来帮我们分辨我们模型的状态（过拟合or欠拟合）

# plot_learning_curve(clf, u'学习曲线', x, y)

# model并不处于overfitting的状态(overfitting的表现一般是训练集上得分高，而交叉验证集上要低很多，中间的gap比较大)。
# 因此我们可以再做些feature engineering的工作，添加一些新产出的特征或者组合特征到模型中。

# 要做交叉验证(cross validation)来判断特征的优化是否是有用的
# 通常情况下，这么做cross validation：把train.csv分成两部分，一部分用于训练我们需要的模型，另外一部分数据上看我们预测算法的效果。
# 我们可以用scikit-learn的cross_validation来完成这个工作
# 在此之前，咱们可以看看现在得到的模型的系数，因为系数和它们最终的判定能力强弱是正相关的

# print(pd.DataFrame({'colums': list(train_df.columns)[1:], 'coef': list(clf.coef_.T)}))

# Sex属性，如果是female会极大提高最后获救的概率，而male会很大程度拉低这个概率。
# Pclass属性，1等舱乘客最后获救的概率会上升，而乘客等级为3会极大地拉低这个概率。
# 有Cabin值会很大程度拉升最后获救概率(这里似乎能看到了一点端倪，事实上从最上面的有无Cabin记录的Survived分布图上看出，
# 即使有Cabin记录的乘客也有一部分遇难了，估计这个属性上我们挖掘还不够)
# Age是一个负相关，意味着在我们的模型里，年龄越小，越有获救的优先权(还得回原数据看看这个是否合理）
# 有一个登船港口S会很大程度拉低获救的概率，另外俩港口压根就没啥作用(这个实际上非常奇怪，
# 因为我们从之前的统计图上并没有看到S港口的获救率非常低，所以也许可以考虑把登船港口这个feature去掉试试)。
# 船票Fare有小幅度的正相关(并不意味着这个feature作用不大，有可能是我们细化的程度还不够，
# 举个例子，说不定我们得对它离散化，再分至各个乘客等级上？)

# 使用交叉验证评估特征优化后的效果

# 简单看看打分情况
# clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin.*|Embarked_.*|Sex_.*|Pclass_.*')
# # print(all_data)
# x = all_data.as_matrix()[:, 1:]
# y = all_data.as_matrix()[:, 0]
# print(cross_val_score(clf, x, y, cv=5))
#
# # 分割数据
# split_train, split_cv = train_test_split(df, test_size=0.3, random_state=0)
# print(split_cv)
# train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin.*|Embarked_.*|Sex_.*|Pclass_.*')
# # 生成模型
# clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
# clf.fit(train_df.as_matrix()[:, 1:], train_df.as_matrix()[:, 0])
#
# # 对cross validation数据进行预测
# cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin.*|Embarked_.*|Sex_.*|Pclass_.*')
# predictions = clf.predict(cv_df.as_matrix()[:, 1:])
# print(u'源数据集中预测错误的样本：')
#
# # 找出预测有误的样本源特征
# bad_cases = origin_data_train.loc[origin_data_train['PassengerId'].isin(
#     split_cv[predictions != cv_df.as_matrix()[:, 0]]['PassengerId'].values)]
#
# print(bad_cases)

# ------------------------------------------------------------------------------------------------------------------- #

# 特征优化处理
# Age属性不使用现在的拟合方式，而是根据名称中的『Mr』『Mrs』『Miss』等的平均值进行填充。
# Age不做成一个连续值属性，而是使用一个步长进行离散化，变成离散的类目feature。
# Cabin再细化一些，对于有记录的Cabin属性，我们将其分为前面的字母部分(我猜是位置和船层之类的信息) 和 后面的数字部分
# (应该是房间号，有意思的事情是，如果你仔细看看原始数据，你会发现，这个值大的情况下，似乎获救的可能性高一些)。
# Pclass和Sex俩太重要了，我们试着用它们去组出一个组合属性来试试，这也是另外一种程度的细化。
# 单加一个Child字段，Age<=12的，设为1，其余为0(你去看看数据，确实小盆友优先程度很高啊)
# 如果名字里面有『Mrs』，而Parch>1的，我们猜测她可能是一个母亲，应该获救的概率也会提高，
# 因此可以多加一个Mother字段，此种情况下设为1，其余情况下设为0
# 登船港口可以考虑先去掉试试(Q和C本来就没权重，S有点诡异)
# 把堂兄弟/兄妹 和 Parch 还有自己 个数加在一起组一个Family_size字段(考虑到大家族可能对最后的结果有影响)
# Name是一个我们一直没有触碰的属性，我们可以做一些简单的处理，
# 比如说男性中带某些字眼的(‘Capt’, ‘Don’, ‘Major’, ‘Sir’)可以统一到一个Title，女性也一样。

# print(data_train[data_train['Name'].str.contains('Major')])
# 已在上方代码里加入了Sex_Pclass特征优化
