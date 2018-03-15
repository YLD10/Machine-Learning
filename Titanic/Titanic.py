import numpy as np
import pandas as pd
from pandas import DataFrame
from patsy import dmatrices
import string
from operator import itemgetter
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.externals import joblib
import warnings

warnings.filterwarnings("ignore")

# Read configuration parameters

train_file = "train.csv"
MODEL_PATH = "./"
test_file = "test.csv"
SUBMISSION_PATH = "./"
seed = 0

print(train_file, seed)


# 输出得分
def report(grid_scores, n_top=3):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
            score.mean_validation_score,
            np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")


# 清理和处理数据
def substrings_in_string(big_string, substrings):
    for substring in substrings:
        if big_string.find(substring) != -1:
            return substring
    print(big_string)
    return np.nan


le = preprocessing.LabelEncoder()
enc = preprocessing.OneHotEncoder()


def clean_and_munge_data(df_copy):
    # 处理缺省值
    df_copy.Fare = df_copy.Fare.map(lambda x: np.nan if x == 0 else x)
    # 处理一下名字，生成Title字段
    title_list = ['Mrs', 'Mr', 'Master', 'Miss', 'Major', 'Rev',
                  'Dr', 'Ms', 'Mlle', 'Col', 'Capt', 'Mme', 'Countess',
                  'Don', 'Jonkheer']
    df_copy['Title'] = df_copy['Name'].map(lambda x: substrings_in_string(x, title_list))

    # 处理特殊的称呼，全处理成mr, mrs, miss, master
    def replace_titles(x):
        title = x['Title']
        if title in ['Mr', 'Don', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col']:
            return 'Mr'
        elif title in ['Master']:
            return 'Master'
        elif title in ['Countess', 'Mme', 'Mrs']:
            return 'Mrs'
        elif title in ['Mlle', 'Ms', 'Miss']:
            return 'Miss'
        elif title == 'Dr':
            if x['Sex'] == 'Male':
                return 'Mr'
            else:
                return 'Mrs'
        elif title == '':
            if x['Sex'] == 'Male':
                return 'Master'
            else:
                return 'Miss'
        else:
            return title

    df_copy['Title'] = df_copy.apply(replace_titles, axis=1)

    # 看看家族是否够大，咳咳
    df_copy['Family_Size'] = df_copy['SibSp'] + df_copy['Parch']
    df_copy['Family'] = df_copy['SibSp'] * df_copy['Parch']

    df_copy.loc[(df_copy.Fare.isnull()) & (df_copy.Pclass == 1), 'Fare'] = np.median(
        df_copy[df_copy['Pclass'] == 1]['Fare'].dropna())
    df_copy.loc[(df_copy.Fare.isnull()) & (df_copy.Pclass == 2), 'Fare'] = np.median(
        df_copy[df_copy['Pclass'] == 2]['Fare'].dropna())
    df_copy.loc[(df_copy.Fare.isnull()) & (df_copy.Pclass == 3), 'Fare'] = np.median(
        df_copy[df_copy['Pclass'] == 3]['Fare'].dropna())

    df_copy['Gender'] = df_copy['Sex'].map({'female': 0, 'male': 1}).astype(int)

    df_copy['AgeFill'] = df_copy['Age']
    mean_ages = np.zeros(4)
    mean_ages[0] = np.average(df_copy[df_copy['Title'] == 'Miss']['Age'].dropna())
    mean_ages[1] = np.average(df_copy[df_copy['Title'] == 'Mrs']['Age'].dropna())
    mean_ages[2] = np.average(df_copy[df_copy['Title'] == 'Mr']['Age'].dropna())
    mean_ages[3] = np.average(df_copy[df_copy['Title'] == 'Master']['Age'].dropna())
    df_copy.loc[(df_copy.Age.isnull()) & (df_copy.Title == 'Miss'), 'AgeFill'] = mean_ages[0]
    df_copy.loc[(df_copy.Age.isnull()) & (df_copy.Title == 'Mrs'), 'AgeFill'] = mean_ages[1]
    df_copy.loc[(df_copy.Age.isnull()) & (df_copy.Title == 'Mr'), 'AgeFill'] = mean_ages[2]
    df_copy.loc[(df_copy.Age.isnull()) & (df_copy.Title == 'Master'), 'AgeFill'] = mean_ages[3]

    df_copy['AgeCat'] = df_copy['AgeFill']
    df_copy.loc[(df_copy.AgeFill <= 10), 'AgeCat'] = 'child'
    df_copy.loc[(df_copy.AgeFill > 60), 'AgeCat'] = 'aged'
    df_copy.loc[(df_copy.AgeFill > 10) & (df_copy.AgeFill <= 30), 'AgeCat'] = 'adult'
    df_copy.loc[(df_copy.AgeFill > 30) & (df_copy.AgeFill <= 60), 'AgeCat'] = 'senior'

    df_copy.Embarked = df_copy.Embarked.fillna('S')

    df_copy.loc[df_copy.Cabin.isnull(), 'Cabin'] = 0.5
    df_copy.loc[~df_copy.Cabin.isnull(), 'Cabin'] = 1.5

    df_copy['Fare_Per_Person'] = df_copy['Fare'] / (df_copy['Family_Size'] + 1)

    # Age times class

    df_copy['AgeClass'] = df_copy['AgeFill'] * df_copy['Pclass']
    df_copy['ClassFare'] = df_copy['Pclass'] * df_copy['Fare_Per_Person']

    df_copy['HighLow'] = df_copy['Pclass']
    df_copy.loc[(df_copy.Fare_Per_Person < 8), 'HighLow'] = 'Low'
    df_copy.loc[(df_copy.Fare_Per_Person >= 8), 'HighLow'] = 'High'

    le.fit(df_copy['Sex'])
    x_sex = le.transform(df_copy['Sex'])
    df_copy['Sex'] = x_sex.astype(np.float)

    le.fit(df_copy['Ticket'])
    x_ticket = le.transform(df_copy['Ticket'])
    df_copy['Ticket'] = x_ticket.astype(np.float)

    le.fit(df_copy['Title'])
    x_title = le.transform(df_copy['Title'])
    df_copy['Title'] = x_title.astype(np.float)

    le.fit(df_copy['HighLow'])
    x_hl = le.transform(df_copy['HighLow'])
    df_copy['HighLow'] = x_hl.astype(np.float)

    le.fit(df_copy['AgeCat'])
    x_age = le.transform(df_copy['AgeCat'])
    df_copy['AgeCat'] = x_age.astype(np.float)

    le.fit(df_copy['Embarked'])
    x_emb = le.transform(df_copy['Embarked'])
    df_copy['Embarked'] = x_emb.astype(np.float)

    df_copy = df_copy.drop(['PassengerId', 'Name', 'Age', 'Cabin'], axis=1)  # remove Name,Age and PassengerId

    return df_copy


# 读取数据
traindf = pd.read_csv(train_file)
# 清洗数据
df = clean_and_munge_data(traindf)
# #######################################formula################################

formula_ml = 'Survived~Pclass+C(Title)+Sex+C(AgeCat)+Fare_Per_Person+Fare+Family_Size'

y_train, x_train = dmatrices(formula_ml, data=df, return_type='dataframe')
y_train = np.asarray(y_train).ravel()
print(y_train.shape, x_train.shape)

# 选择训练和测试集
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=seed)
# 初始化分类器
clf = RandomForestClassifier(n_estimators=500, criterion='entropy', max_depth=5, min_samples_split=1.0,
                             min_samples_leaf=1, max_features='auto', bootstrap=False, oob_score=False, n_jobs=1,
                             random_state=seed,
                             verbose=0)

# grid search找到最好的参数
param_grid = dict()
# 创建分类pipeline
pipeline = Pipeline([('clf', clf)])
grid_search = GridSearchCV(pipeline, param_grid=param_grid, verbose=3, scoring='accuracy',
                           cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, train_size=None,
                                                     random_state=seed)).fit(X_train, Y_train)
# 对结果打分
print("Best score: %0.3f" % grid_search.best_score_)
print(grid_search.best_estimator_)
report(grid_search.grid_scores_)

print('-----grid search end------------')
print('on all train set')
scores = cross_val_score(grid_search.best_estimator_, x_train, y_train, cv=3, scoring='accuracy')
print(scores.mean(), scores)
print('on test set')
scores = cross_val_score(grid_search.best_estimator_, X_test, Y_test, cv=3, scoring='accuracy')
print(scores.mean(), scores)

# 对结果打分

print(classification_report(Y_train, grid_search.best_estimator_.predict(X_train)))
print('test data')
print(classification_report(Y_test, grid_search.best_estimator_.predict(X_test)))

# save model
model_file = MODEL_PATH + 'model-rf.pkl'
joblib.dump(grid_search.best_estimator_, model_file)


# 测试集预测

# 读取数据
data_test = pd.read_csv(test_file)
# 清洗数据
df_test = clean_and_munge_data(data_test)

formula_ml = 'Pclass~Pclass+C(Title)+Sex+C(AgeCat)+Fare_Per_Person+Fare+Family_Size'
test_id, test = dmatrices(formula_ml, data=df_test, return_type='dataframe')
print(test)
predictions = grid_search.best_estimator_.predict(test)
result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
result.to_csv('titanic.csv', index=False)





