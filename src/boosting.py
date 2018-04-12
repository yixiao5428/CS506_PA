import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor

def data_pre_process(file_path):
    """
    :param file_path:
    :return:
    """
    df = pd.read_csv(file_path)
    df = df.replace(-9999, np.nan)
    df = df.dropna()

    df['last_plant_acres'] = np.nan
    df['last_harvest_acres'] = np.nan
    df['last_yield_acres'] = np.nan
    df['last_bales'] = np.nan

    df = df[
        ['state', 'county', 'station_x', 'year', 'jan_x', 'feb_x', 'mar_x', 'apr_x', 'may_x', 'jun_x', 'jul_x', 'aug_x',
         'sep_x', 'oct_x', 'nov_x', 'dec_x', 'station_y', 'jan_y', 'feb_y', 'mar_y', 'apr_y', 'may_y', 'jun_y', 'jul_y',
         'aug_y', 'sep_y', 'oct_y', 'nov_y', 'dec_y', 'total_percent_lost', 'last_plant_acres', 'last_harvest_acres',
         'last_yield_acres', 'last_bales', 'plant_acres', 'harvest_acres', 'yield_acres', 'bales']]

    df = df.sort_values(by=['state', 'county', 'year'])

    for i in range(df.shape[0]):
        if i > 0:
            if df.iloc[i - 1, 0] == df.iloc[i, 0] and df.iloc[i - 1, 1] == df.iloc[i, 1] and df.iloc[i - 1, 3] == \
                    df.iloc[i, 3] - 1:
                df.iloc[i, -8] = df.iloc[i - 1, -4]
                df.iloc[i, -7] = df.iloc[i - 1, -3]
                df.iloc[i, -6] = df.iloc[i - 1, -2]
                df.iloc[i, -5] = df.iloc[i - 1, -1]

    # df.to_csv('./temp.csv')
    df = df.dropna()
    # print(df)

    df = df[['jan_x', 'feb_x', 'mar_x', 'apr_x', 'may_x', 'jun_x', 'jul_x', 'aug_x', 'sep_x', 'oct_x', 'nov_x', 'dec_x',
             'jan_y', 'feb_y', 'mar_y', 'apr_y', 'may_y', 'jun_y', 'jul_y', 'aug_y', 'sep_y', 'oct_y', 'nov_y', 'dec_y',
             'total_percent_lost',
             'last_plant_acres', 'last_harvest_acres', 'last_yield_acres', 'last_bales',
             'plant_acres', 'harvest_acres', 'yield_acres', 'bales']]

    df = df.sample(frac=1.0)
    return pd.concat([df.iloc[:,:24],df.iloc[:,27]],axis=1),df.iloc[:,-2]

def preprocess():
    df = pd.read_csv("/Users/tony/PycharmProjects/learn_for_506/CS506_PA/data_sets/disease_precipitation_temperature_production.csv")

    label = df.iloc[:,18]
            # /(1-df.iloc[:,-1]/100)
    # print(label)

    tempe = df.iloc[:,4:16]
    prep = df.iloc[:,21:33]
    feature = [tempe, prep, label]
    feature = pd.concat(feature,axis=1)
    for i in range(feature.shape[0]):
        if i > 0:
            if feature.iloc[i - 1, 0] == feature.iloc[i, 0] and feature.iloc[i - 1, 1] == feature.iloc[i, 1] and feature.iloc[i - 1, 3] == \
                            feature.iloc[i, 3] - 1:
                feature.iloc[i, -8] = feature.iloc[i - 1, -4]
                feature.iloc[i, -7] = feature.iloc[i - 1, -3]
                feature.iloc[i, -6] = feature.iloc[i - 1, -2]
                feature.iloc[i, -5] = feature.iloc[i - 1, -1]
    feature = feature.replace(-9999,np.nan)
    feature = feature.dropna()
    tempe = feature.iloc[:,:12]
    prep = feature.iloc[:,12:24]
    label = feature.iloc[:,-1]
    tmax = tempe.max().max()
    tmin = tempe.min().min()
    pmax = prep.max().max()
    pmin = prep.min().min()
    tempe = 1 + (4 * (tempe - tmin) / (tmax - tmin))
    prep = 1 + (4 * (prep - pmin) / (pmax - pmin))
    feature = [tempe, prep]
    feature = pd.concat(feature, axis=1)
    return feature,label

def GaussianReg(X_train, X_test, y_train, y_test):
    regr = GaussianProcessRegressor()
    regr.fit(X_train, y_train)
    y_predict = regr.predict(X_test)
    squared_error = mean_squared_error(y_test, y_predict)
    score = regr.score(X_test,y_test)
    return squared_error,score

def DecisionTreeReg(X_train, X_test, y_train, y_test):
    regr = DecisionTreeRegressor(max_depth=4)
    regr.fit(X_train, y_train)
    y_predict = regr.predict(X_test)
    squared_error = mean_squared_error(y_test, y_predict)
    score = regr.score(X_test, y_test)
    return squared_error, score

def LinearReg(X_train, X_test, y_train, y_test):
    regr = linear_model.Lasso(alpha = 0.1)
    regr.fit(X_train, y_train)
    y_predict = regr.predict(X_test)
    squared_error = mean_squared_error(y_test, y_predict)
    score = regr.score(X_test, y_test)
    return squared_error, score

# 39505.52
def AdaBoostGaussianReg(X_train, X_test, y_train, y_test):
    rng = np.random.RandomState(1)
    regr = AdaBoostRegressor(GaussianProcessRegressor(),
                               n_estimators=300, random_state=rng)
    regr.fit(X_train, y_train)
    y_predict = regr.predict(X_test)
    squared_error = mean_squared_error(y_test, y_predict)
    score = regr.score(X_test, y_test)
    return squared_error, score

# (21696.872541550074, 0.80262771057683124)
def GradiantBoostReg(X_train, X_test, y_train, y_test):
    params = {'n_estimators': 100, 'max_depth': 4, 'min_samples_split': 2,
              'learning_rate': 0.2, 'loss': 'ls'}
    regr = GradientBoostingRegressor(**params)
    regr.fit(X_train, y_train)
    y_predict = regr.predict(X_test)
    squared_error = mean_squared_error(y_test, y_predict)
    score = regr.score(X_test, y_test)
    return squared_error, score

# 38645.06
def RandomForestReg(X_train, X_test, y_train, y_test):
    regr = RandomForestRegressor(n_estimators=20)
    regr.fit(X_train, y_train)
    y_predict = regr.predict(X_test)
    squared_error = mean_squared_error(y_test, y_predict)
    score = regr.score(X_test, y_test)
    return squared_error, score

def regression():
    filename = '../data_sets/disease_precipitation_temperature_production.csv'
    featrue,label = data_pre_process(filename)
    print(featrue)
    # e = np.random.normal(size=featrue.shape[0])
    # label=label+e
    X_train, X_test, y_train, y_test = train_test_split(featrue, label, test_size=0.1, random_state=0)

    print(RandomForestReg(X_train, X_test, y_train, y_test))



    # show plot and squared error with score
    # regr.fit(X_train,y_train)
    # diabetes_y_pred = regr.predict(X_test)
    # score = regr.score(X_test,diabetes_y_pred)
    # print("Mean squared error: %.2f" % mean_squared_error(y_test, diabetes_y_pred),"score=",score)
    # plt.plot(range(225), diabetes_y_pred, color='blue', linewidth=3)
    # plt.plot(range(225), y_test, color='red', linewidth=3)
    # plt.show()

    # scores = cross_val_score(regr, feature.iloc[:,:-1], feature.iloc[:,-1], cv=10)
    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

regression()