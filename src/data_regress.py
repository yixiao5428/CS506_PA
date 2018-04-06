import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LassoCV
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt


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
         'aug_y', 'sep_y', 'oct_y', 'nov_y', 'dec_y', 'last_plant_acres', 'last_harvest_acres', 'last_yield_acres',
         'last_bales', 'plant_acres', 'harvest_acres', 'yield_acres', 'bales']]

    for i in range(df.shape[0]):
        if i > 0:
            if df.iloc[i - 1, 0] == df.iloc[i, 0] and df.iloc[i - 1, 1] == df.iloc[i, 1] and df.iloc[i - 1, 3] - 1 == \
                    df.iloc[i, 3]:
                df.iloc[i, -8] = df.iloc[i - 1, -4]
                df.iloc[i, -7] = df.iloc[i - 1, -3]
                df.iloc[i, -6] = df.iloc[i - 1, -2]
                df.iloc[i, -5] = df.iloc[i - 1, -1]

    df = df.dropna()

    df = df[['jan_x', 'feb_x', 'mar_x', 'apr_x', 'may_x', 'jun_x', 'jul_x', 'aug_x', 'sep_x', 'oct_x', 'nov_x', 'dec_x',
             'jan_y', 'feb_y', 'mar_y', 'apr_y', 'may_y', 'jun_y', 'jul_y', 'aug_y', 'sep_y', 'oct_y', 'nov_y', 'dec_y',
             'last_plant_acres', 'last_harvest_acres', 'last_yield_acres', 'last_bales', 'plant_acres', 'harvest_acres',
             'yield_acres', 'bales']]

    df = df.sample(frac=1.0)
    cut_idx = int(round(0.1 * df.shape[0]))
    test_df, train_df = df.iloc[:cut_idx], df.iloc[cut_idx:]
    # print(train_df)
    # print(test_df)

    train_data_set = np.mat(train_df.values.tolist())
    test_data_set = np.mat(test_df.values.tolist())

    return train_data_set, test_data_set


def data_regression(train_data_set, test_data_set):
    """

    :param train_data_set:
    :param test_data_set:
    :return:
    """
    train_x = train_data_set[:, :27]
    # train_y_1 = train_data_set[:, 16]
    # train_y_2 = train_data_set[:, 17]
    train_y_3 = train_data_set[:, 30]
    # train_y_4 = train_data_set[:, 19]

    test_x = test_data_set[:, :27]
    # test_y_1 = test_data_set[:, 16]
    # test_y_2 = test_data_set[:, 17]
    test_y_3 = test_data_set[:, 30]
    # test_y_4 = test_data_set[:, 19]

    # feature selection
    clf = LassoCV()
    sfm = SelectFromModel(clf)
    train_x = sfm.fit_transform(train_x, train_y_3)
    test_x = sfm.transform(test_x)
    # print(test_x.shape)

    # Neural Network Regression
    regress_neural = MLPRegressor().fit(train_x, train_y_3)
    prediction = regress_neural.predict(test_x)

    score = regress_neural.score(test_x, test_y_3)
    print(score)

    plt.plot(range(len(test_x)), test_y_3, 'ro', alpha=0.5, label='True Value')
    plt.plot(range(len(test_x)), prediction, 'bo', alpha=0.5, label='Predict Value')
    plt.legend(loc='best')
    plt.title('True and Predict Value of Neural Network Regression')
    plt.show()

    # Polynomial Regression
    quadratic_feature = PolynomialFeatures(degree=2)
    train_x_quadratic = quadratic_feature.fit_transform(train_x)
    test_x_quadratic = quadratic_feature.transform(test_x)
    regress_quadratic = LinearRegression().fit(train_x_quadratic, train_y_3)

    prediction = regress_quadratic.predict(test_x_quadratic)

    score = regress_quadratic.score(test_x_quadratic, test_y_3)
    print(score)

    plt.plot(range(len(test_x_quadratic)), test_y_3, 'ro', alpha=0.5, label='True Value')
    plt.plot(range(len(test_x_quadratic)), prediction, 'bo', alpha=0.5, label='Predict Value')
    plt.legend(loc='best')
    plt.title('True and Predict Value of Polynomial Regression')
    plt.show()

    return None


def main():
    """

    :return:
    """
    filename = '../data_sets/precipitation_temperature_production.csv'
    train_data_set, test_data_set = data_pre_process(filename)
    data_regression(train_data_set, test_data_set)


main()
