import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def data_pre_process(file_path):
    """

    :param file_path:
    :return:
    """
    df = pd.read_csv(file_path)
    df = df[['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec',
             'plant_acres', 'harvest_acres', 'yield_acres', 'bales']]

    df = df.sample(frac=1.0)
    cut_idx = int(round(0.2 * df.shape[0]))
    test_df, train_df = df.iloc[:cut_idx], df.iloc[cut_idx:]
    # print(train_df)
    # print(test_df)

    train_data_set = np.mat(train_df.values.tolist())
    test_data_set = np.mat(test_df.values.tolist())
    # print(data_set)

    return train_data_set, test_data_set


def data_regression(train_data_set, test_data_set):
    """

    :param train_data_set:
    :param test_data_set:
    :return:
    """
    train_x = train_data_set[:, :12]
    train_y_1 = train_data_set[:, 12]
    train_y_2 = train_data_set[:, 13]
    train_y_3 = train_data_set[:, 14]
    train_y_4 = train_data_set[:, 15]

    test_x = test_data_set[:, :12]
    test_y_1 = test_data_set[:, 12]
    test_y_2 = test_data_set[:, 13]
    test_y_3 = test_data_set[:, 14]
    test_y_4 = test_data_set[:, 15]

    # model = LinearRegression().fit(train_x, train_y_3)
    #
    #
    #
    # predictions = model.predict(test_x)
    # for i, prediction in enumerate(predictions):
    #     print('Predicted: %s, Target: %s' % (prediction, test_y_3[i]))
    # print('R-squared: %.2f' % model.score(test_x, test_y_3))

    # quadratic_featurizer = PolynomialFeatures(degree=2)
    # X_train_quadratic = quadratic_featurizer.fit_transform(X_train)
    # X_test_quadratic = quadratic_featurizer.transform(X_test)
    # regressor_quadratic = LinearRegression()
    # regressor_quadratic.fit(X_train_quadratic, y_train)
    # xx_quadratic = quadratic_featurizer.transform(xx.reshape(xx.shape[0], 1))
    # plt.plot(xx, regressor_quadratic.predict(xx_quadratic), 'r-')
    # plt.show()

    quadratic_feature = PolynomialFeatures(degree=4)
    train_x_quadratic = quadratic_feature.fit_transform(train_x)
    test_x_quadratic = quadratic_feature.transform(test_x)
    regress_quadratic = LinearRegression().fit(train_x_quadratic, train_y_3)

    score = regress_quadratic.score(test_x_quadratic, test_y_3)
    print(score)

    return None


def main():
    """

    :return:
    """
    filename = '../data_sets/temperature_production.csv'
    train_data_set, test_data_set = data_pre_process(filename)
    data_regression(train_data_set, test_data_set)


main()
