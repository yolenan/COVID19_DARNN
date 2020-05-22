import numpy as np
import pandas as pd
from sklearn import preprocessing

cut_day = 110


def data_clean(data_lst):
    data_lst = data_lst.replace([np.inf], 0)
    data_lst[data_lst.isnull()] = 0
    data_lst[data_lst < 0] = 0
    return data_lst.values


def read_data(disease_path, index_path):
    df = pd.read_csv(disease_path, usecols=['confirmed', 'recovered', 'deaths'])
    # df = pd.read_csv(disease_path, usecols=['beta', 'sigma', 'gamma'])
    df_index = pd.read_csv(index_path)
    del df_index['date']
    # df2 = df.drop(["Date", 'beta', 'sigma', 'gamma','add_confirm','add_recover','add_deaths',''], axis=1)
    X = df_index.values
    beta = data_clean(df['confirmed'])
    sigma = data_clean(df['recovered'])
    gamma = data_clean(df['deaths'])
    print(len(beta))
    # beta = data_clean(df['beta'])
    # sigma = data_clean(df['sigma'])
    # gamma = data_clean(df['gamma'])
    # x_scaler = preprocessing.MinMaxScaler()
    # beta_scaled = x_scaler.fit_transform(beta)
    # gamma_scaled = x_scaler.fit_transform(gamma)
    # print(beta, beta_scaled)
    # print(gamma)
    # beta = df['beta'].values
    # sigma = df['sigma'].values
    # gamma = df['gamma'].values
    # beta = df['Confirmed'].values
    # sigma = df['Recovered'].values
    # gamma = df['Deaths'].values
    return [X, beta, sigma, gamma]


# def read_data(input_path):
#     df = pd.read_csv(input_path)
#     df2 = df.drop(["Date", 'beta', 'sigma', 'gamma'], axis=1)
#     X = df2.values
#     beta = df['beta'].values
#     sigma = df['sigma'].values
#     gamma = df['gamma'].values
#     return X, beta, sigma, gamma


def train_val_test_split(X, beta, is_Val, sigma, gamma):
    # Train set
    X_train = X[:cut_day, :]
    beta_train = beta[:cut_day]
    sigma_train = sigma[:cut_day]
    gamma_train = gamma[:cut_day]
    # Test set
    X_test = X[cut_day:, :]
    beta_test = beta[cut_day:]
    sigma_test = sigma[cut_day:]
    gamma_test = gamma[cut_day:]

    # Val set
    if is_Val:
        X_val = X[cut_day:cut_day + 7, :]
        beta_val = beta[cut_day:cut_day + 7]
        sigma_val = sigma[cut_day:cut_day + 7]
        gamma_val = gamma[cut_day:cut_day + 7]
    else:
        X_val = np.zeros_like(X_test)
        beta_val = np.zeros_like(beta_test)
        sigma_val = np.zeros_like(sigma_test)
        gamma_val = np.zeros_like(gamma_test)

    return X_train, beta_train, X_test, beta_test, X_val, beta_val, sigma_train, gamma_train, sigma_test, gamma_test, sigma_val, gamma_val
