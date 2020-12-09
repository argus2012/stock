import pandas as pd
import numpy as np
import pandas_ta as ta
import joblib

from yahoo_fin import stock_info as si

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import accuracy_score

from numpy import mean
from numpy import median
from numpy import percentile

from datetime import datetime
import datetime
import os
from pathlib import Path

from src.IO.storage_tools import get_model_from_bucket, upload_file_to_bucket

para_test_size = 0.2
para_estimator = 170
para_max_depth = 5
para_min_leaf = 1
para_random_state = 1

min_df_length = 60
bucket_name = 'model_bucket_ycng_228_raymondmin_110'


def cal_confidence_interval(data_list):
    data_median = median(data_list)
    data_mean = mean(data_list)

    # calculate 95% confidence intervals (100 - alpha)
    alpha = 5.0
    # calculate lower percentile (e.g. 2.5)
    lower_p = alpha / 2.0
    # retrieve observation at lower percentile
    data_lower = max(0.0, percentile(data_list, lower_p))

    # calculate upper percentile (e.g. 97.5)
    upper_p = (100 - alpha) + (alpha / 2.0)

    # retrieve observation at upper percentile
    data_upper = min(1.0, percentile(data_list, upper_p))

    return data_lower, data_median, data_upper, data_mean


def create_historical_data(df_stock, nlags=10):
    df_resampled = df_stock.copy()
    lags_col_names = []
    for i in range(nlags + 1):
        df_resampled['lags_' + str(i)] = df_resampled['close'].shift(i)
        lags_col_names.append('lags_' + str(i))
    df = df_resampled[lags_col_names]

    return df


def create_feature(df):
    """
        Create some more features
    """
    # Preprocessing data
    df.drop(columns=['ticker'], inplace=True)

    # Dealing with missing data:
    # if drop na here, it will cause some features discontinuities. Need to be improved to some extend
    df.dropna(inplace=True)

    # Length of data is not enough to create additional features
    if len(df) < min_df_length:
        return None

    # Add lag data (default 10 lags)
    df_history = create_historical_data(df)
    df = pd.concat([df, df_history], axis=1)

    # Add technical analysis indicators
    # MACD
    df[['macd_fast', 'macd_slow', 'macd_signal']] = ta.macd(df['close'])

    # PVT
    df['pvt'] = ta.pvt(df['close'], df['volume'])

    # Rate of Change
    roc_list = [1, 5, 10]
    for roc in roc_list:
        df['roc_' + str(roc)] = ta.roc(df['close'], length=roc)

    # SMA
    ma_list = [5, 10, 20, 50]
    for ma in ma_list:
        df['MA_' + str(ma)] = ta.sma(df['close'], length=ma)

    # EMA
    ema_list = [12, 26]
    for ma in ema_list:
        df['EMA_' + str(ma)] = ta.ema(df['close'], length=ma)

    # Optional features
    # DIFF
    df['diff'] = ta.ema(df['close'], length=12) - ta.ema(df['close'], length=26)

    # Return
    df['return'] = ta.percent_return(df['close'])
    df['return_30'] = ta.percent_return(df['close'], length=30)

    df['log_return'] = ta.log_return(df['close'])

    rsi_list = [3, 20]
    for rsi in rsi_list:
        df['RSI_' + str(rsi)] = ta.rsi(df['close'], length=rsi)

    df['bias'] = ta.bias(df['close'])

    # Delta Open Close, High Low
    df['close_open'] = df['close'] - df['open']
    df['high_low'] = df['high'] - df['low']

    # df['CMO'] = ta.cmo(df['close'], 14)

    # cci_list = [14, 24]
    # for cci in cci_list:
    #     df['CCI_' + str(cci)] = ta.cci(df['high'], df['low'], df['close'], cci)

    # df['AD'] = ta.ad(df['high'], df['low'], df['close'], df['volume'], df['open'], 14)

    # df['psy'] = ta.psl(df['close'], length=20)

    # # cross_pvt_ema
    # df['cross_pvt_ema'] = ta.cross(ta.pvt(df['close'], df['volume']),
    #                                ta.ema(df['close'], length=10))
    #
    # # cross_ma5_ma10
    # df['cross_ma5_ma10'] = ta.cross_value(ta.sma(df['close'], length=5),
    #                                       ta.sma(df['close'], length=10))
    # End of Optional features

    # Dealing with missing data again, by the process of creating features
    df.dropna(inplace=True)

    if len(df) == 0:
        return None

    return df


def get_sector_list():
    sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    return sp500.drop_duplicates('GICS Sector')['GICS Sector'].values.tolist()


def get_ticker_sector(ticker):
    ticker_sector = None
    sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    symbols = sp500['Symbol'].values.tolist()
    sectors = sp500['GICS Sector'].values.tolist()

    for i in range(len(symbols)):
        if symbols[i] == ticker:
            ticker_sector = sectors[i]

    return ticker_sector


def get_ticker_list_by_sector(ticker_sector):
    sp500 = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]
    sector_list = sp500[sp500['GICS Sector'] == ticker_sector]
    return sector_list['Symbol'].tolist()


def create_X_Y(df, predict_period=1):

    # Create Y, Labeling the data
    df['label'] = np.where((df['close'].shift(-1) - df['close']) > 0, 1, 0)

    # Delete the last row without predict result
    df.drop(df.tail(1).index, inplace=True)

    # Create X, creating more features
    df = create_feature(df)

    if df is None:
        return None, None

    if predict_period > 1:

        if len(df) < predict_period:
            return None, None

        df_y = df.tail(predict_period)['label']
        # Get predict data, last prediction rows
        df_X = df.tail(predict_period).drop(columns=['label'])

    else:

        df_y = df['label']

        df_X = df.drop(columns=['label'])

    return df_X, df_y


def create_predict_data(ticker, end_date=None):

    data_df = si.get_data(ticker.replace('.', '-'), None, end_date)

    # Create X, creating the features
    data_df = create_feature(data_df)

    if data_df is None:
        return None

    # Get predict data, last prediction rows
    df_predict = data_df.tail(1)

    return df_predict


def get_or_build_model(X_train, y_train, model_name, rebuild_model=False):
    """
        Get or build the model of the specific sector name with all the data of all tickers
    """
    model_filename = f'{model_name}.pkl'
    model = get_model_from_bucket(model_filename, bucket_name)

    if rebuild_model or not model:

        model = RandomForestClassifier(n_estimators=para_estimator,
                                       max_depth=para_max_depth,
                                       min_samples_leaf=para_min_leaf,
                                       oob_score=True, random_state=para_random_state)

        model.fit(X_train, y_train)

        with open(model_filename, 'wb') as f:
            joblib.dump(model, f)
        upload_file_to_bucket(model_filename, bucket_name)

    return model


def predict_by_sector_model(ticker, predict_period=1):

    """
    Predict by sector model

    Build the model with all the tickers in the same sector.
    First, to check the sector of the ticker;
    Second, to retrieve the data of the ticker, and add some features;
    Third, to build the model with the data of all the tickers in this sector;
    Finally, to predict with this model.

    Parameters
    ----------
    ticker : string
        The ticker name of the stock.
    predict_period : int, default=1
        if predict_period=1:
            Prediction Mode, for final prediction
        if predict_period>1:
            Testing Mode, training the model with partly data,
            and predict testing with the last {predict_period} data to calculate predict BA

    Returns
    -------
    final_predict : ndarray
        The final predicted classes (1, 0 or -1).
    training_ba : float
        The balanced accuracy of training data
    test_ba : float
        The balanced accuracy of testing data
    predict_ba : float
        The balanced accuracy of predicting sample data

    """

    # Variable initialization
    df_X = pd.DataFrame()
    df_y = pd.DataFrame()
    final_predict = [-1]
    training_ba = 0
    test_ba = 0
    predict_ba = 0

    stock_sector = get_ticker_sector(ticker)
    if stock_sector is None:
        return final_predict, training_ba, test_ba, predict_ba

    ticker_list = get_ticker_list_by_sector(stock_sector)

    model_filename = stock_sector + '.pkl'

    predict_model = get_model_from_bucket(model_filename, bucket_name)

    if predict_model is None:
        for n_ticker in ticker_list:

            print(f'Downloading {n_ticker}')

            # For some ticker containing . change to -
            ticker_data = si.get_data(n_ticker.replace('.', '-'))

            if len(ticker_data) < min_df_length:
                continue

            ticker_X, ticker_y = create_X_Y(ticker_data)

            if ticker_X is None:
                continue

            if predict_period > 1:

                ticker_X.drop(ticker_X.tail(predict_period).index, inplace=True)
                ticker_y.drop(ticker_y.tail(predict_period).index, inplace=True)

            df_X = pd.concat([df_X, ticker_X])
            df_y = pd.concat([df_y, ticker_y])

        features = df_X.columns
        X = df_X[features].values

        # Dataframe to array
        y = df_y.values.ravel()

        # Split training data set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=para_test_size,
                                                            random_state=para_random_state)

        predict_model = get_or_build_model(X_train, y_train, stock_sector, True)

        predict_y = predict_model.predict(X_test)
        training_ba = accuracy_score(y_train, predict_model.predict(X_train))
        test_ba = balanced_accuracy_score(y_test, predict_y)
        print(f'Training BA: {training_ba}\nTest BA: {test_ba}')

    # Make the final prediction
    if predict_period == 1:

        predict_data = create_predict_data(ticker)

        if predict_data is None:
            return final_predict, training_ba, test_ba, predict_ba

        final_predict = predict_model.predict(predict_data)
        print(f'{ticker} Sector Predict: {final_predict[-1]}')

    # Testing Mode: Make the predictions of the latest days to calculate the predict BA
    else:

        data_df = si.get_data(ticker.replace('.', '-'))
        predict_data, predict_y = create_X_Y(data_df, predict_period)

        if predict_data is None:
            return final_predict, training_ba, test_ba, predict_ba

        final_predict = predict_model.predict(predict_data)
        predict_ba = balanced_accuracy_score(predict_y, final_predict)
        print(f'{ticker} Sector Predict BA: {predict_ba}')

    return final_predict, training_ba, test_ba, predict_ba


def update_sector_models():
    """
        Update all the sector models with the prediction mode or testing mode
    """
    list_ticker_snp_500 = si.tickers_sp500()

    data_csv = pd.DataFrame(columns=('sector', 'predict', 'training_ba', 'test_ba', 'predict_ba'),
                            index=list_ticker_snp_500)

    for ticker in list_ticker_snp_500:
        final_predict, training_ba, test_ba, predict_ba = predict_by_sector_model(ticker)
        # final_predict, training_ba, test_ba, predict_ba = sector_predict(ticker, 200)
        data_csv.loc[ticker, 'predict'] = final_predict[-1]
        data_csv.loc[ticker, 'training_ba'] = training_ba
        data_csv.loc[ticker, 'test_ba'] = test_ba
        data_csv.loc[ticker, 'predict_ba'] = predict_ba
        data_csv.loc[ticker, 'sector'] = get_ticker_sector(ticker)

    # For output the csv file
    str_now = datetime.datetime.now().strftime('%Y_%m_%d %H_%M_%S')
    data_dir = Path('C:\\Users\\argus\\Workplace\\10.data')
    if os.path.exists(data_dir):
        data_csv.to_csv(data_dir / f'tickers_{str_now}.csv')

    return f'Update Successfully.'


def update_system():
    """
        Update all the data and models in the system (Under developing)
    """

    # Clean all the models in Google bucket

    # Update sector models of all tickers
    update_sector_models()

    return


def stock_predict(ticker):
    """
        Make the final prediction and output the final result
    """
    # final_predict, training_ba, test_ba, predict_ba = predict_by_ticker_model(ticker)
    final_predict, training_ba, test_ba, predict_ba = predict_by_sector_model(ticker)
    output = final_predict[-1]

    if output == -1:
        final_predict_string = 'Invalid Ticker'
    elif output > 0:
        final_predict_string = 'Buy'
    else:
        final_predict_string = 'Sell'

    return final_predict_string
