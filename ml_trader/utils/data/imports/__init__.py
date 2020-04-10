import pandas as pd
import numpy as np

import ml_trader.config as config
import ml_trader.utils.file as file
import ml_trader.utils.data.imports.get as get
import ml_trader.utils.stock.indicators as stock_indicators

from pprint import pprint
from sklearn import preprocessing

from ml_trader.config import history_points
from ml_trader.utils.compute import earnings


def from_csv():
    data = get.dataset()

    # The first day of trading that stock often looked anomalous due to the massively high volume (IPO).
    # This inflated max volume value also affected how other volume values in the dataset were scaled when normalising the data,
    # so we drop the oldest data point out of every set)

    # Transform data
    #date_time = pd.DataFrame( np.array( data['date'], dtype='datetime64' ) )
    date_time = pd.DataFrame( data['date'], dtype='datetime64' )
    date_time = date_time.drop( 0, axis=0 ).reset_index( drop=True ) # Drop one day (oldest) and reset index
    date_time = date_time.values # Convert to numpy array

    data = data.drop( 'date', axis=1 )
    data = data.drop( 0, axis=0 ) # Drop one day (oldest)
    data = data.values

    '''
    # DEBUG
    df = pd.DataFrame( data )
    dt = pd.DataFrame( date_time )
    print( df.head() )
    print( df.tail() )
    print( dt.head() )
    print( dt.tail() )
    '''

    # Normalise the data — scale it between 0 and 1 — to improve how quickly our network converges
    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform( data )

    '''
    # DEBUG
    df = pd.DataFrame( data )
    dn = pd.DataFrame( data_normalised )
    print( "Data:", df.head() )
    print( "Normalized:", dn.head() )
    exit()
    '''

    # using the last {history_points} open close high low volume data points, predict the next  value
    ohlcv_histories_normalised       = np.array( [data_normalised[i:i + history_points].copy() for i in range( len( data_normalised ) - history_points )] )
    next_day_close_values_normalised = np.array( [data_normalised[:, 3][i + history_points].copy() for i in range( len( data_normalised ) - history_points )] )
    next_day_close_values_normalised = np.expand_dims( next_day_close_values_normalised, -1 )

    next_day_close_values = np.array( [data[:, 3][i + history_points].copy() for i in range( len( data ) - history_points)] )
    next_day_close_values = np.expand_dims( next_day_close_values, -1 )

    date_time = np.array( [date_time[:, 0][i + history_points].copy() for i in range( len( data_normalised ) - history_points )] )
    date_time = np.expand_dims( date_time, -1 )

    y_normaliser = preprocessing.MinMaxScaler()
    y_normaliser.fit( next_day_close_values )

    '''
    technical_indicators = []

    for his in ohlcv_histories_normalised:
        sma = np.mean( his[:, 3] ) # Note since we are using his[3] we are taking the SMA (Simple Moving Average) of the closing price
        macd = indicator.calc_ema( his, 12 ) - indicator.calc_ema( his, 26 ) # Moving average convergence divergence
        technical_indicators.append( np.array( [sma] ) )
        # technical_indicators.append(np.array([sma,macd,]))

    technical_indicators = np.array( technical_indicators )

    tech_ind_scaler = preprocessing.MinMaxScaler()
    technical_indicators_normalised = tech_ind_scaler.fit_transform( technical_indicators )
    '''
    technical_indicators_normalised = stock_indicators.get_technical_indicators( preprocessing.MinMaxScaler(), ohlcv_histories_normalised )

    assert ohlcv_histories_normalised.shape[0] == next_day_close_values_normalised.shape[0] == technical_indicators_normalised.shape[0]
    return date_time, ohlcv_histories_normalised, technical_indicators_normalised, next_day_close_values_normalised, next_day_close_values, y_normaliser
