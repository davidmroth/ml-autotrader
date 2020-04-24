import talib
from talib import MA_Type
import numpy as np
import pandas as pd
import math as m

import ml_trader.utils.data.meta as meta
import ml_trader.config as config


def EMA( values, time_period ):
    if len( values ) < 13:
        raise Exception( "ERROR: Please make sure 'config.history_points' is not less than 13. This is requried for exponential-moving-average (ema) formula.")

    # https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
    sma = np.mean( values[:, meta.column_index['close']] )
    ema_values = [sma]
    k = 2 / ( 1 + time_period )

    for i in range( len( values ) - time_period, len( values ) ):
        close = values[i][meta.column_index['close']]
        ema_values.append( close * k + ema_values[-1] * ( 1 - k ) )

    return ema_values[-1]

#Moving Average
def MA( df, col, n ):
    MA = df.iloc[:,col].head( n ).mean()
    return MA

#Rate of Change
def ROC( df, col, n ):
    M = df[col].iloc[-1]
    N = df[col].iloc[50 - n]
    return ( M - N ) / N * 100

def get_technical_indicators_talib( min_scaler, ohlcv, len ):
    tech_ind = pd.DataFrame()

    tech_ind["macd"], tech_ind["sigal"], tech_ind["hist"] = talib.MACD( ohlcv.close )
    tech_ind["ma10"] = talib.MA ( ohlcv.close, timeperiod=10 )
    #tech_ind["ma30"] = talib.MA( ohlcv.close, timeperiod=30 )
    #tech_ind["mom"] = talib.MOM( ohlcv.close, timeperiod=5 )
    #tech_ind["upper"], tech_ind["middle"], tech_ind["lower"] = talib.BBANDS( ohlcv.close, matype=MA_Type.T3 )
    #tech_ind['RSI'] = talib.RSI( ohlcv.close, timeperiod=14 )

    tech_ind = tech_ind.tail( len )

    # Auto set configuration
    config.technical_indictors_input_size = tech_ind.shape[1]
    return min_scaler.fit_transform( np.array( tech_ind ) )

def get_technical_indicators( min_scaler, ohlcv_histories ):
    technical_indicators = []

    for history in ohlcv_histories:
        history_df = pd.DataFrame( history )

        '''
        SMA (Simple Moving Average) of the closing price
        '''
        sma50 = MA( history_df, meta.column_index['close'], 50 )

        '''
        ROC (Rate of change)
        '''
        roc = ROC( history_df, meta.column_index['close'], 4 )

        '''
        Moving average convergence divergence
        '''
        macd = EMA( history, 12 ) - EMA( history, 26 )

        #technical_indicators.append( np.array( [sma50] ) )
        technical_indicators.append( np.array( [sma50, macd, roc,] ) )

    return min_scaler.fit_transform( np.array( technical_indicators ) )
