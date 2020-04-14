import numpy as np

import ml_trader.utils.data.meta as meta

def _calc_ema( values, time_period ):
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

def get_technical_indicators( min_scaler, ohlcv_histories ):
    technical_indicators = []

    for history in ohlcv_histories:
        '''
        SMA (Simple Moving Average) of the closing price
        '''
        sma50 = np.mean( history[:, meta.column_index['close']] )

        '''
        Moving average convergence divergence
        '''
        macd = _calc_ema( history, 12 ) - _calc_ema( history, 26 )
        #technical_indicators.append( np.array( [sma50] ) )
        technical_indicators.append( np.array( [sma50,macd,] ) )

    return min_scaler.fit_transform( np.array( technical_indicators ) )
