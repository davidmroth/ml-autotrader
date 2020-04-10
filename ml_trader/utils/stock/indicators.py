import numpy as np

def get_technical_indicators( min_scaler, histories_normalized ):
    def calc_ema( values, time_period ):
        if len( values ) < 13:
            raise Exception( "ERROR: Please make sure 'config.history_points' is not less than 13. This is requried for exponential-moving-average (ema) formula.")

        # https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
        sma = np.mean( values[:, 3] )
        ema_values = [sma]
        k = 2 / ( 1 + time_period )

        print( history )
        for i in range( len( history ) - time_period, len( history ) ):
            close = history[i][3]
            ema_values.append( close * k + ema_values[-1] * ( 1 - k ) )

        return ema_values[-1]

    technical_indicators = []

    for history in histories_normalized:
        sma = np.mean( history[:, 3] ) # Note since we are using history[3] we are taking the SMA (Simple Moving Average) of the closing price
        macd = calc_ema( history, 12 ) - calc_ema( history, 26 ) # Moving average convergence divergence
        technical_indicators.append( np.array( [sma] ) )
        # technical_indicators.append(np.array([sma,macd,]))

    technical_indicators = np.array( technical_indicators )
    return min_scaler.fit_transform( technical_indicators )
