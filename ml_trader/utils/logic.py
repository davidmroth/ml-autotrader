import datetime
import numpy as np

import ml_trader.config as config
import ml_trader.utils.data.meta as meta
import ml_trader.utils.analysis.insight as insight


def do_trade( model, packed_trade_data, y_normaliser ):
    # Initialize
    start = 0
    end = -1
    buys = []
    sells = []
    thresh = config.trade_threshold #in dollars?
    predicted_price_yhat = np.array( [None] )

    # Trasnform data
    ohlcv_test, tech_ind_test, y_test_dates = packed_trade_data

    #TODO: Do I need start and end? If 0 ='s the begining the array, and -1
    # ='s the 'end' of array, which whould mean the whole array
    trade_data = zip( ohlcv_test[start: end], tech_ind_test[start: end], y_test_dates[start: end] )

    for ohlcv, ind, date in trade_data:
        # Get the last 'close' price in history
        normalised_price_today = [[ohlcv[-1][meta.column_index[meta.label_column]]]]

        # Get actual price (not normalized)
        price_today = y_normaliser.inverse_transform( normalised_price_today )

        # Get predicted 'close' price for the next day
        predicted_price_tomorrow = np.squeeze( model.predict( [[ohlcv], [ind]] ) )

        # Get price difference
        delta = predicted_price_tomorrow - price_today

        # Buy / Sell Logic
        if delta > thresh:
            buys.append( ( date, price_today[0][0] ) )

        elif delta < -thresh:
            sells.append( ( date, price_today[0][0] ) )

        #print( "X %s, ind: %s " % ( x, ind ) )

        # For plotting: append predicted prices into an array
        predicted_price_yhat = np.append( predicted_price_yhat, predicted_price_tomorrow )

        # Print insight
        insight.get_trade_insight( date, price_today, predicted_price_tomorrow )

    return ( predicted_price_yhat, buys, sells )
